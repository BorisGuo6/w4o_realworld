import os
import sys
from typing import Dict, Tuple

import torch
import numpy as np
from pytorch3d import transforms

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'graspnet_baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet_baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet_baseline', 'utils'))

# from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image, transform_point_cloud

import imageio
from PIL import Image
import open3d as o3d
from scipy.spatial import KDTree

class GroundedGraspNet:
    def __init__(self, ckpt_path, num_points=20000, num_view=300, collision_thresh=0.01, voxel_size=0.01, task_name=None):
        self.collision_thresh = collision_thresh
        self.num_points = num_points
        self.num_view = num_view
        self.ckpt_path = ckpt_path
        self.voxel_size = voxel_size
        self.task_name = task_name
        
        self.net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.ckpt_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.ckpt_path, self.start_epoch))
        # set model to eval mode
        self.net.eval()
        
    def get_grounded_mask(self, obs, prompt):
        from grounded_sam import grounded_segmentation, plot_detections
        depth = obs.wrist_depth
        if prompt is None:
            return (depth > 0)  
        else:
            image = obs.wrist_rgb
            image = Image.fromarray(image)
            threshold = 0.3
            
            detector_id = "IDEA-Research/grounding-dino-tiny"
            segmenter_id = "facebook/sam-vit-base"
            
            image_array, detections = grounded_segmentation(
                image=image,
                labels=prompt,
                threshold=threshold,
                polygon_refinement=True,
                detector_id=detector_id,
                segmenter_id=segmenter_id
            )
            
            plot_detections(image_array, detections, f"tmp/{self.task_name}_grounded_mask.png")
            print(f"Grounded mask saved to {self.task_name}_grounded_mask.png")
            
            return detections[0].mask.astype(np.bool)  # mask of the first object
            
    def _filter_grasps(self, grasp_group, obs, prompt):
        mask = self.get_grounded_mask(obs, prompt)
        filtered = []
        masked_pcd = obs.wrist_point_cloud[mask]
        self.distance_threshold = 0.01

        try:
            pcd_tree = KDTree(masked_pcd)
        except ValueError as e:
            # This can happen if masked_pcd becomes non-2D or has other issues
            print(f"Error creating KDTree from masked_pcd (shape: {masked_pcd.shape}): {e}. Returning no grasps.")
            return []
        
        filtered_grasps = []
        for grasp in grasp_group:
            center = grasp.translation

            distance_to_closest_point, _ = pcd_tree.query(center, k=1)

            if distance_to_closest_point < self.distance_threshold:
                filtered_grasps.append(grasp)
        
        return filtered_grasps
    
    
    def process_input(self, obs, prompt, cam_int):
        """
        Process the input data and return the point cloud and color data.
        """
        
        color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
        depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
        workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
        meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        
        # generate cloud
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # prompt = None
        # mask = self.get_grounded_mask(obs, prompt)  
        mask = (depth > 0)
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        
        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        # o3d.io.write_point_cloud("tmp/point_cloud.pcd", cloud)
        
        return end_points, cloud
    
    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        
        return gg
    
    def predict_grasps(self, obs, prompt, cam_int, cam2world):
        end_points, cloud = self.process_input(obs, prompt, cam_int)

        with torch.no_grad():
            pred = self.net(end_points)
            grasp_group = pred_decode(pred)
        gg_array = grasp_group[0].detach().cpu().numpy()
        
        # graspnet api collision detection with CoppeliaSim if imported before
        from graspnetAPI import GraspGroup
        gg = GraspGroup(gg_array)
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        
        gg.nms()
        gg.sort_by_score()
        
        # gg = gg[:50]
        
        filtered_grasp_list = self._filter_grasps(gg, obs, prompt)
        
        grippers = gg.to_open3d_geometry_list()
        gg = filtered_grasp_list[:1]
        
        grippers = [g.to_open3d_geometry() for g in gg]
        width, height = 512, 512
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

        # Create a scene
        scene = renderer.scene
        scene.set_background([1, 1, 1, 1])  # white background

        # Add the point cloud
        cloud_material = o3d.visualization.rendering.MaterialRecord()
        cloud_material.shader = "defaultUnlit"
        scene.add_geometry("cloud", cloud, cloud_material)

        # Add grippers
        gripper_material = o3d.visualization.rendering.MaterialRecord()
        gripper_material.shader = "defaultUnlit"
        for i, gripper in enumerate(grippers):
            scene.add_geometry(f"gripper_{i}", gripper, gripper_material)

        # Compute scene bounds for camera placement
        bounds = renderer.scene.bounding_box
        center = bounds.get_center()
        extent = bounds.get_extent().max()

        # Define camera view
        # eye = center + np.array([0.25, 0.25, -0.25])  # move camera to the side (+X)
        eye = center + np.array([-0.25, 0.0, -0.25])
        up = np.array([0.0, 0.0, -1.0])  # -Z is up
        renderer.setup_camera(60.0, center, eye, up)
        
        # Render and save to file
        img = renderer.render_to_image()
        o3d.io.write_image(f"tmp/grasp_scene.png", img)
        print("Image saved to tmp/grasp_scene.png")
        
        # ---------- NEW: save the whole scene as a PCD -----------------
        #   1. start with the original cloud
        scene_pcd = o3d.geometry.PointCloud(cloud)

        #   2. sample each gripper mesh uniformly (1 000 pts) and append
        for mesh in grippers:
            # ensure we have a TriangleMesh (to sample); convert LineSet etc.
            if isinstance(mesh, o3d.geometry.LineSet):
                mesh = mesh.tetra_mesh()   # basic surface from lines
            pts = mesh.sample_points_uniformly(number_of_points=1000)
            scene_pcd += pts

        #   3. write to disk (PCD/PLY – change extension as you like)
        o3d.io.write_point_cloud("tmp/graspnet_scene.pcd", scene_pcd)
        print("Scene point cloud saved to tmp/graspnet_scene.pcd")
        # ---------------------------------------------------------------
        
        # Currently we directly apply the grasp with the highest score
        tar_gg = gg[0]
        
        grasp = {}
        
        rotation = tar_gg.rotation_matrix
        R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]) # Tool Frame conversion
        rotation = np.dot(rotation, R)
        translation = tar_gg.translation

        grasp = {'translation': translation, 'rotation_matrix': rotation}
        
        print('Grasp:', grasp)
        return grasp
    
    def postprocess_grasp(self, grasp):
        translation = grasp['translation']
        rotation = grasp['rotation_matrix']
        
        # convert grasp to end effector pose, [x,y,z,qx,qy,qz,qw]
        rotation = torch.tensor(rotation)
        quat = transforms.matrix_to_quaternion(rotation).numpy() # w,x,y,z
        #  convert quat to x,y,z,w 
        quat = np.roll(quat, -1)
        ee_pose = np.concatenate([translation, quat], axis=0)
        
        return ee_pose
    
    def execute_grasp(self, grasp):
        # TODO
        pass
    
    def step(self, obs, prompt, cam_int, cam2world):
        grasp = self.predict_grasps(obs, prompt, cam_int, cam2world)
        ee_pose = self.postprocess_grasp(grasp)
        self.execute_grasp(ee_pose)
        
        return ee_pose