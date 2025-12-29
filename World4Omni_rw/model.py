import os
import sys
from typing import Dict, Tuple

import torch
import numpy as np
# from pytorch3d import transforms

# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'graspnet_baseline', 'models'))
# sys.path.append(os.path.join(ROOT_DIR, 'graspnet_baseline', 'dataset'))
# sys.path.append(os.path.join(ROOT_DIR, 'graspnet_baseline', 'utils'))

# print(ROOT_DIR)
# print(sys.path)
# exit()

# from graspnetAPI import GraspGroup
# from graspnet import GraspNet, pred_decode
# from collision_detector import ModelFreeCollisionDetector
# from data_utils import CameraInfo, create_point_cloud_from_depth_image, transform_point_cloud

import imageio
from PIL import Image
import open3d as o3d

import viser
import time
from scipy.spatial.transform import Rotation as R
server = viser.ViserServer()
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
        depth = obs['depth']
        if prompt is None:
            return (depth > 0)  
        else:
            image = obs['rgb']
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
            
    
    def process_input(self, obs, prompt, intrinsic):
        """
        obs: [wrist, overhead, left_shoulder, right_shoulder, wrist]_[depth, mask, point_cloud, rgb]"""
        # cloud = obs.wrist_point_cloud   # In World Frame, see https://github.com/stepjam/PyRep/blob/8f420be8064b1970aae18a9cfbc978dfb15747ef/pyrep/objects/vision_sensor.py#L155
        # color = obs.wrist_rgb.astype(np.float32) / 255.0
        # depth = obs.wrist_depth
        
        print(f"before get mask")
        mask = self.get_grounded_mask(obs, prompt)
        print(f"After get mask")

        # mask = (depth>0)  

        color = obs['rgb'] / 255.0
        depth = obs['depth']
        print(f"depth: {depth}")
        # print max depth
        print(f"max depth: {np.max(depth)}")
        # print min depth
        print(f"min depth: {np.min(depth)}")
        # print mean depth
        print(f"mean depth: {np.mean(depth)}")

        print(f"color: {color}")
        print(f"color.shape: {color.shape}")

        intrinsic = intrinsic
        factor_depth = 1.0
        # generate cloud
        camera = CameraInfo(1920.0, 1080.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # mask = ((depth > 0))

        cloud_masked = cloud[mask]
        color_masked = color[mask]

        print(f"cloud_masked: {cloud_masked.shape}")
        print(f"color_masked: {color_masked.shape}")
        # print min, max, mean of cloud_masked
        print(f"min cloud_masked: {np.min(cloud_masked)}")
        print(f"max cloud_masked: {np.max(cloud_masked)}")
        print(f"mean cloud_masked: {np.mean(cloud_masked)}")
        
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

        X_BaseCamera_manual_path = f"/home/chn-4o/gpt-4o/rwVR/data/cameras/CL8H74100BB/0515_excalib_capture00/manual_X_BaseCamera.npy"
        X_BaseCamera = np.load(X_BaseCamera_manual_path)

        # convert cloud_masked to base camera frame
        cloud_masked_base_camera = X_BaseCamera[:3, :3] @ cloud_masked.T + X_BaseCamera[:3, 3:4]
        cloud_masked_base_camera = cloud_masked_base_camera.T
        
        server.scene.add_point_cloud(
                "pc",
                cloud_masked_base_camera,
                color_masked, 
                point_size=0.00001,
                point_shape="square"
            )
        print(f"render in viser")
        # breakpoint()
        time.sleep(1)
        o3d.visualization.draw_geometries([cloud])
        # o3d.io.write_point_cloud("tmp/point_cloud.pcd", cloud)
        
        return end_points, cloud
    
    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        
        return gg
    
    def vis_grasps(self, gg, cloud):
        gg.nms()
        gg.sort_by_score()
        gg = gg[:50]
        print(f"gg = {gg}")
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

    def get_target_obj_center(self, obs, prompt, intrinsic):
        end_points, cloud = self.process_input(obs, prompt, intrinsic)
        pc = end_points['point_clouds'] # [1, 20000, 3]
        pc = pc.squeeze(0) # [20000, 3]
        pc = pc.cpu().numpy()
        print(f"pc: {pc}")
        pc_center = np.mean(pc, axis=0)
        print(f"pc_center: {pc_center}")
        return pc_center

    def select_target_obj_grasp(self, gg, target_obj_center, distance_thresh=0.05):
        len_gg = len(gg)
        gg_selected_index = []
        for i in range(len_gg):
            gg_i = gg[i]
            print(f"gg_i.translation: {gg_i.translation}")
            print(f"target_obj_center: {target_obj_center}")
            if np.linalg.norm(gg_i.translation - target_obj_center) < distance_thresh:
                gg_selected_index.append(i)
        gg_selected = gg[gg_selected_index]
        print(f"gg_selected: {gg_selected}")
        return gg_selected

    def predict_grasps(self, obs, prompt_scene, prompt_obj, intrinsic):
        print("process scene input start")
        end_points, cloud = self.process_input(obs, prompt_scene, intrinsic)
        print("process scene input complete")

        with torch.no_grad():
            pred = self.net(end_points)
            grasp_group = pred_decode(pred)
        gg_array = grasp_group[0].detach().cpu().numpy()
        print("get grasp_group complete")
        # graspnet api collision detection with CoppeliaSim if imported before
        from graspnetAPI import GraspGroup
        gg = GraspGroup(gg_array)

        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        
        gg.nms()
        gg.sort_by_score()
        print(f"len(gg): {len(gg)}")
        gg = gg[:50]
        print(f"gg: {gg}")
        # visualize all gg
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
        ### select gg for target object
        target_obj_center = self.get_target_obj_center(obs, prompt_obj, intrinsic)
        
        
        # distance_thresh = 0.05 # tomato
        # distance_thresh = 0.2 # white box
        distance_thresh = 0.08 # duck


        gg = self.select_target_obj_grasp(gg, target_obj_center, distance_thresh=distance_thresh)
        gg = gg[:1]



        ## visualize the selected best gg
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
        # width, height = 512, 512
        # renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

        # # Create a scene
        # scene = renderer.scene
        # scene.set_background([1, 1, 1, 1])  # white background

        # # Add the point cloud

        # cloud_material = o3d.visualization.rendering.MaterialRecord()
        # cloud_material.shader = "defaultUnlit"
        # scene.add_geometry("cloud", cloud, cloud_material)

        # # Add grippers
        # gripper_material = o3d.visualization.rendering.MaterialRecord()
        # gripper_material.shader = "defaultUnlit"
        # for i, gripper in enumerate(grippers):
        #     scene.add_geometry(f"gripper_{i}", gripper, gripper_material)

        # # Compute scene bounds for camera placement
        # bounds = renderer.scene.bounding_box
        # center = bounds.get_center()
        # extent = bounds.get_extent().max()

        # # Define camera view
        # # eye = center + np.array([0.25, 0.25, -0.25])  # move camera to the side (+X)
        # eye = center + np.array([0.0, 0.0, 0.5])
        # up = np.array([0.0, 0.0, 1.0])  # Z is up
        # renderer.setup_camera(60.0, center, eye, up)
        
        # # Render and save to file
        # img = renderer.render_to_image()
        # o3d.io.write_image(f"tmp/grasp_scene.png", img)
        # print("Image saved to PickUpCup.png")
        
        # # ---------- NEW: save the whole scene as a PCD -----------------
        # #   1. start with the original cloud
        # scene_pcd = o3d.geometry.PointCloud(cloud)

        # #   2. sample each gripper mesh uniformly (1 000 pts) and append
        # for mesh in grippers:
        #     # ensure we have a TriangleMesh (to sample); convert LineSet etc.
        #     if isinstance(mesh, o3d.geometry.LineSet):
        #         mesh = mesh.tetra_mesh()   # basic surface from lines
        #     pts = mesh.sample_points_uniformly(number_of_points=1000)
        #     scene_pcd += pts

        # #   3. write to disk (PCD/PLY – change extension as you like)
        # o3d.io.write_point_cloud("tmp/graspnet_scene.pcd", scene_pcd)
        # print("Scene point cloud saved to tmp/graspnet_scene.pcd")
        # # ---------------------------------------------------------------
        
        # Currently we directly apply the grasp with the highest score
        tar_gg = gg[0]
        
        grasp = {}
        grasp = {'translation': tar_gg.translation, 'rotation_matrix': tar_gg.rotation_matrix}
        
        print('Grasp:', grasp)
        return grasp
    
    def postprocess_grasp(self, grasp):

        ## grasp pose in camera frame
        translation_camera_grasp = grasp['translation']
        rotation_camera_grasp= grasp['rotation_matrix']
        
        # R_fix = np.array([[ -1.,  0.,  0.],       
        #                   [  0.,  0., -1.],       
        #                   [  0.,  1.,  0.]]) # R_fix adopted by RLBench, used by graspnet_rlbench.py

        # # R_fix = np.array([[  0.,  0.,  1.],       
        # #                   [  0., -1.,  0.],       
        # #                   [ -1.,  0.,  0.]]) # R_fix adopted by Curobo, used by graspnet_rlbench_curobo.py
        
        # rotation = np.dot(R_fix, rotation)

        # For real-world implementation, we need to rotate the grasp to align with the robot's end effector frame
        
        
        ## camera pose in base camera frame
        X_BaseCamera_manual_path = f"/home/chn-4o/gpt-4o/rwVR/data/cameras/CL8H74100BB/0515_excalib_capture00/manual_X_BaseCamera.npy"
        X_BaseCamera = np.load(X_BaseCamera_manual_path)
        print(f"X_BaseCamera: {X_BaseCamera}")

        rotation_base_camera = X_BaseCamera[:3, :3]
        translation_base_camera = X_BaseCamera[:3, 3]

        R_base_camera = R.from_matrix(rotation_base_camera)
        R_camera_grasp = R.from_matrix(rotation_camera_grasp)

        # grasp pose in arm base frame
        R_base_grasp = R_base_camera.as_matrix() @ R_camera_grasp.as_matrix()
        R_base_grasp = R.from_matrix(R_base_grasp)

        # translation
        t_base_grasp = R_base_camera.apply(translation_camera_grasp) + translation_base_camera

        print(f"R_base_grasp = {R_base_grasp}")
        print(f"t_base_grasp = {t_base_grasp}")


        R_base_grasp = R_base_grasp.as_matrix() @ R.from_euler('y', 90, degrees=True).as_matrix() # @ R.from_euler('z', -90, degrees=True).as_matrix()
        R_base_grasp = R.from_matrix(R_base_grasp)
        
        coef = 0.03
        t_ee_pose_in = t_base_grasp + R_base_grasp.as_matrix()[:3, 2] * coef

        # R_base_grasp to euler angle
        euler_angles = R_base_grasp.as_euler('xyz', degrees=True)
        print(f"euler_angles = {euler_angles}")

        ee_pose = np.concatenate([t_base_grasp, euler_angles], axis=0)
        ee_pose_in = np.concatenate([t_ee_pose_in, euler_angles], axis=0)
        print(f"ee_pose = {ee_pose}")
        
        # convert to viser format

        # convert R_base_grasp to quaternion
        quat = R_base_grasp.as_quat()[[3, 0, 1, 2]]
        print(f"quat = {quat}")
        # visualize ee_pose
        server.scene.add_frame(
            "ee_pose",
            wxyz=quat,
            position=t_base_grasp,
        )

        server.scene.add_frame(
            "ee_pose_in",
            wxyz=quat,
            position=t_ee_pose_in,
        )
        breakpoint()
        
        # # convert grasp to end effector pose, [x,y,z,qx,qy,qz,qw]
        # rotation = torch.tensor(rotation)
        # quat = transforms.matrix_to_quaternion(rotation).numpy() # w,x,y,z
        # #  convert quat to x,y,z,w 
        # quat = np.roll(quat, -1)
        # ee_pose = np.concatenate([translation, quat], axis=0)
        
        return ee_pose, ee_pose_in
    
    def step(self, obs, prompt_scene, prompt_obj, intrinsic):
        grasp = self.predict_grasps(obs, prompt_scene, prompt_obj, intrinsic)
        return grasp
        ee_pose, ee_pose_in = self.postprocess_grasp(grasp)
        
        return ee_pose, ee_pose_in