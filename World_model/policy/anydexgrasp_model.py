import os
import sys
from pathlib import Path
from typing import Dict, Tuple
import copy
import json
import os
import sys
import pdb
import time
import datetime
import random
import argparse
import torch
import numpy as np
from pytorch3d import transforms

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Point to the graspnet-baseline submodule under RLBench
WORLD4OMNI_ROOT = Path(__file__).resolve().parents[1]
GRASP_BASELINE_ROOT = WORLD4OMNI_ROOT / '3rdparty' / 'RLBench' / '3rdparty' / 'graspnet-baseline'
ANYDEX_ROOT = WORLD4OMNI_ROOT / '3rdparty' / 'RLBench' / '3rdparty' / 'AnyDexGrasp'
# Ensure AnyDexGrasp models (with its own loss.py) take precedence over similarly named modules
sys.path.insert(0, str(ANYDEX_ROOT / 'models'))
sys.path.insert(0, str(ANYDEX_ROOT / 'utils'))
sys.path.insert(0, str(ANYDEX_ROOT))
sys.path.append(str(GRASP_BASELINE_ROOT / 'models'))
sys.path.append(str(GRASP_BASELINE_ROOT / 'dataset'))
sys.path.append(str(GRASP_BASELINE_ROOT / 'utils'))

# from graspnetAPI import GraspGroup
# Explicitly import the two-finger collision detector from graspnet-baseline to avoid
# name conflict with AnyDexGrasp's own utils/collision_detector.py
import importlib.util as _importlib_util
_gb_collision_path = str(GRASP_BASELINE_ROOT / 'utils' / 'collision_detector.py')
_spec = _importlib_util.spec_from_file_location('gb_collision_detector', _gb_collision_path)
_gb_mod = _importlib_util.module_from_spec(_spec) if _spec and _spec.loader else None
if _spec and _spec.loader and _gb_mod is not None:
    _spec.loader.exec_module(_gb_mod)
    ModelFreeCollisionDetector = getattr(_gb_mod, 'ModelFreeCollisionDetector')
else:
    raise ImportError(f"Failed to import ModelFreeCollisionDetector from {_gb_collision_path}")
# from data_utils import CameraInfo, create_point_cloud_from_depth_image, transform_point_cloud
try:
    from AnyDexGrasp.models.minkowski_graspnet_single_point import (
        MinkowskiGraspNet,
        MinkowskiGraspNetMultifingerType1Inference,
    )
    from AnyDexGrasp.utils.np_utils import transform_point_cloud
    from AnyDexGrasp.utils.pt_utils import batch_viewpoint_params_to_matrix
except ModuleNotFoundError:
    # Try relative imports first
    try:
        from models.minkowski_graspnet_single_point import (
            MinkowskiGraspNet,
            MinkowskiGraspNetMultifingerType1Inference,
        )
        from utils.np_utils import transform_point_cloud
        from utils.pt_utils import batch_viewpoint_params_to_matrix
    except ModuleNotFoundError:
        # Ultimate fallback: import by absolute file path to avoid namespace collisions
        _np_utils_path = str(ANYDEX_ROOT / 'utils' / 'np_utils.py')
        _pt_utils_path = str(ANYDEX_ROOT / 'utils' / 'pt_utils.py')
        _mink_path = str(ANYDEX_ROOT / 'models' / 'minkowski_graspnet_single_point.py')

        _np_spec = _importlib_util.spec_from_file_location('adg_np_utils', _np_utils_path)
        _np_mod = _importlib_util.module_from_spec(_np_spec) if _np_spec and _np_spec.loader else None
        if _np_spec and _np_spec.loader and _np_mod is not None:
            _np_spec.loader.exec_module(_np_mod)
            transform_point_cloud = getattr(_np_mod, 'transform_point_cloud')
        else:
            raise ImportError(f'Failed to import transform_point_cloud from {_np_utils_path}')

        _pt_spec = _importlib_util.spec_from_file_location('adg_pt_utils', _pt_utils_path)
        _pt_mod = _importlib_util.module_from_spec(_pt_spec) if _pt_spec and _pt_spec.loader else None
        if _pt_spec and _pt_spec.loader and _pt_mod is not None:
            _pt_spec.loader.exec_module(_pt_mod)
            batch_viewpoint_params_to_matrix = getattr(_pt_mod, 'batch_viewpoint_params_to_matrix')
        else:
            raise ImportError(f'Failed to import batch_viewpoint_params_to_matrix from {_pt_utils_path}')

        _mink_spec = _importlib_util.spec_from_file_location('adg_mink', _mink_path)
        _mink_mod = _importlib_util.module_from_spec(_mink_spec) if _mink_spec and _mink_spec.loader else None
        if _mink_spec and _mink_spec.loader and _mink_mod is not None:
            _mink_spec.loader.exec_module(_mink_mod)
            MinkowskiGraspNet = getattr(_mink_mod, 'MinkowskiGraspNet')
            MinkowskiGraspNetMultifingerType1Inference = getattr(_mink_mod, 'MinkowskiGraspNetMultifingerType1Inference')
        else:
            raise ImportError(f'Failed to import MinkowskiGraspNet from {_mink_path}')

import imageio
from PIL import Image
from scipy.spatial import KDTree
import MinkowskiEngine as ME
import open3d as o3d

MAX_GRASP_WIDTH = 0.11
MIN_GRASP_WIDTH = 0.04
BATCH_SIZE = 1
DEBUG = True
NUM_OF_ALLEGRO_DEPTH = 4
NUM_OF_ALLEGRO_TYPE = 10
Allegro_VOXElGRID = 0.003
POINTCLOUD_AUGMENT_NUM = 10
DEFAULT_DEPTH = 0.00
RANDOM = False


def parse_preds(end_points, use_v2=False):
    ## load preds
    before_generator = end_points['before_generator']  # (B, Ns, 256)
    point_features = end_points['point_features']  # (B, Ns, 512)
    coords = end_points['sinput'].C  # (\Sigma Ni, 4)
    objectness_pred = end_points['stage1_objectness_pred']  # (Sigma Ni, 2)
    objectness_mask = torch.argmax(objectness_pred, dim=1).bool()  # (\Sigma Ni,)
    seed_xyz = end_points['stage2_seed_xyz']  # (B, Ns, 3)
    seed_inds = end_points['stage2_seed_inds']  # (B, Ns)
    grasp_view_xyz = end_points['stage2_view_xyz']  # (B, Ns, 3)
    grasp_view_inds = end_points['stage2_view_inds']
    grasp_view_scores = end_points['stage2_view_scores']
    grasp_scores = end_points['stage3_grasp_scores']  # (B, Ns, A, D)
    grasp_features_two_finger = end_points['stage3_grasp_features'].view(grasp_scores.size()[0], grasp_scores.size()[1], -1) # (B, Ns, 3 + C)
    grasp_widths = MAX_GRASP_WIDTH * end_points['stage3_normalized_grasp_widths']  # (B, Ns, A, D)
    grasp_widths[grasp_widths > MAX_GRASP_WIDTH] = MAX_GRASP_WIDTH

    grasp_preds = []
    grasp_features = []
    grasp_vdistance_list = []
    for i in range(BATCH_SIZE):
        
        cloud_mask_i = (coords[:, 0] == i)
        seed_inds_i = seed_inds[i]
        objectness_mask_i = objectness_mask[cloud_mask_i][seed_inds_i]  # (Ns,)

        if objectness_mask_i.any() == False:
            continue

        seed_xyz_i = seed_xyz[i] # [objectness_mask_i]  # (Ns', 3)
        point_features_i = point_features[i] # [objectness_mask_i]
        
        seed_inds_i = seed_inds_i # [objectness_mask_i]
        before_generator_i = before_generator[i] # [objectness_mask_i]
        grasp_view_xyz_i = grasp_view_xyz[i] # [objectness_mask_i]  # (Ns', 3)
        grasp_view_inds_i = grasp_view_inds[i] # [objectness_mask_i]
        grasp_view_scores_i = grasp_view_scores[i] # [objectness_mask_i]
        grasp_scores_i = grasp_scores[i] # [objectness_mask_i]  # (Ns', A, D)
        grasp_widths_i = grasp_widths[i] # [objectness_mask_i] # (Ns', A, D)
        
        Ns, A, D = grasp_scores_i.size()
        grasp_features_two_finger_i = grasp_features_two_finger[i] # [objectness_mask_i] # (Ns', 3 + C)
        grasp_scores_i_A_D = copy.deepcopy(grasp_scores_i).view(Ns, -1)

        grasp_scores_i = torch.minimum(grasp_scores_i[:,:24,:], grasp_scores_i[:,24:,:])

        seed_inds_i = seed_inds_i.view(Ns, -1)
        grasp_view_inds_i = grasp_view_inds_i.view(Ns, -1)
        grasp_view_scores_i = grasp_view_scores_i.view(Ns, -1)

        grasp_scores_i, grasp_angles_class_i = torch.max(grasp_scores_i, dim=1) # (Ns', D), (Ns', D)
        grasp_angles_i = (grasp_angles_class_i.float()-12) / 24 * np.pi  # (Ns', topk, D)

        # grasp width & vdistance
        grasp_angles_class_i = grasp_angles_class_i.unsqueeze(1) # (Ns', 1, D)
        grasp_widths_pos_i = torch.gather(grasp_widths_i, 1, grasp_angles_class_i).squeeze(1) # (Ns', D)
        grasp_widths_neg_i = torch.gather(grasp_widths_i, 1, grasp_angles_class_i+24).squeeze(1) # (Ns', D)

        ## slice preds by grasp score/depth
        # grasp score & depth
        grasp_scores_i, grasp_depths_class_i = torch.max(grasp_scores_i, dim=1, keepdims=True) # (Ns', 1), (Ns', 1)
        grasp_depths_i = (grasp_depths_class_i.float() + 1) * 0.01  # (Ns'*topk, 1)
 
        grasp_depths_i -= 0.01
        grasp_depths_i[grasp_depths_class_i==0] = 0.005
        # grasp angle & width & vdistance
        grasp_angles_i = torch.gather(grasp_angles_i, 1, grasp_depths_class_i) # (Ns', 1)
        grasp_widths_pos_i = torch.gather(grasp_widths_pos_i, 1, grasp_depths_class_i) # (Ns', 1)
        grasp_widths_neg_i = torch.gather(grasp_widths_neg_i, 1, grasp_depths_class_i) # (Ns', 1)

        # convert to rotation matrix
        rotation_matrices_i = batch_viewpoint_params_to_matrix(-grasp_view_xyz_i, grasp_angles_i.squeeze(1))
        # # adjust gripper centers
        grasp_widths_i = grasp_widths_pos_i + grasp_widths_neg_i
        rotation_matrices_i = rotation_matrices_i.view(Ns, 9)

        # merge preds
        grasp_preds.append(torch.cat([grasp_scores_i, grasp_widths_i, grasp_depths_i, rotation_matrices_i, seed_xyz_i],axis=1))  # (Ns, 15)
        grasp_features.append(torch.cat([grasp_scores_i_A_D, grasp_features_two_finger_i, before_generator_i, point_features_i, grasp_view_inds_i, grasp_view_scores_i, seed_inds_i, grasp_angles_i*24/np.pi+12, grasp_depths_i], axis=1)) # (Ns'*3, A, D)
        
    return grasp_preds, grasp_features


class GroundedAnyDexGraspNet:
    def __init__(self, ckpt_path, task_name=None, num_points=20000, num_view=300, collision_thresh=0.01, voxel_size=0.005):
        self.collision_thresh = collision_thresh
        self.num_points = num_points
        self.num_view = num_view
        self.ckpt_path = ckpt_path
        self.voxel_size = voxel_size
        self.task_name = task_name
        
        self.net = MinkowskiGraspNet(num_depth=5, num_seed=2048, is_training=False, half_views=False)
        
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

            # Robust fallback: if no detections/masks, return a trivial depth>0 mask
            if detections is None or len(detections) == 0 or detections[0].mask is None:
                return (depth > 0)

            return detections[0].mask.astype(bool)  # mask of the first object


    def process_input(self, obs, prompt, cam_int, voxel_size=0.005):
        """
        obs: [wrist, overhead, left_shoulder, right_shoulder, wrist]_[depth, mask, point_cloud, rgb]"""
        cloud = obs.wrist_point_cloud   # In World Frame, see https://github.com/stepjam/PyRep/blob/8f420be8064b1970aae18a9cfbc978dfb15747ef/pyrep/objects/vision_sensor.py#L155
        color = obs.wrist_rgb.astype(np.float32) / 255.0
        depth = obs.wrist_depth

        cloud[:,:,-1] = cloud[:,:,-1] * -1.0  # flip Z axis to match the camera frame
        cloud[:,:,0] = cloud[:,:,0] * -1.0
        # # self.ori_cloud = cloud.copy()
        
        # factor_depth = 0.3
        
        # camera = CameraInfo(512.0, 512.0, cam_int[0][0], cam_int[1][1], cam_int[0][2], cam_int[1][2], factor_depth)
        # cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        
        prompt = None
        mask = self.get_grounded_mask(obs, prompt)  

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
        ori_end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        ori_end_points['point_clouds'] = cloud_sampled
        ori_end_points['cloud_colors'] = color_sampled

        # o3d.io.write_point_cloud("tmp/point_cloud.pcd", cloud)
        
        new_points = cloud_sampled.cpu().squeeze()
        coords = np.ascontiguousarray(new_points / voxel_size, dtype=int)
        _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
        coords = coords[idxs]
        new_points = new_points[idxs]
        coords_batch, points_batch = ME.utils.sparse_collate([coords], [new_points])
        sinput = ME.SparseTensor(points_batch, coords_batch, device=device)
        new_end_points = {'sinput': sinput, 'point_clouds':[sinput.F]}
        
        return new_end_points, cloud
    
    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        
        return gg
    
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
        
    
    def predict_grasps(self, obs, prompt, cam_int, cam2world):
        end_points, cloud = self.process_input(obs, prompt, cam_int, voxel_size=self.voxel_size)
        
        with torch.no_grad():
            end_points = self.net(end_points)
            preds, grasp_features = parse_preds(end_points, use_v2=True)
            if len(preds) == 0:
                print('No grasp detected')
            else:
                preds = preds[0]
                print('Grasps detected.')
        # mask = (preds[:,9] > 0.93) & (preds[:,1] < MAX_GRASP_WIDTH) & (preds[:,1] > MIN_GRASP_WIDTH)
        # workspace_mask = (preds[:,12] > -0.2) & (preds[:,12] < 0.2) & (preds[:,13] > -0.20) & (preds[:,13] < 0.07) 

        # preds = preds[workspace_mask & mask]
        # grasp_features = grasp_features[0][workspace_mask & mask]
        grasp_features = grasp_features[0]
        if len(preds) == 0:
            print('No grasp detected after masking')
        
        # points = points.cuda()
        heights = 0.03 * torch.ones([preds.shape[0], 1]).cuda()
        object_ids = -1 * torch.ones([preds.shape[0], 1]).cuda()
        gg_array = torch.cat([preds[:, 0:2], heights, preds[:, 2:15], preds[:, 15:16], object_ids], axis=-1).cpu().numpy()

        # gg_array = grasp_group[0].detach().cpu().numpy()
        
        # graspnet api collision detection with CoppeliaSim if imported before
        from graspnetAPI import GraspGroup
        # breakpoint()
        gg = GraspGroup(gg_array)
        print('Grasp Group Converted.')
        
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        
        
        print('Filtering Grasps')
        gg.nms()
        gg.sort_by_score()
        
        filtered_grasp_list = self._filter_grasps(gg, obs, prompt)
        print('Filtering Done')
        
        # ===== 新增筛选部分：对抓取预测的接近方向进行垂直角度限制 =====
        # 将 gg 转换为普通列表
        # all_grasps = list(gg)
        # vertical = np.array([0, 0, 1])  # 期望抓取接近方向（垂直桌面）
        # angle_threshold = np.deg2rad(30)  # 30度的弧度值
        # filtered = []
        # for grasp in all_grasps:
        #     # 抓取的接近方向取 grasp.rotation_matrix 的第三列
        #     approach_dir = grasp.rotation_matrix[:, 0]
        #     # 计算夹角：cos(angle)=dot(approach_dir, vertical)
        #     cos_angle = np.dot(approach_dir, vertical)
        #     cos_angle = np.clip(cos_angle, -1.0, 1.0)
        #     angle = np.arccos(cos_angle)
        #     if angle < angle_threshold:
        #         filtered.append(grasp)
        # if len(filtered) == 0:
        #     print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
        #     filtered = all_grasps
        # else:
        #     print(
        #         f"\n[DEBUG] Filtered {len(filtered)} grasps within ±30° of vertical out of {len(all_grasps)} total predictions.")

        # # 对过滤后的抓取根据 score 排序（降序）
        # filtered.sort(key=lambda g: g.score, reverse=True)
        
        gg = filtered_grasp_list[:50]
        
        # # Visualize the grasps and point cloud using Open3D
        grippers = [g.to_open3d_geometry() for g in gg]

        
        # Uncomment the following if Open3D is correctly installed on your device.
                
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
        # eye = center + np.array([-0.25, 0.0, -0.25])
        # up = np.array([0.0, 0.0, -1.0])  # -Z is up
        # renderer.setup_camera(60.0, center, eye, up)
        
        # # Render and save to file
        # img = renderer.render_to_image()
        # o3d.io.write_image(f"tmp/grasp_scene.png", img)
        # print("Image saved to tmp/grasp_scene.png")
        
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
        o3d.io.write_point_cloud(f"tmp/anydexgrasp_scene_{self.task_name}.pcd", scene_pcd)
        print(f"Scene point cloud saved to tmp/anydexgrasp_scene_{self.task_name}.pcd")
        # ---------------------------------------------------------------
        
        
        # Currently we directly apply the grasp with the highest score
        tar_gg = gg[0]
        
        grasp = {}
        
        rotation = tar_gg.rotation_matrix
        translation = tar_gg.translation
        
        T_grasp = np.eye(4)
        T_grasp[:3, :3] = rotation
        T_grasp[:3, 3] = translation
        
        R = np.array([
            [0.0, 0.0, 1.0],  # newX = oldZ
            [0.0, 1.0, 0.0],  # newY = oldY
            [-1.0, 0.0, 0.0],  # newZ = -oldX  (或者相当于 oldX = -newZ)
        ])
        R = np.array([[0.0,-1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        t = np.array([0.0, 0.0, tar_gg.depth])
        T_tool = np.eye(4)
        T_tool[:3, :3] = R
        T_tool[:3, 3] = t
        
        # rotation = np.dot(rotation, R)
        
        T_real_grasp = T_grasp @ T_tool
        translation = T_real_grasp[:3, 3]
        rotation = T_real_grasp[:3, :3]
        
        R = np.array([[-1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0,-1.0]])
        rotation = np.dot(R, rotation)
        translation = np.dot(R, translation)
        

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
    
    def step(self, obs, prompt, cam_int, cam2world):
        grasp = self.predict_grasps(obs, prompt, cam_int, cam2world)
        ee_pose = self.postprocess_grasp(grasp)
        
        return ee_pose