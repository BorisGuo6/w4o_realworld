import copy
import json
import os
import sys
import pdb
import time
import datetime
import argparse
import torch
import numpy as np
import cv2
import open3d as o3d

from graspnetAPI import GraspGroup

from multiprocessing import shared_memory
from collections import OrderedDict
import matplotlib.pyplot as plt
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from np_utils import transform_point_cloud
from pt_utils import batch_viewpoint_params_to_matrix
from collision_detector import ModelFreeCollisionDetectorMultifinger, ModelFreeCollisionDetector
import queue
from itertools import count
from threading import Thread
from queue import Queue
import multiprocessing as mp
import MinkowskiEngine as ME
from models.minkowski_graspnet_single_point import MinkowskiGraspNet
from robot_arm_toolbox.realman_gripper import RealMan_Gripper
from realsense import RealSense

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--robot_ip', required=True, help='Robot IP')
parser.add_argument('--camera_serial', required=True, help='Camera IP')
parser.add_argument('--use_graspnet_v2', action='store_true', help='Whether to use graspnet v2 format')
parser.add_argument('--half_views', action='store_true', help='Use only half views in network.')
parser.add_argument('--global_camera', action='store_true', help='Use the settings for global camera.')
cfgs = parser.parse_args()

MAX_GRASP_WIDTH = 0.09
MIN_GRASP_WIDTH = 0.05
BATCH_SIZE = 1
DEBUG = True
VOXElGRID = 0.003
POINTCLOUD_AUGMENT_NUM = 10
DEFAULT_DEPTH = 0.00
RANDOM = False
approach_distance = 0.02

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

def get_net(checkpoint_path, use_v2=False):
    if use_v2:
        net = MinkowskiGraspNet(num_depth=5, num_seed=2048, is_training=False, half_views=cfgs.half_views)
    else:
        net = MinkowskiGraspNet(num_depth=4, num_seed=2048, is_training=False, half_views=cfgs.half_views)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    return net

def filter_worksapce(preds, large_workspace, small_workspace):
    # backward:y, right:x, down:z
    # large_workspace [-x, x, -y, y]
    # small_workspace [-x, x, -y, y]
    lw_mask = (preds[:,12] > large_workspace[0]) & (preds[:,12] < large_workspace[1]) & (preds[:,13] > large_workspace[2]) & (preds[:,13] < large_workspace[3])
    forward_mask = (preds[:,13] < small_workspace[2]) & (preds[:, 6] < 0)
    backward_mask = (preds[:,13] > small_workspace[3]) & (preds[:, 6] > 0)
    left_mask = (preds[:,12] < small_workspace[0]) & (preds[:, 3] < 0)
    right_mask = (preds[:,12] > small_workspace[1]) & (preds[:, 3] > 0)
    
    return lw_mask #& forward_mask & backward_mask & left_mask & right_mask

def get_robot(robot_ip="192.168.1.100",gripper_type='eg2'):
    robot = RealMan_Gripper(host=robot_ip, gripper_type=gripper_type)
    return robot

def get_depth(camera):
    time.sleep(0.1)
    colors, depths = camera.get_rgbd_image()
    colors = (cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
    return colors, depths

def get_grasp(net, colors, depths, camera, augment_mat=np.eye(4), flip=False, voxel_size=0.005):
    cx, cy, fx, fy = camera.color_intrin_part
    s = 1000.0

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0.15) & (points_z < 0.56)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)

    if DEBUG:
        colors = colors[mask].astype(np.float32)

    cloud = None
    if DEBUG:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)

    points = transform_point_cloud(points, augment_mat).astype(np.float32)
    points = torch.from_numpy(points)
    coords = np.ascontiguousarray(points / voxel_size, dtype=int)
    # Upd Note. API change.
    _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
    coords = coords[idxs]
    points = points[idxs]
    coords_batch, points_batch = ME.utils.sparse_collate([coords], [points])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sinput = ME.SparseTensor(points_batch, coords_batch, device=device)

    end_points = {'sinput': sinput, 'point_clouds': [sinput.F]}
    with torch.no_grad():
        end_points = net(end_points)
        preds, grasp_features = parse_preds(end_points, use_v2=cfgs.use_graspnet_v2)
        if len(preds) == 0:
            print('No grasp detected')
            return None, cloud, points.cuda(), None, [sinput]
        else:
            preds = preds[0]
    # filter
    if flip:
        augment_mat[:, 0] = -augment_mat[:, 0]
    augment_mat_tensor = torch.tensor(copy.deepcopy(np.linalg.inv(augment_mat).astype(np.float32)), device=device)
    rotation = augment_mat_tensor[:3, :3].reshape((-1)).repeat((preds.size()[0], 1)).view((preds.size()[0], 3, 3))
    translation = augment_mat_tensor[:3, 3]

    preds[:,12:15] = torch.matmul(rotation, preds[:,12:15].view((-1, 3, 1))).view(-1, 3) + translation
    pose_rotation = torch.matmul(rotation, preds[:,3:12].view((-1, 3, 3)))
    if flip:
        preds[:, 12] = -preds[:, 12]
        pose_rotation[:, 0, :] = -pose_rotation[:, 0, :]
        pose_rotation[:, :, 1] = -pose_rotation[:, :, 1]
    preds[:, 3:12] = pose_rotation.view((-1, 9))

    mask = (preds[:,9] > 0.85) & (preds[:,1] < MAX_GRASP_WIDTH) & (preds[:,1] > MIN_GRASP_WIDTH)
    # workspace_mask = (preds[:,12] > -0.1) & (preds[:,12] < 0.1) & (preds[:,13] > -0.1) & (preds[:,13] < 0.05)
    large_workspace = [-0.17, 0.1, -0.13, 0.05]
    small_workspace = [-0.05, 0.05, -0.05, 0.025]
    z_mask = (preds[:,14] > 0.3) & (preds[:,14] < 0.56)
    workspace_mask = filter_worksapce(preds=preds, large_workspace=large_workspace, small_workspace=small_workspace) #& z_mask
    preds = preds[workspace_mask & mask]
    grasp_features = grasp_features[0][workspace_mask & mask]
    if len(preds) == 0:
        print('No grasp detected after masking')
        return None, cloud, points.cuda(), None, [sinput]

    points = points.cuda()
    heights = 0.03 * torch.ones([preds.shape[0], 1]).cuda()
    object_ids = -1 * torch.ones([preds.shape[0], 1]).cuda()
    ggarray = torch.cat([preds[:, 0:2], heights, preds[:, 2:15], preds[:, 15:16], object_ids], axis=-1)

    return ggarray, cloud, points, grasp_features, [sinput]

def augment_data(flip=False):
    flip_mat = np.identity(4)
    # Flipping along the YZ plane
    if flip:
        flip_mat = np.array([[-1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

    # Rotation along up-axis/Z-axis
    rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    rot_mat = np.array([[c, -s, 0, 0],
                        [s, c, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    # Translation along X/Y/Z-axis
    offset_x = np.random.random() * 0.1 - 0.05  # -0.05 ~ 0.05
    offset_y = np.random.random() * 0.1 - 0.05  # -0.05 ~ 0.05
    trans_mat = np.array([[1, 0, 0, offset_x],
                          [0, 1, 0, offset_y],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

    aug_mat = np.dot(trans_mat, np.dot(rot_mat, flip_mat).astype(np.float32)).astype(np.float32)
    return aug_mat

def get_ggarray_features(colors, depths, net, camera):
    augment_mat1 = np.eye(4)
    augment_mats = []
    for i in range(POINTCLOUD_AUGMENT_NUM):
        if i % 2 == 0:
            augment_mat = augment_data()
        else:
            augment_mat = augment_data(flip=True)
        augment_mats.append(augment_mat)

    ggarray, cloud, points_down, grasp_features, sinput = get_grasp(net, colors, depths, camera, augment_mat=augment_mat1)
    for i in range(POINTCLOUD_AUGMENT_NUM):
        if i % 2 == 0:
            ggarray2, _, _, grasp_features2, sinput2 = get_grasp(net, colors, depths, camera, augment_mat=augment_mats[i])
        else:
            ggarray2, _, _, grasp_features2, sinput2 = get_grasp(net, colors, depths, camera, augment_mat=augment_mats[i], flip=True)
        if ggarray2 is None:
            continue
        if ggarray is None:
            ggarray = ggarray2
            grasp_features = grasp_features2
            sinput = sinput2
        else:
            sinput.append(sinput2[0])
            ggarray = torch.cat([ggarray, ggarray2], axis=0)
            grasp_features = torch.cat([grasp_features, grasp_features2], axis=0)
    return ggarray, cloud, points_down, grasp_features, sinput

def robot_grasp(cfgs):
    net = get_net(cfgs.checkpoint_path, use_v2=cfgs.use_graspnet_v2)
    robot = get_robot(cfgs.robot_ip, gripper_type='eg2')
    camera = RealSense(serial=cfgs.camera_serial, frame_rate=30)
    # robot.rm_set_gripper_position(999, block=True, timeout=2)  
    try:
        vel = 20
        while True:
            t1 = time.time()
            robot.rm_movel(robot.ready_pose(), v=vel, r=0, connect=0, block=1) 
            t2 = time.time()
            time.sleep(0.3)
            colors, depths = get_depth(camera)
            ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(colors, depths, net, camera)
            t3 = time.time()
            print(f'Net Time:{t3 - t2}')
            if ggarray is None:
                if DEBUG:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                    o3d.visualization.draw_geometries([cloud, frame, sphere])
                continue

            ########## PROCESS GRASPS ##########
            # collision detection
            ggarray = GraspGroup(ggarray.cpu().numpy())
            # if DEBUG:
            #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            #     o3d.visualization.draw_geometries([cloud, frame] + ggarray.to_open3d_geometry_list())
            topk = 400
            ggarray.sort_by_score()
            ggarray = ggarray[:topk]
            # if DEBUG:
            #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            #     o3d.visualization.draw_geometries([cloud, frame] + ggarray.to_open3d_geometry_list())
            mfcdetector = ModelFreeCollisionDetector(points_down.cpu().numpy(), voxel_size=0.001)
            collision_mask, gg = mfcdetector.detect(ggarray, approach_dist=approach_distance, collision_thresh=0.0, return_ious=False, adjust_gripper_centers=True)
            gg = gg[~collision_mask]
            t4 = time.time()
            print(f'colllision Time:{t4 - t3}')
            if len(gg) == 0:
                print('No Grasp detected after collision detection!')
                if DEBUG:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                    o3d.visualization.draw_geometries([cloud, frame, sphere])
                continue

            # sort
            gg.sort_by_score()
            gg_pick = gg[0:10]
            g_old = gg_pick[0]

            if DEBUG:
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                translation = g_old.translation
                rotation = g_old.rotation_matrix
                g_old_matrix = np.vstack((np.hstack((rotation, translation.reshape((3,1)))), np.array((0,0,0,1.0))))
                frame_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(0.15).transform(g_old_matrix)
                o3d.visualization.draw_geometries([ g_old.to_open3d_geometry((0,1,0)), cloud, frame, frame_pose])

            # robot.rm_set_gripper_position(min(int(g_old.width*999/0.07), 999), block=True, timeout=2)
            robot.grasp_and_throw(g_old, vel=vel, approach_dist=approach_distance,
                                             execute_grasp=True, use_ready_pose=True)

            t5 = time.time()
            mpph = 3600 / (t5 - t1)
            print(f'\033[1;31mMPPH:{mpph}\033[0m\n--------------------')
    finally:
        pass


if __name__ == '__main__':

    t0 = time.time()
    try:
        robot_grasp(cfgs)
    finally:
        tn = time.time()
        print(f'total time:{tn - t0}')