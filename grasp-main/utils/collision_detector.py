__version__ = '1.0'

import copy
import math
import numpy as np
import open3d as o3d

from graspnetAPI import GraspGroup

class CollisionType:
    NONE    = 0B00000000
    SELF    = 0B00000001
    OTHERS  = 0B00000010
    TABLE   = 0B00000100
    BOX     = 0B00001000
    ANY     = 0B11111111


class ModelFreeCollisionDetector():
    ''' Collision detection in scenes without object labels.
        example usage:
            mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.005)
            collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.03)
            collision_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05, return_ious=True)
            collision_mask, empty_mask = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01)
            collision_mask, empty_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01, return_ious=True)
    '''
    def __init__(self, scene_points, voxel_size=0.005):
        ''' Init function. Current finger width and length are fixed.
            Input:
                scene_points: [numpy.ndarray, (N,3), numpy.float32]
                        the scene points to detect
                voxel_size: [float]
                        used for downsample
        '''
        self.finger_width = 0.01
        self.finger_length = 0.06
        self.voxel_size = voxel_size
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        self.scene_points = np.array(scene_cloud.points, dtype=np.float32)

    def detect(self, grasp_group, camera_gripper_distance=np.array([-0.082, 0, 0.069]), approach_dist=0.03, collision_thresh=0.05, return_empty_grasp=False, empty_thresh=0.01, return_ious=False, adjust_gripper_centers=False):
        ''' Detect collision of grasps.
            Input:
                grasp_group: [GraspGroup, M grasps]
                        the grasps to check
                approach_dist: [float]
                        the distance for a gripper to move along approaching direction before grasping
                        this shifting space requires no point either
                collision_thresh: [float]
                        if global collision iou is greater than this threshold,
                        a collision is detected
                return_empty_grasp: [bool]
                        if True, return a mask to imply whether there are objects in a grasp
                empty_thresh: [float]
                        if inner space iou is smaller than this threshold,
                        a collision is detected
                        only set when [return_empty_grasp] is True
                return_ious: [bool]
                        if True, return global collision iou and part collision ious
                adjust_gripper_centers: [bool]
                        if True, add an offset to grasp which makes grasp point closer to object center
                camera_gripper_distance: [x, y, z]
                        if True, add an offset to grasp which makes grasp point closer to object center
            Output:
                collision_mask: [numpy.ndarray, (M,), numpy.bool]
                        True implies collision
                [optional] empty_mask: [numpy.ndarray, (M,), numpy.bool]
                        True implies empty grasp
                        only returned when [return_empty_grasp] is True
                [optional] iou_list: list of [numpy.ndarray, (M,), numpy.float32]
                        global and part collision ious, containing
                        [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
                        only returned when [return_ious] is True
                [optional] grasp_group: [GraspGroup, M grasps]
                        translated grasps
                        only returned when [adjust_gripper_centers] is True

        '''
        approach_dist = max(approach_dist, self.finger_width)
        T = grasp_group.translations
        R = grasp_group.rotation_matrices
        heights = grasp_group.heights[:,np.newaxis]
        depths = grasp_group.depths[:,np.newaxis]
        widths = grasp_group.widths[:,np.newaxis]
        targets = self.scene_points[np.newaxis,:,:] - T[:,np.newaxis,:]
        targets = np.matmul(targets, R)
        camera_heights = np.array([[0.026]] * len(grasp_group)) 
        camera_lengths = np.array([[0.09]] * len(grasp_group)) 
        camera_widths = np.array([[0.025]] * len(grasp_group)) 
        # camera_targets = copy.deepcopy(targets) + camera_gripper_distance

        ## adjust gripper centers
        if adjust_gripper_centers:
            grasp_group, targets = self._adjust_gripper_centers(grasp_group, targets, heights, depths, widths)

        ## collision detection
        # get grasp collision masks
        left_mask, right_mask, bottom_mask, shifting_mask, inner_mask = self._get_grasp_collision_masks(targets, heights, depths, widths, approach_dist)
        camera_mask = self._get_camera_collision_masks(targets, camera_gripper_distance, heights=camera_heights, lengths=camera_lengths, widths=camera_widths)
        global_mask = (left_mask | right_mask | bottom_mask | shifting_mask | camera_mask)

        # calculate equivalant volume of each part
        left_right_volume = (heights * self.finger_length * self.finger_width / (self.voxel_size**3)).reshape(-1)
        bottom_volume = (heights * (widths+2*self.finger_width) * self.finger_width / (self.voxel_size**3)).reshape(-1)
        shifting_volume = (heights * (widths+2*self.finger_width) * approach_dist / (self.voxel_size**3)).reshape(-1)
        volume = left_right_volume*2 + bottom_volume + shifting_volume

        # get collision iou of each part
        global_iou = global_mask.sum(axis=1) / (volume+1e-6)

        # get collison mask
        collision_mask = (global_iou > collision_thresh)

        if not (return_empty_grasp or return_ious or adjust_gripper_centers):
            return collision_mask

        ret_value = [collision_mask,]
        if return_empty_grasp:
            inner_volume = (heights * self.finger_length * widths / (self.voxel_size**3)).reshape(-1)
            empty_mask = (inner_mask.sum(axis=-1)/inner_volume < empty_thresh)
            ret_value.append(empty_mask)
        if return_ious:
            left_iou = left_mask.sum(axis=1) / (left_right_volume+1e-6)
            right_iou = right_mask.sum(axis=1) / (left_right_volume+1e-6)
            bottom_iou = bottom_mask.sum(axis=1) / (bottom_volume+1e-6)
            shifting_iou = shifting_mask.sum(axis=1) / (shifting_volume+1e-6)
            ret_value.append([global_iou, left_iou, right_iou, bottom_iou, shifting_iou])
        if adjust_gripper_centers:
            ret_value.append(grasp_group)
        return ret_value

    def _adjust_gripper_centers(self, grasp_group, targets, heights, depths, widths):
        ## get point masks
        # height mask
        mask1 = ((targets[:,:,2] > -heights/2) & (targets[:,:,2] < heights/2))
        # left finger mask
        mask2 = ((targets[:,:,0] > depths - self.finger_length) & (targets[:,:,0] < depths))
        mask4 = (targets[:,:,1] < -widths/2)
        # right finger mask
        mask6 = (targets[:,:,1] > widths/2)
        # get inner mask of each point
        inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))

        ## adjust targets and gripper centers
        # get point bounds
        targets_y = targets[:,:,1].copy()
        targets_y[~inner_mask] = 0
        ymin = targets_y.min(axis=1)
        ymax = targets_y.max(axis=1)
        # get offsets
        offsets = np.zeros([targets.shape[0],3], dtype=targets.dtype)
        offsets[:,1] = (ymin + ymax) / 2
        # adjust targets
        targets[:,:,1] -= offsets[:,np.newaxis,1]
        # adjust gripper centers
        R = grasp_group.rotation_matrices
        # grasp_group.widths = ymax - ymin
        grasp_group.widths = np.maximum(0.025 * np.ones(ymax.shape), 1.7 * (ymax - ymin))
        grasp_group.translations += np.matmul(R, offsets[:,:,np.newaxis]).squeeze(2)

        return grasp_group, targets

    def _get_grasp_collision_masks(self, targets, heights, depths, widths, approach_dist):
        # height mask
        mask1 = ((targets[:,:,2] > -heights/2) & (targets[:,:,2] < heights/2))
        # left finger mask
        mask2 = ((targets[:,:,0] > depths - self.finger_length) & (targets[:,:,0] < depths))
        mask3 = (targets[:,:,1] > -(widths/2 + self.finger_width))
        mask4 = (targets[:,:,1] < -widths/2)
        # right finger mask
        mask5 = (targets[:,:,1] < (widths/2 + self.finger_width))
        mask6 = (targets[:,:,1] > widths/2)
        # bottom mask
        mask7 = ((targets[:,:,0] <= depths - self.finger_length)\
                & (targets[:,:,0] > depths - self.finger_length - self.finger_width))
        # shifting mask
        mask8 = ((targets[:,:,0] <= depths - self.finger_length - self.finger_width)\
                & (targets[:,:,0] > depths - self.finger_length - self.finger_width - approach_dist))

        # get collision mask of each point
        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)
        shifting_mask = (mask1 & mask3 & mask5 & mask8)
        inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))

        return left_mask, right_mask, bottom_mask, shifting_mask, inner_mask
    
    def _get_camera_collision_masks(self, targets, camera_gripper_distance, heights, lengths, widths):
        # camera mask
        mask1 = ((targets[:,:,0] > -heights/2 + camera_gripper_distance[0]) & (targets[:,:,0] < heights/2 + camera_gripper_distance[0]))
        mask2 = ((targets[:,:,1] > -lengths/2 + camera_gripper_distance[1]) & (targets[:,:,1] < lengths/2 + camera_gripper_distance[1]))
        mask3 = ((targets[:,:,2] > -widths/2 + camera_gripper_distance[2]) & (targets[:,:,2] < widths/2 + camera_gripper_distance[2]))
        
        camera_mask = mask1 & mask2 & mask3
        return camera_mask




class ModelFreeCollisionDetectorTwofinger():
    def __init__(self, scene_points, voxel_size=0.001):
        ''' Init function. Current finger width and length are fixed.
            Input:
                scene_points: [numpy.ndarray, (N,3), numpy.float32]
                        the scene points to detect
                voxel_size: [float]
                        used for downsample
        '''
        self.finger_width = 0.02
        self.finger_length = 0.06
        self.voxel_size = voxel_size
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        self.scene_cloud = scene_cloud
        self.scene_points = np.array(scene_cloud.points, dtype=np.float32)

    def _adjust_gripper_centers(self, grasp_group, targets, heights, depths, widths):
        targets = np.array(targets, dtype=np.float32)
        ## get point masks
        # height mask
        mask1 = ((targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2))
        # left finger mask
        mask2 = ((targets[:, :, 0] > depths - self.finger_length) & (targets[:, :, 0] < depths))
        mask4 = (targets[:, :, 1] < -widths / 2)
        # right finger mask
        mask6 = (targets[:, :, 1] > widths / 2)
        # get inner mask of each point
        inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))

        ## adjust targets and gripper centers
        # get point bounds
        targets_y = targets[:, :, 1].copy()
        targets_y[~inner_mask] = 0
        ymin = targets_y.min(axis=1)
        ymax = targets_y.max(axis=1)
        # get offsets
        offsets = np.zeros([targets.shape[0], 3], dtype=targets.dtype)
        offsets[:, 1] = (ymin + ymax) / 2
        # adjust targets
        targets[:, :, 1] -= offsets[:, np.newaxis, 1]
        # adjust gripper centers
        R = grasp_group.rotation_matrices
        grasp_group.widths = np.maximum(0.025 * np.ones(ymax.shape), 1.7 * (ymax - ymin))
        grasp_group.translations += np.matmul(R, offsets[:, :, np.newaxis]).squeeze(2)
        return grasp_group, targets

    def normalize(self, x):
        return np.array([x[0], x[1], x[2]]) / math.sqrt(np.power(x[0], 2) + np.power(x[1], 2) + np.power(x[2], 2))

    def load_meshes_pcls(self, meshes_pcls, two_fingers_ggarray, multifinger_ggarray):
        multifinger_pcls = []
        for id, multifinger_grasp in enumerate(multifinger_ggarray):
            grasp_type = multifinger_grasp.get_grasp_type_with_finger_name()
            width = multifinger_grasp.width
            width = str(round(width * 100, 1))
            key = grasp_type + '_' + width

            translation = multifinger_grasp.translation.reshape(3, 1)
            rotation = multifinger_grasp.rotation_matrix.reshape(3, 3)
            depth = multifinger_grasp.depth

            t = two_fingers_ggarray.translations[id].reshape(3, 1)
            r = two_fingers_ggarray.rotation_matrices[id].reshape(3, 3)

            transform_mat_two_fingers = np.vstack((np.hstack((r, t)), np.array([0, 0, 0, 1])))

            grasp_direction = np.dot(transform_mat_two_fingers, np.array([[0.01], [0], [0], [1]]))[:3]
            grasp_direction = np.array(grasp_direction).reshape(3)
            grasp_direction = grasp_direction - two_fingers_ggarray.translations[id].reshape((3))
            grasp_direction = self.normalize(grasp_direction)
            grasp_depth = np.array([[1, 0, 0, grasp_direction[0] * depth],
                                    [0, 1, 0, grasp_direction[1] * depth],
                                    [0, 0, 1, grasp_direction[2] * depth],
                                    [0, 0, 0, 1]], dtype=np.float32)

            transform_mat = np.vstack((np.hstack((rotation, translation)), np.array([0, 0, 0, 1])))
            source_mesh_pointclouds = copy.deepcopy(meshes_pcls[key])
            source_mesh_pointclouds.transform(transform_mat)
            source_mesh_pointclouds.transform(grasp_depth)
            multifinger_pcls.append(source_mesh_pointclouds)
        return multifinger_pcls

    def detect(self, multifinger_ggarray, two_fingers_ggarray, path_mesh_json, meshes_pcls, min_grasp_width=0.05, VoxelGrid=0.03, approach_dist=0.04, collision_thresh=10,
               adjust_gripper_centers=False, DEBUG=False):
        ''' Detect collision of grasps.
            Input:
                multifinger_ggarray(class multifingerGraspGroup()): [multifinger_ggarray, M grasps]
                        the grasps to check
                two_fingers_ggarray(class graspnetAPI.GraspGroup): [GraspGroup, M grasps]
                approach_dist: [float]
                        the distance for a gripper to move along approaching direction before grasping
                        this shifting space requires no point either
                collision_thresh: [float]
                        if global collision iou is greater than this threshold,
                        a collision is detected
                adjust_gripper_centers: [bool]
                        if True, add an offset to grasp which makes grasp point closer to object center
            Output:
                empty_mask: [numpy.ndarray, (M,), numpy.bool]
                        True implies empty grasp
                        only returned when [return_empty_grasp] is True
        '''

        T = two_fingers_ggarray.translations
        R = two_fingers_ggarray.rotation_matrices
        heights = two_fingers_ggarray.heights[:, np.newaxis]
        depths = two_fingers_ggarray.depths[:, np.newaxis]
        widths = two_fingers_ggarray.widths[:, np.newaxis]
        targets = self.scene_points[np.newaxis, :, :] - T[:, np.newaxis, :]
        targets = np.matmul(targets, R)

        ## adjust gripper centers
        if adjust_gripper_centers:
            two_fingers_ggarray, targets = self._adjust_gripper_centers(two_fingers_ggarray, targets, heights, depths,
                                                                        widths)
        two_fingers_ggarray.widths = two_fingers_ggarray.widths * 1.7
        min_width_index = two_fingers_ggarray.widths > min_grasp_width
        multifinger_ggarray = multifinger_ggarray[two_fingers_ggarray.widths > min_grasp_width]
        two_fingers_ggarray = two_fingers_ggarray[two_fingers_ggarray.widths > min_grasp_width]

        if len(multifinger_ggarray) == 0:
            print('min_grasp_width filter 0 ')
            return multifinger_ggarray, two_fingers_ggarray, [], min_width_index
        multifinger_ggarray.graspgroupTR_2_TR(two_fingers_ggarray, path_mesh_json)

        meshes_pointclouds_multifinger = self.load_meshes_pcls(meshes_pcls, two_fingers_ggarray, multifinger_ggarray)
        collision_mask = []
        empty_mask = []

        ps = self.scene_cloud.points
        ps = o3d.utility.Vector3dVector(ps)
        st = time.time()
        for idx, grasp in enumerate(multifinger_ggarray):
            meshes_pointclouds = meshes_pointclouds_multifinger[idx]
            grasp_direction = two_fingers_ggarray.rotation_matrices[idx].reshape(3, 3)[:3,0]
            for i in range(2, int(approach_dist * 100)+1, 3):
                meshes_pointclouds_cache = copy.deepcopy(meshes_pointclouds)
                back_distince = [[1, 0, 0, -grasp_direction[0] * i * 0.01],
                                    [0, 1, 0, -grasp_direction[1] * i * 0.01],
                                    [0, 0, 1, -grasp_direction[2] * i * 0.01],
                                    [0, 0, 0, 1]]
                meshes_pointclouds_cache.transform(np.array(back_distince, dtype=np.float32)).paint_uniform_color([0, 1, 0])
                meshes_pointclouds = meshes_pointclouds + meshes_pointclouds_cache
            st3 = time.time()
            meshes_pointclouds = meshes_pointclouds.voxel_down_sample(VoxelGrid)

            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=meshes_pointclouds, voxel_size=VoxelGrid)
            output = voxel_grid.check_if_included(ps)

            if DEBUG:
                print('transformed mesh')
                FOR_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                o3d.visualization.draw_geometries(
                    [self.scene_cloud, meshes_pointclouds, FOR_base, two_fingers_ggarray[int(idx)].to_open3d_geometry()])

            if np.array(output).astype(int).sum() > collision_thresh:
                empty_mask.append(False)
                collision_mask.append(output)
                if DEBUG:
                    print("collision")
                    collision_point_cloud = o3d.geometry.PointCloud()
                    collision_point_cloud.points = o3d.utility.Vector3dVector(
                        np.array(self.scene_cloud.points)[np.array(output)])
                    collision_point_cloud.paint_uniform_color([1, 0, 0])
                    normal_point_cloud = o3d.geometry.PointCloud()
                    normal_point_cloud.points = o3d.utility.Vector3dVector(
                        np.array(self.scene_cloud.points)[~(np.array(output))])
                    normal_point_cloud.paint_uniform_color([0, 0, 1])
                    o3d.visualization.draw_geometries([normal_point_cloud, collision_point_cloud, FOR_base, two_fingers_ggarray[int(idx)].to_open3d_geometry()])
            else:
                empty_mask.append(True)
                collision_mask.append(output)
                
        collision_mask = np.array(collision_mask)
        empty_mask = np.array(empty_mask)
        return multifinger_ggarray, two_fingers_ggarray, empty_mask, min_width_index



class ModelFreeCollisionDetectorMultifinger():
    def __init__(self, scene_points, voxel_size=0.001):
        ''' Init function. Current finger width and length are fixed.
            Input:
                scene_points: [numpy.ndarray, (N,3), numpy.float32]
                        the scene points to detect
                voxel_size: [float]
                        used for downsample
        '''
        self.finger_width = 0.02
        self.finger_length = 0.06
        self.voxel_size = voxel_size
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        self.scene_cloud = scene_cloud
        self.scene_points = np.array(scene_cloud.points, dtype=np.float32)

    def _adjust_gripper_centers(self, grasp_group, targets, heights, depths, widths):
        targets = np.array(targets, dtype=np.float32)
        ## get point masks
        # height mask
        mask1 = ((targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2))
        # left finger mask
        mask2 = ((targets[:, :, 0] > depths - self.finger_length) & (targets[:, :, 0] < depths))
        mask4 = (targets[:, :, 1] < -widths / 2)
        # right finger mask
        mask6 = (targets[:, :, 1] > widths / 2)
        # get inner mask of each point
        inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))

        ## adjust targets and gripper centers
        # get point bounds
        targets_y = targets[:, :, 1].copy()
        targets_y[~inner_mask] = 0
        ymin = targets_y.min(axis=1)
        ymax = targets_y.max(axis=1)
        # get offsets
        offsets = np.zeros([targets.shape[0], 3], dtype=targets.dtype)
        offsets[:, 1] = (ymin + ymax) / 2
        # adjust targets
        targets[:, :, 1] -= offsets[:, np.newaxis, 1]
        # adjust gripper centers
        R = grasp_group.rotation_matrices
        grasp_group.widths = np.maximum(0.025 * np.ones(ymax.shape), 1.7 * (ymax - ymin))
        grasp_group.translations += np.matmul(R, offsets[:, :, np.newaxis]).squeeze(2)
        return grasp_group, targets

    def normalize(self, x):
        return np.array([x[0], x[1], x[2]]) / math.sqrt(np.power(x[0], 2) + np.power(x[1], 2) + np.power(x[2], 2))

    def load_meshes_pcls(self, meshes_pcls, two_fingers_ggarray, multifinger_ggarray):
        multifinger_pcls = []
        for id, multifinger_grasp in enumerate(multifinger_ggarray):
            grasp_type = multifinger_grasp.get_grasp_type_with_finger_name()
            width = multifinger_grasp.width
            width = str(round(width * 100, 1))
            key = grasp_type + '_' + width

            translation = multifinger_grasp.translation.reshape(3, 1)
            rotation = multifinger_grasp.rotation_matrix.reshape(3, 3)
            depth = multifinger_grasp.depth

            t = two_fingers_ggarray.translations[id].reshape(3, 1)
            r = two_fingers_ggarray.rotation_matrices[id].reshape(3, 3)

            transform_mat_two_fingers = np.vstack((np.hstack((r, t)), np.array([0, 0, 0, 1])))

            grasp_direction = np.dot(transform_mat_two_fingers, np.array([[0.01], [0], [0], [1]]))[:3]
            grasp_direction = np.array(grasp_direction).reshape(3)
            grasp_direction = grasp_direction - two_fingers_ggarray.translations[id].reshape((3))
            grasp_direction = self.normalize(grasp_direction)
            grasp_depth = np.array([[1, 0, 0, grasp_direction[0] * depth],
                                    [0, 1, 0, grasp_direction[1] * depth],
                                    [0, 0, 1, grasp_direction[2] * depth],
                                    [0, 0, 0, 1]], dtype=np.float32)

            transform_mat = np.vstack((np.hstack((rotation, translation)), np.array([0, 0, 0, 1])))
            source_mesh_pointclouds = copy.deepcopy(meshes_pcls[key])
            source_mesh_pointclouds.transform(transform_mat)
            source_mesh_pointclouds.transform(grasp_depth)
            multifinger_pcls.append(source_mesh_pointclouds)
        return multifinger_pcls

    def detect(self, multifinger_ggarray, two_fingers_ggarray, path_mesh_json, meshes_pcls, min_grasp_width=0.05, VoxelGrid=0.03, approach_dist=0.04, collision_thresh=10,
               adjust_gripper_centers=False, DEBUG=False):
        ''' Detect collision of grasps.
            Input:
                multifinger_ggarray(class multifingerGraspGroup()): [multifinger_ggarray, M grasps]
                        the grasps to check
                two_fingers_ggarray(class graspnetAPI.GraspGroup): [GraspGroup, M grasps]
                approach_dist: [float]
                        the distance for a gripper to move along approaching direction before grasping
                        this shifting space requires no point either
                collision_thresh: [float]
                        if global collision iou is greater than this threshold,
                        a collision is detected
                adjust_gripper_centers: [bool]
                        if True, add an offset to grasp which makes grasp point closer to object center
            Output:
                empty_mask: [numpy.ndarray, (M,), numpy.bool]
                        True implies empty grasp
                        only returned when [return_empty_grasp] is True
        '''

        T = two_fingers_ggarray.translations
        R = two_fingers_ggarray.rotation_matrices
        heights = two_fingers_ggarray.heights[:, np.newaxis]
        depths = two_fingers_ggarray.depths[:, np.newaxis]
        widths = two_fingers_ggarray.widths[:, np.newaxis]
        targets = self.scene_points[np.newaxis, :, :] - T[:, np.newaxis, :]
        targets = np.matmul(targets, R)

        ## adjust gripper centers
        if adjust_gripper_centers:
            two_fingers_ggarray, targets = self._adjust_gripper_centers(two_fingers_ggarray, targets, heights, depths,
                                                                        widths)
        two_fingers_ggarray.widths = two_fingers_ggarray.widths * 1.7
        min_width_index = two_fingers_ggarray.widths > min_grasp_width
        multifinger_ggarray = multifinger_ggarray[two_fingers_ggarray.widths > min_grasp_width]
        two_fingers_ggarray = two_fingers_ggarray[two_fingers_ggarray.widths > min_grasp_width]

        if len(multifinger_ggarray) == 0:
            print('min_grasp_width filter 0 ')
            return multifinger_ggarray, two_fingers_ggarray, [], min_width_index
        multifinger_ggarray.graspgroupTR_2_TR(two_fingers_ggarray, path_mesh_json)

        meshes_pointclouds_multifinger = self.load_meshes_pcls(meshes_pcls, two_fingers_ggarray, multifinger_ggarray)
        collision_mask = []
        empty_mask = []

        ps = self.scene_cloud.points
        ps = o3d.utility.Vector3dVector(ps)
        st = time.time()
        for idx, grasp in enumerate(multifinger_ggarray):
            meshes_pointclouds = meshes_pointclouds_multifinger[idx]
            grasp_direction = two_fingers_ggarray.rotation_matrices[idx].reshape(3, 3)[:3,0]
            for i in range(2, int(approach_dist * 100)+1, 3):
                meshes_pointclouds_cache = copy.deepcopy(meshes_pointclouds)
                back_distince = [[1, 0, 0, -grasp_direction[0] * i * 0.01],
                                    [0, 1, 0, -grasp_direction[1] * i * 0.01],
                                    [0, 0, 1, -grasp_direction[2] * i * 0.01],
                                    [0, 0, 0, 1]]
                meshes_pointclouds_cache.transform(np.array(back_distince, dtype=np.float32)).paint_uniform_color([0, 1, 0])
                meshes_pointclouds = meshes_pointclouds + meshes_pointclouds_cache
            st3 = time.time()
            meshes_pointclouds = meshes_pointclouds.voxel_down_sample(VoxelGrid)

            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=meshes_pointclouds, voxel_size=VoxelGrid)
            output = voxel_grid.check_if_included(ps)

            if DEBUG:
                print('transformed mesh')
                FOR_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                o3d.visualization.draw_geometries(
                    [self.scene_cloud, meshes_pointclouds, FOR_base, two_fingers_ggarray[int(idx)].to_open3d_geometry()])

            if np.array(output).astype(int).sum() > collision_thresh:
                empty_mask.append(False)
                collision_mask.append(output)
                if DEBUG:
                    print("collision")
                    collision_point_cloud = o3d.geometry.PointCloud()
                    collision_point_cloud.points = o3d.utility.Vector3dVector(
                        np.array(self.scene_cloud.points)[np.array(output)])
                    collision_point_cloud.paint_uniform_color([1, 0, 0])
                    normal_point_cloud = o3d.geometry.PointCloud()
                    normal_point_cloud.points = o3d.utility.Vector3dVector(
                        np.array(self.scene_cloud.points)[~(np.array(output))])
                    normal_point_cloud.paint_uniform_color([0, 0, 1])
                    o3d.visualization.draw_geometries([normal_point_cloud, collision_point_cloud, FOR_base, two_fingers_ggarray[int(idx)].to_open3d_geometry()])
            else:
                empty_mask.append(True)
                collision_mask.append(output)
                
        collision_mask = np.array(collision_mask)
        empty_mask = np.array(empty_mask)
        return multifinger_ggarray, two_fingers_ggarray, empty_mask, min_width_index
