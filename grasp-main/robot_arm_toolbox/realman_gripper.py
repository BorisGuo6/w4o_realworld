import os
import numpy as np
import cv2
import copy
import time
import math
from math import pi
import xlrd2

from .transformation.pose import pose_array_2_matrix, pose_matrix_2_array, translation_rotation_2_matrix, \
    translation_rotation_2_array, translation_euler_2_matrix, matrix_2_translation_euler

from Robotic_Arm.rm_robot_interface import *
from graspnetAPI import Grasp


def to_str(var):
    return '[' + str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1] + ']'


class RealMan_Gripper(RoboticArm):
    def __init__(self, host, gripper_type='eg2'):
        '''
        **Input:**
        - host: string of the ip address of the robot
        - use_rt: use real time mode or not. get_force is only available in realtime mode. Consumes CPU.
        - camera: if "realsense" use old camera interface which take 5 pictures and use the last one.
        - robot_debug: if False, the camera frame is in the depth image. if True,  in the rgb image.
        - RT: the rotation and translation
        '''
        super().__init__(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.rm_create_robot_arm(host, 8080)
        print(self.rm_get_robot_info())
        self.rm_set_gripper_pick(900, 900, True, 10)
        self.rm_set_gripper_pick_on(900, 900, True, 10)
        self.gripper_type = gripper_type

    def get_camera_tcp_matrix(self):
        '''
        **Output:**
        - numpy array of shape (4,4) of the camera tcp transformation matrix
        '''
        # Relative offset from camera center to the gripper center.
        # The realsense center is on one of the two cameras.

        if self.gripper_type == 'eg2': 
            return np.array([[0, 1, 0, -0.0396], [-1, 0, 0, 0.0327], [0, 0, 1, 0.0257], [0, 0, 0, 1]], dtype=np.float32)
            # return np.array([[-0.01339246, 0.99990369, 0.00364167, -0.03925293], [-0.99989991, -0.01340885, 0.00451462, 0.03234782], [0.00456301, -0.00358084, 0.99998318, 0.027677], [0, 0, 0, 1]], dtype=np.float32)
        else:
            raise NotImplementedError

    def get_gripper_tcp_matrix(self):
        '''
        **Output:**
        - numpy array of shape (4,4) of the gripper tcp transformation matrix
        '''
        if self.gripper_type == 'eg2':
            return np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.156], [0, 0, 0, 1]], dtype=np.float32)
        else:
            raise NotImplementedError

    def get_target_gripper_base_pose(self, gripper_camera_pose, use_ready_pose=True):
        '''
        **Input:**
        - gripper_camera_pose: np.array of shape (4,4) of the gripper pose in camera coordinate.
        **Output:**
        - target_gripper_base_pose: target gripper pose in base coordinate, shape=(4,4)
        '''
        return np.dot(
            np.dot(
                self.get_tcp_base_matrix(use_ready_pose=use_ready_pose),
                self.get_camera_tcp_matrix()
            ),
            gripper_camera_pose,
        )

    def gripper_camera_pose_2_tcp_base_pose(self, gripper_camera_pose, use_ready_pose=True):
        '''
        **Input:**
        - gripper_camera_pose: np.array of shape (4,4) of the gripper pose in camera coordinate.
        **Output:**
        - tcp_base_pose: target tcp pose in base coordinate, shape=(4,4)
        '''
        tcp_pose = np.dot(
            np.dot(
                np.dot(
                    self.get_tcp_base_matrix(use_ready_pose=use_ready_pose),  # base / tcp1
                    self.get_camera_tcp_matrix()  # tcp1 / camera
                ),
                gripper_camera_pose  # camera / gripper
            ),
            np.linalg.inv(self.get_gripper_tcp_matrix())  # (tcp2 / gripper)^(-1)
        )

        return tcp_pose

    def tcp_base_pose_2_gripper_camera_pose(self, tcp_base_pose):
        '''
        **Input:**
        - tcp_base_pose: np.array of shape (4,4) of the tcp pose in base coordinate.
        **Output:**
        - gripper_camera_pose: target gripper pose in camera coordinate, shape=(4,4)
        '''
        return np.dot(
            np.dot(
                np.dot(
                    np.linalg.inv(self.get_camera_tcp_matrix()),  # tcp 1/ camera
                    np.linalg.inv(self.get_tcp_base_matrix())  # base / tcp1
                ),
                tcp_base_pose  # base / tcp2
            ),
            self.get_gripper_tcp_matrix(),  # (tcp2 / gripper)^(-1)
        )  # camera / gripper

    def ready_pose(self):
        '''
        **Output:**
        - np.array of shape (6) of the robot pose in ready state.
        '''
        return [0.23, 0.05, 0.45, 3.14, 0, 3.14]

    def throw_pose(self):
        '''
        **Output:**
        - np.array of shape (6) of the robot pose in ready state.
        '''

        return [0.25, -0.3, 0.28, 3.14, 0, 3.14]

    def throw(self, vel=10):
        '''
        **Output:**
        - no output but put the robot to throw pose and open the gripper.
        '''
        self.rm_movel(self.throw_pose(), v=vel, r=0, connect=0, block=1)

    def ready(self, vel=10):
        '''
        **Output:**
        - no output but put the robot to ready state and open the gripper.
        '''
        self.rm_movel(self.ready_pose(), v=vel, r=0, connect=0, block=1)

    def normalize(self, x):
        return np.array([x[0], x[1], x[2]]) / math.sqrt(np.power(x[0], 2) + np.power(x[1], 2) + np.power(x[2], 2))

    def execute(self, grasp, vel=10, approach_dist=0.10):
        '''
        **Input:**
        - grasp: Grasp instance or numpy array of shape (4,4) or numpy array of shape (6,) in base coordinate.
        - acc: float of the maximum acceleration.
        - vel: float of the maximum velocity.
        - approach_dist: float of the distance to move along the z axis of tcp coordinate.
        **Output:**
        - No output but the robot moves to the given pose.
        '''
        if isinstance(grasp, Grasp):
            translation = grasp.translation
            rotation = grasp.rotation_matrix
            pose = translation_rotation_2_array(translation, rotation)
        elif isinstance(grasp, np.ndarray):
            if grasp.shape == (4, 4):
                pose = pose_matrix_2_array(grasp)
            elif grasp.shape == (6,):
                pose = grasp
            else:
                raise ValueError('Shape of Grasp Array must be (4,4) or (6,), but it is {}'.format(grasp.shape))
        else:
            raise ValueError('execute must be called with Grasp or numpy array, but it is {}'.format(type(grasp)))
        tcp_pose = pose_array_2_matrix(pose)
        tcp_pre_pose = copy.deepcopy(tcp_pose)
        tcp_pre_pose[:3, 3] = tcp_pre_pose[:3, 3] - approach_dist * tcp_pre_pose[:3, 2]
        self.rm_movel(tcp_pre_pose, v=vel, r=0, connect=0, block=1)
        self.rm_movel(tcp_pose, v=vel, r=0, connect=0, block=1)

    def execute_camera_pose(self, grasp, vel=10, approach_dist=0.10):
        '''
        **Input:**
        - grasp: Grasp instance or numpy array of shape (4,4) or numpy array of shape (6,) in camera coordinate
        - acc: float of the maximum acceleration.
        - vel: float of the maximum velocity.
        - approach_dist: float of the distance to move along the z axis of tcp coordinate.
        **Output:**
        - No output but the robot moves to the ready pose first, and then moves to the given pose along the z axis of the tcp coordinate.
        '''
        if isinstance(grasp, Grasp):
            translation = grasp.translation
            rotation = grasp.rotation_matrix
            pose = translation_rotation_2_array(translation, rotation)
        elif isinstance(grasp, np.ndarray):
            if grasp.shape == (4, 4):
                pose = pose_matrix_2_array(grasp)
            elif grasp.shape == (6,):
                pose = grasp
            else:
                raise ValueError('Shape of Grasp Array must be (4,4) or (6,), but it is {}'.format(grasp.shape))
        else:
            raise ValueError('execute must be called with Grasp or numpy array, but it is {}'.format(type(grasp)))
        pose = pose_array_2_matrix(pose)
        tcp_pose = self.gripper_camera_pose_2_tcp_base_pose(pose)
        tcp_pre_pose = copy.deepcopy(tcp_pose)
        tcp_pre_pose[:3, 3] = tcp_pre_pose[:3, 3] - approach_dist * tcp_pre_pose[:3, 2]
        tcp_final_pose = copy.deepcopy(tcp_pose)
        tcp_final_pose[:3, 3] = tcp_final_pose[:3, 3] + approach_dist / 3 * tcp_final_pose[:3, 2]
        self.rm_movel(tcp_pre_pose, v=vel, r=0, connect=0, block=1)
        self.rm_movel(tcp_pose, v=vel, r=0, connect=0, block=1)

    def grasp_and_throw(self, grasp, vel=10, approach_dist=0.07, camera_pose=True,
                        execute_grasp=True, use_ready_pose=True):
        '''
        **Input:**
        - grasp: Grasp instance or numpy array of shape (4,4) or numpy array of shape (6,) in camera coordinate
        - vel: float of the maximum velocity.
        - approach_dist: float of the distance to move along the z axis of tcp coordinate.
        - camera_pose: If true, grasp pose is given in camera coordinate. Else, it is given in tcp coordinate.
ss        **Output:**
        - No output but the robot moves to the ready pose first, and then moves to the given pose along the z axis of the tcp coordinate. Maybe it will close the gripper and move up to away pose and finally throw the object.
        '''
        if isinstance(grasp, Grasp):
            translation = grasp.translation
            rotation = grasp.rotation_matrix
            pose = translation_rotation_2_array(translation, rotation)
        elif isinstance(grasp, np.ndarray):
            if grasp.shape == (4, 4):
                pose = pose_matrix_2_array(grasp)
            elif grasp.shape == (6,):
                pose = grasp
            else:
                raise ValueError('Shape of Grasp Array must be (4,4) or (6,), but it is {}'.format(grasp.shape))
        else:
            raise ValueError('execute must be called with Grasp or numpy array, but it is {}'.format(type(grasp)))
        
        pose = pose_array_2_matrix(pose)
        if camera_pose:
            tcp_pose = self.gripper_camera_pose_2_tcp_base_pose(pose, use_ready_pose=use_ready_pose)
        else:
            tcp_pose = copy.deepcopy(pose)
        
        target_gripper_pose = self.normalize(self.get_target_gripper_base_pose(pose, use_ready_pose=use_ready_pose)[:3, 0])
        if self.gripper_type in ['eg2']:
            tcp_pose[:3, 3] = tcp_pose[:3, 3] + (grasp.depth-0.018) * target_gripper_pose

        # to avoid the wire twisted together. flip the gripper which is the same.
        tcp_y_axis_on_base_frame = tcp_pose[1,1] 
        if tcp_y_axis_on_base_frame < 0:
            tcp_pose[:3,0:2] = -tcp_pose[:3,0:2]
        tcp_pre_pose = copy.deepcopy(tcp_pose)
        tcp_pre_pose[:3, 3] = tcp_pre_pose[:3, 3] - approach_dist * target_gripper_pose

        tcp_pose = matrix_2_translation_euler(tcp_pose)
        tcp_pre_pose = matrix_2_translation_euler(tcp_pre_pose)
        t1 = time.time()
        # tcp_pre pose -> tcp pose -> grasp -> tcp_pre pose -> throw pose -> gripper release
        if execute_grasp:
            self.rm_movel(tcp_pre_pose, v=vel, r=0, connect=0, block=1)
            self.rm_movel(tcp_pose, v=vel, r=0, connect=0, block=1)
            # self.rm_set_gripper_position(1, block=True, timeout=20)
            self.rm_movel(tcp_pre_pose, v=vel, r=0, connect=0, block=1)
            self.rm_movel(self.ready_pose(), v=vel, r=0, connect=0, block=1)
            self.rm_movel(self.throw_pose(), v=vel, r=0, connect=0, block=1)
            # self.rm_set_gripper_position(999, block=True, timeout=20)
        
        t2 = time.time()
        print(f'excute grasp time{t2 - t1}')
        return 

    def get_tcp_base_matrix(self, use_ready_pose=False):
        '''
        **Output:**
        - Homogeneous transformation matrix of shape (4,4) for tcp(tool center point) to the base.
        '''
        if use_ready_pose:
            pose = np.array((self.ready_pose()), dtype=np.float32)
        else:
            pose = np.array((self.getl()), dtype=np.float32)
        return translation_euler_2_matrix(pose)