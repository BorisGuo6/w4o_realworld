from rel.cameras.orbbec import Orbbec
from model import GroundedGraspNet
from rel.robots.rw_robot import XArm7RealWorld
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle

import viser
# server = viser.ViserServer()

def transform_object_pose(T_arm_cam, R_obj_cam, t_obj_cam):
    # 构造物体在相机坐标系下的齐次变换矩阵
    T_obj_cam = np.eye(4)
    T_obj_cam[:3, :3] = R_obj_cam
    T_obj_cam[:3, 3] = t_obj_cam

    # 计算逆变换
    R_arm_cam = T_arm_cam[:3, :3]
    t_arm_cam = T_arm_cam[:3, 3]
    T_cam_arm = np.eye(4)
    T_cam_arm[:3, :3] = R_arm_cam.T
    T_cam_arm[:3, 3] = -R_arm_cam.T @ t_arm_cam

    # 计算物体在机械臂坐标系下的变换
    T_obj_arm = T_arm_cam @ T_obj_cam @ T_cam_arm

    # 提取旋转和平移
    R_obj_arm = T_obj_arm[:3, :3]
    t_obj_arm = T_obj_arm[:3, 3]

    print(f"R_obj_arm = {R_obj_arm}")
    print(f"t_obj_arm = {t_obj_arm}")

    return R_obj_arm, t_obj_arm


def compute_goal_pose(t_ee_pose_in, euler_angles, R_transform, t_transform, degrees=False):
    """
    根据给定的变换，计算 gripper 的目标位姿。

    参数：
        t_ee_pose_in   – list 或 array, 长度 3, 当前末端执行器平移 [x, y, z]
        euler_angles   – list 或 array, 长度 3, 当前末端执行器欧拉角 [roll, pitch, yaw]
        R_transform    – array, shape (3,3), 从当前位姿到目标位姿的旋转矩阵
        t_transform    – array, shape (3,),   从当前位姿到目标位姿的平移向量
        degrees        – bool, 默认为 False；如果 True，则输入/输出欧拉角为度 (°)，否则为弧度

    返回：
        t_goal         – array, shape (3,), 目标平移 [x, y, z]
        euler_goal     – array, shape (3,), 目标欧拉角 [roll, pitch, yaw]
    """
    # 1. 构造当前末端的齐次变换矩阵 X_ee
    rot_ee = R.from_euler('xyz', euler_angles, degrees=degrees)
    X_ee = np.eye(4)
    X_ee[:3, :3] = rot_ee.as_matrix()
    X_ee[:3,  3] = t_ee_pose_in

    # 2. 构造给定的变换矩阵 X_trans
    X_trans = np.eye(4)
    X_trans[:3, :3] = R_transform
    X_trans[:3,  3] = t_transform

    # 3. 计算目标齐次变换：X_goal = X_trans @ X_ee
    X_goal = X_trans @ X_ee

    # 4. 提取目标平移和旋转
    t_goal = X_goal[:3, 3]
    rot_goal = R.from_matrix(X_goal[:3, :3])
    euler_goal = rot_goal.as_euler('xyz', degrees=degrees)

    return t_goal, euler_goal

def postprocess_grasp(grasp):

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
        
        coef_in = 0.03
        t_ee_pose_in = t_base_grasp + R_base_grasp.as_matrix()[:3, 2] * coef_in

        coef_out = -0.05
        t_ee_pose_out = t_base_grasp + R_base_grasp.as_matrix()[:3, 2] * coef_out

        # R_base_grasp to euler angle
        euler_angles = R_base_grasp.as_euler('xyz', degrees=True)
        print(f"euler_angles = {euler_angles}")

        ee_pose = np.concatenate([t_base_grasp, euler_angles], axis=0)
        ee_pose_in = np.concatenate([t_ee_pose_in, euler_angles], axis=0)
        ee_pose_out = np.concatenate([t_ee_pose_out, euler_angles], axis=0)
        print(f"ee_pose = {ee_pose}")
        
        # convert to viser format

        # convert R_base_grasp to quaternion
        quat = R_base_grasp.as_quat()[[3, 0, 1, 2]]
        print(f"quat = {quat}")
        # visualize ee_pose

        server = viser.ViserServer()

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

        # breakpoint()

        # close server
        server.stop()
        
        # # convert grasp to end effector pose, [x,y,z,qx,qy,qz,qw]
        # rotation = torch.tensor(rotation)
        # quat = transforms.matrix_to_quaternion(rotation).numpy() # w,x,y,z
        # #  convert quat to x,y,z,w 
        # quat = np.roll(quat, -1)
        # ee_pose = np.concatenate([translation, quat], axis=0)
        
        return ee_pose, ee_pose_in, ee_pose_out


def get_input():
    

def segment():
    pass



def main():
    image, depth, text, transformation = get_input()
    mask = segment(image, text)
    ee_pose_grasp = graspnetv2(image, depth, mask)
    postprocess_grasp(ee_pose_grasp)

if __name__ == "__main__":
    main()