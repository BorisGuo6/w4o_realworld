# from rel.cameras.orbbec import Orbbec
# from model import GroundedGraspNet
from rel.robots.rw_robot import XArm7RealWorld
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle

import viser
import sys
from pathlib import Path
import os

PROJECT_ROOT = Path('/home/world4omni/w4o')       # should be ~/w4o
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
GRIPPER_MODEL_PATH = PROJECT_ROOT / "xarm7" / "assets" / "xarm_gripper" / "hand_open.obj"
X_BaseCamera_manual_path = f"{PROJECT_ROOT}/rwVR/data/cameras/CL8H74100BB/0919_excalib_capture00/optimized_X_BaseCamera.npy"
sys.path.insert(0, str(PROJECT_ROOT))
from World4Omni_rw.tools.get_new import get_newest


def arm_init(xarm):
    xarm.arm.set_position(
        198.9,
        -0.1,
        259.9,
        178.2,
        -0.6,
        1.3,
        wait=True,
        is_radian=False,
        speed=150,
    )


def calculate_gripper_pose_in_base(
    X_BaseCamera: np.ndarray,
    grasp_pose_in_camera: np.ndarray,
    transformation_in_camera: np.ndarray
) -> np.ndarray:
    """
    计算机器人夹爪在经历一次变换后的、相对于基座(Base)的最终位姿。

    Args:
        X_BaseCamera (np.ndarray): 
            一个 4x4 的齐次变换矩阵，表示相机坐标系相对于机器人基座坐标系(Base)的位姿。
            即: Base_T_Camera

        grasp_pose_in_camera (np.ndarray): 
            一个 4x4 的齐次变换矩阵，表示夹爪的初始位姿(姿态1)，该位姿是在相机坐标系下描述的。
            即: Camera_T_Gripper1

        transformation_in_camera (np.ndarray): 
            一个 4x4 的齐次变换矩阵，表示夹爪从姿态1到姿态2的变换过程，该变换是在相机坐标系下描述的。
            即: Delta_T_in_Camera

    Returns:
        np.ndarray: 
            一个 4x4 的齐次变换矩阵，表示夹爪的最终位姿(姿态2)，该位姿是在机器人基座坐标系(Base)下描述的。
            即: Base_T_Gripper2
    """
    # 验证输入矩阵是否都是4x4
    for matrix, name in [
        (X_BaseCamera, "X_BaseCamera"),
        (grasp_pose_in_camera, "grasp_pose_in_camera"),
        (transformation_in_camera, "transformation_in_camera"),
    ]:
        if matrix.shape != (4, 4):
            raise ValueError(f"输入矩阵 '{name}' 的形状必须是 (4, 4)，但现在是 {matrix.shape}")

    # 第一步：计算 Gripper 的姿态2 在相机坐标系下的位姿
    # Camera_T_Gripper2 = Delta_T_in_Camera * Camera_T_Gripper1
    # 使用 @ 运算符进行矩阵乘法，它在 NumPy 中专用于此。
    
    print('pose2_in_camera:\n', grasp_pose_in_camera)
    pose2_in_camera = transformation_in_camera @ grasp_pose_in_camera
    # pose2_in_camera = grasp_pose_in_camera.copy()
    print('transformation_in_camera:\n', transformation_in_camera)
    # pose2_in_camera[:3, 3] += transformation_in_camera[:3, 3]
    print('pose2_in_camera:\n', pose2_in_camera)

    # 第二步：将 Gripper 的姿态2 从相机坐标系转换到 Base 坐标系
    # Base_T_Gripper2 = Base_T_Camera * Camera_T_Gripper2
    pose2_in_base = X_BaseCamera @ pose2_in_camera

    # server = viser.ViserServer()
    # server.scene.add_frame(
    #     "grasp_pose_in_camera",
    #     wxyz=R.from_matrix(grasp_pose_in_camera[:3, :3]).as_quat()[[3, 0, 1, 2]],
    #     position=grasp_pose_in_camera[:3, 3],
    # )

    # server.scene.add_frame(
    #     "pose2_in_camera",
    #     wxyz=R.from_matrix(pose2_in_camera[:3, :3]).as_quat()[[3, 0, 1, 2]],
    #     position=pose2_in_camera[:3, 3],
    # )
    # input()
    # assert 0

    t_goal = pose2_in_base[:3, 3]
    rot_goal = R.from_matrix(pose2_in_base[:3, :3])
    euler_goal = rot_goal.as_euler('xyz', degrees=True)
    gripper_pose = np.concatenate([t_goal, euler_goal], axis=0)
    return pose2_in_base, gripper_pose


def calculate_gripper_pose_in_base_v2(
    pose_of_base_in_camera: np.ndarray,
    grasp_pose_in_camera: np.ndarray,
    transformation_in_camera: np.ndarray
) -> np.ndarray:
    """
    计算机器人夹爪在经历一次变换后的、相对于基座(Base)的最终位姿。
    这个版本假设输入的第一个矩阵是 Base 在 Camera 坐标系下的位姿。

    Args:
        pose_of_base_in_camera (np.ndarray): 
            一个 4x4 的齐次变换矩阵，表示机器人基座(Base)坐标系相对于相机坐标系的位姿。
            即: Camera_T_Base  <-- 注意！这里的定义与原函数不同。

        grasp_pose_in_camera (np.ndarray): 
            一个 4x4 的齐次变换矩阵，表示夹爪的初始位姿(姿态1)，该位姿是在相机坐标系下描述的。
            即: Camera_T_Gripper1

        transformation_in_camera (np.ndarray): 
            一个 4x4 的齐次变换矩阵，表示夹爪从姿态1到姿态2的变换过程，该变换是在相机坐标系下描述的。
            即: Delta_T_in_Camera

    Returns:
        np.ndarray: 
            一个 4x4 的齐次变换矩阵，表示夹爪的最终位姿(姿态2)，该位姿是在机器人基座坐标系(Base)下描述的。
            即: Base_T_Gripper2
    """
    # 验证输入矩阵是否都是4x4
    for matrix, name in [
        (pose_of_base_in_camera, "pose_of_base_in_camera"),
        (grasp_pose_in_camera, "grasp_pose_in_camera"),
        (transformation_in_camera, "transformation_in_camera"),
    ]:
        if matrix.shape != (4, 4):
            raise ValueError(f"输入矩阵 '{name}' 的形状必须是 (4, 4)，但现在是 {matrix.shape}")

    # 第一步：计算 Gripper 的姿态2 在相机坐标系下的位姿 (此步骤不变)
    # Camera_T_Gripper2 = Delta_T_in_Camera * Camera_T_Gripper1
    pose2_in_camera = transformation_in_camera @ grasp_pose_in_camera

    # 第二步：将 Gripper 的姿态2 从相机坐标系转换到 Base 坐标系
    
    # --- 主要修改点在这里 ---
    # 我们需要的是 Base_T_Camera，但输入的是 Camera_T_Base。
    # 因此，我们需要对输入矩阵求逆。
    # Base_T_Camera = inverse(Camera_T_Base)
    base_T_camera = np.linalg.inv(pose_of_base_in_camera)
    
    # 使用求逆后的矩阵进行坐标变换
    # Base_T_Gripper2 = Base_T_Camera * Camera_T_Gripper2
    pose2_in_base = base_T_camera @ pose2_in_camera

    return pose2_in_base


def create_transformation_matrix(
    t_goal: list | tuple | np.ndarray,
    euler_goal: list | tuple | np.ndarray,
    convention: str = 'zyx',
    degrees: bool = False
) -> np.ndarray:
    """
    根据给定的平移向量和欧拉角，构建一个4x4的齐次变换矩阵。

    Args:
        t_goal (list | tuple | np.ndarray): 
            目标平移向量 [x, y, z]，长度为3。

        euler_goal (list | tuple | np.ndarray): 
            目标欧拉角，长度为3。其顺序由 'convention' 参数决定。

        convention (str, optional): 
            欧拉角的旋转顺序。默认为 'zyx'，这在机器人学中常对应 Yaw, Pitch, Roll。
            'scipy' 支持多种顺序，如 'xyz', 'zyx', 'zxz', 'yzx' 等（均为小写）。

        degrees (bool, optional): 
            如果为 True，则输入的欧拉角被视为角度；否则视为弧度。默认为 False (弧度)。

    Returns:
        np.ndarray: 
            构建好的 4x4 齐次变换矩阵。
            
    Raises:
        ValueError: 如果输入向量的长度不为3。
    """
    if len(t_goal) != 3 or len(euler_goal) != 3:
        raise ValueError("平移向量和欧拉角向量的长度都必须是3。")

    # 1. 从欧拉角创建3x3旋转矩阵
    # 使用scipy的Rotation类可以方便、准确地处理各种旋转顺序
    try:
        rotation = R.from_euler(convention, euler_goal, degrees=degrees)
        rotation_matrix = rotation.as_matrix()
    except Exception as e:
        print(f"从欧拉角创建旋转矩阵时出错: {e}")
        print("请检查 'convention' 字符串是否有效 (例如, 'zyx', 'xyz', 'zxz' 等)")
        raise

    # 2. 构建4x4齐次变换矩阵
    # 创建一个4x4的单位矩阵作为基础
    transformation_matrix = np.identity(4)
    
    # 将3x3旋转矩阵填充到左上角
    transformation_matrix[0:3, 0:3] = rotation_matrix
    
    # 将平移向量填充到最后一列
    transformation_matrix[0:3, 3] = t_goal
    
    return transformation_matrix


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

    quaternion_goal = rot_goal.as_quat() 

    return t_goal, euler_goal, quaternion_goal


def mat_from_cam_to_base(x_camera_transformation: np.ndarray, x_base_camera: np.ndarray) -> np.ndarray:
    """
    Transformation矩阵需要两面夹！

    将一个变换矩阵从相机坐标系转换到机器人基础坐标系。

    Args:
        transformation_in_cam (np.ndarray): 描述物体相对于相机坐标系的 4x4 齐次变换矩阵。
        x_base_camera (np.ndarray): 描述相机坐标系相对于机器人基础坐标系的 4x4 齐次变换矩阵。

    Returns:
        np.ndarray: 描述物体相对于机器人基础坐标系的 4x4 齐次变换矩阵。
    """
    # 检查输入矩阵的形状是否正确
    if x_camera_transformation.shape != (4, 4) or x_base_camera.shape != (4, 4):
        raise ValueError("输入矩阵必须是 4x4 的齐次变换矩阵。")

    # 核心转换公式: T_base_obj = T_base_cam * T_cam_obj
    # T_base_cam 是 x_base_camera
    # T_cam_obj 是 x_camera_transformation
    # 使用 @ 运算符进行矩阵乘法
    x_base_transformation = x_base_camera @ x_camera_transformation @ np.linalg.inv(x_base_camera)
    
    t_base_transformation = x_base_transformation[:3, 3]
    R_base_transformation = R.from_matrix(x_base_transformation[:3, :3])
    euler_angles = R_base_transformation.as_euler('xyz', degrees=True)
    print(f"euler_angles = {euler_angles}")

    ee_pose_trans = np.concatenate([t_base_transformation, euler_angles], axis=0)
    print(f"ee_pose_trans = {ee_pose_trans}")


    # convert R_base_grasp to quaternion
    quat = R_base_transformation.as_quat()[[3, 0, 1, 2]]
    print(f"quat = {quat}")
    # visualize ee_pose

    # server = viser.ViserServer()

    # R_base_camera = R.from_matrix(x_base_camera[:3, :3])
    # t_base_camera = x_base_camera[:3, 3]
    # server.scene.add_frame(
    #     "cam_pose",
    #     wxyz=R_base_camera.as_quat()[[3, 0, 1, 2]],
    #     position=t_base_camera,
    # )

    # server.scene.add_frame(
    #     "ee_pose",
    #     wxyz=quat,
    #     position=t_base_transformation,
    # )

    return x_base_transformation, ee_pose_trans


def get_cam_in_base():
    """
    Get the transformation matrix from camera frame to base frame.
    """
    X_BaseCamera = np.load(X_BaseCamera_manual_path)
    return X_BaseCamera


def motion_planning(X_ee_pose_cur, X_ee_pose_trans):
    X_ee_pose_goal = X_ee_pose_cur @ X_ee_pose_trans

    R_goal_ee_pose = X_ee_pose_goal[:3, :3]

    t_goal_ee_pose = X_ee_pose_goal[:3, 3]


    quat_cur = R.from_matrix(X_ee_pose_cur[:3, :3]).as_quat()[[3, 0, 1, 2]]
    t_cur = X_ee_pose_cur[:3, 3]

    quat_goal = R.from_matrix(R_goal_ee_pose).as_quat()[[3, 0, 1, 2]]
    t_goal = t_goal_ee_pose

    server = viser.ViserServer()

    server.scene.add_frame(
        "ee_pose_cur",
        wxyz=quat_cur,
        position=t_cur,
    )

    server.scene.add_frame(
        "ee_pose_goal",
        wxyz=quat_goal,
        position=t_goal,
    )
    input()



def postprocess_grasp(grasp):
        """
        Postprocess the grasp output from the model.
        Input: grasp (4x4 numpy array), in camera frame
        Output: grasp (4x4 numpy array), in arm base frame
        """
        print(grasp)

        ## grasp pose in camera frame
        translation_camera_grasp = grasp[:3, 3]
        rotation_camera_grasp= grasp[:3, :3]

        # Due to definition of URDF, there's an offset along the gripper depth direction
        gripper_depth_offset = 0.16 # gripper length
        # The Z-axis of the grasp frame in camera coordinates is the 3rd column of the rotation matrix.
        grasp_z_axis_in_camera = rotation_camera_grasp[:, 2]
        translation_camera_grasp += grasp_z_axis_in_camera * gripper_depth_offset
        print(f"Applied gripper depth offset. New translation in camera frame: {translation_camera_grasp}")
        
        # R_fix = np.array([[ -1.,  0.,  0.],       
        #                   [  0.,  0., -1.],       
        #                   [  0.,  1.,  0.]]) # R_fix adopted by RLBench, used by graspnet_rlbench.py

        # # R_fix = np.array([[  0.,  0.,  1.],       
        # #                   [  0., -1.,  0.],       
        # #                   [ -1.,  0.,  0.]]) # R_fix adopted by Curobo, used by graspnet_rlbench_curobo.py
        
        # rotation = np.dot(R_fix, rotation)

        # For real-world implementation, we need to rotate the grasp to align with the robot's end effector frame
        
        
        ## camera pose in base camera frame
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


        R_base_grasp = R_base_grasp.as_matrix() # @ R.from_euler('y', 90, degrees=True).as_matrix() # @ R.from_euler('z', -90, degrees=True).as_matrix()
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

        # server = viser.ViserServer()

        # server.scene.add_frame(
        #     "cam_pose",
        #     wxyz=R_base_camera.as_quat()[[3, 0, 1, 2]],
        #     position=translation_base_camera,
        # )

        # server.scene.add_frame(
        #     "ee_pose",
        #     wxyz=quat,
        #     position=t_base_grasp,
        # )

        # server.scene.add_frame(
        #     "ee_pose_in",
        #     wxyz=quat,
        #     position=t_ee_pose_in,
        # )

        # # breakpoint()
        # input()

        # # close server
        # server.stop()
        
        # # convert grasp to end effector pose, [x,y,z,qx,qy,qz,qw]
        # rotation = torch.tensor(rotation)
        # quat = transforms.matrix_to_quaternion(rotation).numpy() # w,x,y,z
        # #  convert quat to x,y,z,w 
        # quat = np.roll(quat, -1)
        # ee_pose = np.concatenate([translation, quat], axis=0)

        return ee_pose, ee_pose_in, ee_pose_out, R_base_grasp.as_matrix(), t_base_grasp


def main():
    basename = get_newest(RAW_DATA_DIR)
    # basename = "20250912_132100"
    # basename = "20250913_142245"
    ## camera pose in base camera frame
    transformation_path = os.path.join(RAW_DATA_DIR, basename, "transformation.pkl")

    xarm = XArm7RealWorld()
    # arm_init(xarm)

    X_BaseCamera = get_cam_in_base()
    with open(transformation_path, 'rb') as f:
        R_transform, t_transform = pickle.load(f)

    print(f"R_transform = {R_transform}")
    print(f"t_transform = {t_transform}")

    R_obj_arm, t_obj_arm = transform_object_pose(T_arm_cam=X_BaseCamera, R_obj_cam=R_transform, t_obj_cam=t_transform)

    x_camera_transformation = np.eye(4)
    x_camera_transformation[:3, :3] = R_transform
    x_camera_transformation[:3, 3] = t_transform
    x_base_transformation, ee_pose_trans = mat_from_cam_to_base(x_camera_transformation, X_BaseCamera)       # in and out is offset by gripper depth
    grasp_path = f"{RAW_DATA_DIR}/{basename}/grasps_in_cam.npz"
    x_cam_grasp = np.load(grasp_path)['grasps'][1]  # (4,4) numpy array
    grasp_in_base_frame, ee_pose_in, ee_pose_out, R_base_grasp, t_base_grasp = postprocess_grasp(x_cam_grasp) 
    x_base_grasp = np.eye(4)
    x_base_grasp[:3, :3] = R_base_grasp
    x_base_grasp[:3, 3] = t_base_grasp
    # motion_planning(x_base_grasp, x_base_transformation)

    # t_goal, euler_goal, quaternion_goal = compute_goal_pose(ee_pose_in[:3], ee_pose_in[3:], R_obj_arm, t_obj_arm, degrees=True)
    # print(f"t_goal = {t_goal}")
    # print(f"euler_goal = {euler_goal}")
    # goal_pose_in_base = create_transformation_matrix(t_goal, euler_goal)

    # x_base_pose2, gripper_goal_pose = calculate_gripper_pose_in_base(X_BaseCamera, x_cam_grasp, x_camera_transformation)
    # x_base_pose2v2 = calculate_gripper_pose_in_base_v2(X_BaseCamera, x_cam_grasp, x_camera_transformation)


    server = viser.ViserServer()

    server.scene.add_frame(
        "cam_pose",
        wxyz=R.from_matrix(X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]],
        position=X_BaseCamera[:3, 3],
    )

    server.scene.add_frame(
        "grasp_pose_in_base",
        wxyz=R.from_matrix(R_base_grasp).as_quat()[[3, 0, 1, 2]],
        position=t_base_grasp,
    )

    # # server.scene.add_frame(
    # #     "goal_pose_from_euler",
    # #     wxyz=R.from_matrix(R_obj_arm).as_quat()[[3, 0, 1, 2]],
    # #     position=t_obj_arm,
    # # )

    # # server.scene.add_frame(
    # #     "goal_pose_from_X",
    # #     wxyz=R.from_matrix(x_base_transformation[:3, :3]).as_quat()[[3, 0, 1, 2]],
    # #     position=x_base_transformation[:3, 3],
    # # )

    # server.scene.add_frame(
    #     "grasp_pose_in_cam",
    #     wxyz=R.from_matrix(x_cam_grasp[:3, :3]).as_quat()[[3, 0, 1, 2]],
    #     position=x_cam_grasp[:3, 3],
    # )

    # pose2_in_camera = x_camera_transformation @ x_cam_grasp

    # server.scene.add_frame(
    #     "goal_pose2_in_cam",
    #     wxyz=R.from_matrix(pose2_in_camera[:3, :3]).as_quat()[[3, 0, 1, 2]],
    #     position=pose2_in_camera[:3, 3],
    # )

    # server.scene.add_frame(
    #     "goal_pose_from_X2",
    #     wxyz=R.from_matrix(x_base_pose2[:3, :3]).as_quat()[[3, 0, 1, 2]],
    #     position=x_base_pose2[:3, 3],
    # )

    # server.scene.add_frame(
    #     "goal_pose_from_X2v2",
    #     wxyz=R.from_matrix(x_base_pose2v2[:3, :3]).as_quat()[[3, 0, 1, 2]],
    #     position=x_base_pose2v2[:3, 3],
    # )

    # server.scene.add_frame(
    #     "goal_pose_in_base",
    #     wxyz=quaternion_goal,
    #     position=t_goal,
    # )

    R_goal_2_in_cam = R_transform @ x_cam_grasp[:3, :3]
    t_goal_2_in_cam = x_cam_grasp[:3, 3] + t_transform 
    print(f"t_goal_2_in_cam = {t_goal_2_in_cam}")
    print(x_cam_grasp[:3, 3])
    print(R_transform, t_transform)
    print(t_goal_2_in_cam - x_cam_grasp[:3, 3])

    # R_transform = np.eye(3)
    # R_goal_2 = X_BaseCamera[:3, :3] @ R_transform @ x_cam_grasp[:3, :3]  # R_cam_to_base @ R_trans_in_cam @ R_grasp_in_cam
    # t_goal_2 = X_BaseCamera[:3, :3] @ (R_transform @ x_cam_grasp[:3, 3] + t_transform) + X_BaseCamera[:3, 3]
    # euler_goal_2 = R.from_matrix(R_goal_2).as_euler('zyx', degrees=True)


    # x_base_goal = x_base_transformation @ x_base_grasp
    x_base_goal = np.eye(4)
    x_base_goal[:3, :3] = X_BaseCamera[:3, :3] @ R_goal_2_in_cam
    x_base_goal[:3, 3] = X_BaseCamera[:3, :3] @ t_goal_2_in_cam * 0.88 + X_BaseCamera[:3, 3]
    print(f"x_base_goal = {x_base_goal}")
    # print("x_base_transformation: ", x_base_transformation)
    t_goal_2 = x_base_goal[:3, 3]* 0.88 # scale down to mitigate the overshoot issue caused by camera
    R_goal_2 = x_base_goal[:3, :3]
    euler_goal_2 = R.from_matrix(R_goal_2).as_euler('xyz', degrees=True)

    save_goal_path = f"{RAW_DATA_DIR}/{basename}/goal_pose_in_base.npz"       # (xyz, euler)
    np.savez(save_goal_path, goal_pose=np.concatenate([t_goal_2, euler_goal_2], axis=0))
    print(f"Saved goal pose in base frame to {save_goal_path}")

    server.scene.add_frame(
        "goal_obj_pose",
        wxyz=R.from_matrix(R_goal_2[:3, :3]).as_quat()[[3, 0, 1, 2]],
        position=t_goal_2,
    )
    
    server.scene.add_frame(
        "grasp_obj_pose_in_cam",
        wxyz=R.from_matrix(x_cam_grasp[:3, :3]).as_quat()[[3, 0, 1, 2]],
        position=x_cam_grasp[:3, 3],
    )

    server.scene.add_frame(
        "goal_obj_pose_in_cam",
        wxyz=R.from_matrix(R_goal_2_in_cam[:3, :3]).as_quat()[[3, 0, 1, 2]],
        position=t_goal_2_in_cam,
    )

    server.scene.add_frame(
        "x_base_goal",
        wxyz=R.from_matrix(x_base_goal[:3, :3]).as_quat()[[3, 0, 1, 2]],
        position=x_base_goal[:3, 3],
    )

    input()
    # xarm.arm.set_position(
    #     gripper_goal_pose[0] * 1000, 
    #     gripper_goal_pose[1] * 1000, 
    #     gripper_goal_pose[2] * 1000, 
    #     gripper_goal_pose[3],
    #     gripper_goal_pose[4],
    #     gripper_goal_pose[5],
    #     wait=True,
    #     is_radian=False,
    #     speed=100,
    # )

    # xarm.arm.set_position(
    #     t_goal_2[0] * 1000 - 200, 
    #     t_goal_2[1] * 1000 - 200, 
    #     t_goal_2[2] * 1000 + 200, 
    #     euler_goal_2[0],
    #     euler_goal_2[1],
    #     euler_goal_2[2],
    #     wait=True,
    #     is_radian=False,
    #     speed=100,
    # )

if __name__ == "__main__":
    main()