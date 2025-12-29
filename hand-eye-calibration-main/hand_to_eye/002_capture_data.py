import cv2
import time
import os
import sys
import viser 
import numpy as np 
import open3d as o3d
from loguru import logger as lgr
from enum import Enum, auto
from scipy.spatial.transform import Rotation as R
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from utils.realsense import Realsense

from config.config import HandToEyeCalibConfig

# Dynamic import robot class based on configuration
from importlib import import_module
def get_robot_class_pk():
    """Dynamically imports and returns the robot class from config."""
    module = import_module("utils.arm_pk")
    return getattr(module, HandToEyeCalibConfig.robot_class_pk)
def get_robot_class_rw():
    """Dynamically imports and returns the robot class from config."""
    module = import_module("utils.arm_rw")
    return getattr(module, HandToEyeCalibConfig.robot_class_rw)
RobotPK = get_robot_class_pk()
RobotRW = get_robot_class_rw()

class VerificationState(Enum):
    WAITING_FOR_VERIFICATION = auto()
    VERIFIED_YES = auto()
    VERIFIED_NO = auto()


def update_viser_current_arm(sv:viser.ViserServer, arm_pk, current_joint_values, camera, init_camera_pose):
    current_arm_mesh = arm_pk.get_state_trimesh(current_joint_values, visual=True, collision=False)["visual"]
    sv.scene.add_mesh_trimesh("current_arm_mesh", current_arm_mesh)

    # add the camera pose
    camera_wxyz = R.from_matrix(init_camera_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
    camera_pos = init_camera_pose[:3, 3]
    rtr_dict = camera.getCurrentData(pointcloud=True)
    rs_rgb = rtr_dict["rgb"]
    rgb_bgr = cv2.cvtColor(rs_rgb, cv2.COLOR_RGB2BGR) 
    rs_pc = rtr_dict["pointcloud_o3d"]

    sv.scene.add_camera_frustum("rs_camera_img", fov=camera.fov_x, aspect=camera.aspect_ratio, wxyz=camera_wxyz, position=camera_pos, image=rgb_bgr, scale=0.2)
    rs_pc_in_C_np = np.asarray(rs_pc.points) 
    rs_pc_in_B = init_camera_pose @ np.vstack([rs_pc_in_C_np.T, np.ones(rs_pc_in_C_np.shape[0])])
    rs_pc_in_B = rs_pc_in_B[:3].T
    sv.scene.add_point_cloud("rs_camera_pc", rs_pc_in_B, colors=[200, 50, 50], point_size=0.005)

# Setup
serial_number = HandToEyeCalibConfig.serial_number
exp_name = HandToEyeCalibConfig.exp_name
camera_data_path = (HandToEyeCalibConfig.save_data_path / HandToEyeCalibConfig.exp_name).resolve()
init_camera_pose_path = (camera_data_path / "init_camera_pose.npy").resolve()
''' setup the realworld arm and camera '''
# Realworld xarm
robot_rw = RobotRW(ip=HandToEyeCalibConfig.robot_ip)
# Setup camera
camera = Realsense(HandToEyeCalibConfig.serial_number)
if not init_camera_pose_path.exists():
    raise FileNotFoundError(f"Missing initial pose: {init_camera_pose_path}")
init_camera_pose = np.load(init_camera_pose_path)
lgr.info("Loaded init_camera_pose: \n{}".format(init_camera_pose))
# Setup simulation
robot_pk = RobotPK(urdf_path=HandToEyeCalibConfig.urdf_path)
sv = viser.ViserServer()
button_verify_yes = sv.gui.add_button("verify_yes")
buttion_exit = sv.gui.add_button("exit")


def set_verification_state(state):
    global verification_state
    verification_state = state
    lgr.info(f"Verification state: {verification_state}")

def on_exit():
    exit()


n_collected_sample = 0

def save_current_data():
    global n_collected_sample
    rt_dict = camera.getCurrentData(pointcloud=True)
    sample_dir_path = camera_data_path / f"{n_collected_sample:04d}"  # Use existing variable
    sample_dir_path.mkdir(parents=True, exist_ok=True)
    rgb_image = rt_dict["rgb"]
    depth_image = rt_dict["depth"]
    pc_o3d = rt_dict["pointcloud_o3d"]

    np.save(sample_dir_path / "joint_values.npy", robot_rw.get_joint_values())
    cv2.imwrite((sample_dir_path / "rgb_image.jpg").as_posix(), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    np.save(sample_dir_path / "depth_image.npy", depth_image)
    o3d.io.write_point_cloud((sample_dir_path / "point_cloud.ply").as_posix(), pc_o3d)
    n_collected_sample += 1

button_verify_yes.on_click(lambda _: save_current_data())
buttion_exit.on_click(lambda _: on_exit())

while True: 
    current_joint_values = robot_rw.get_joint_values()
    update_viser_current_arm(sv, robot_pk, current_joint_values, camera, init_camera_pose)
    time.sleep(0.4)

