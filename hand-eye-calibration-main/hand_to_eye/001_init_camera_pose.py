import sys
import os
import time
import viser
import sapien
import numpy as np 
from pathlib import Path
from loguru import logger as lgr
from PIL import ImageColor
from scipy.spatial.transform import Rotation as R
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from utils.realsense import Realsense
from utils.arm_mplib import MPlibConfig, RobotMPlib
from utils.transform_utils import convert_camera_pose_z_forward_to_sapien

from config.config import HandToEyeCalibConfig

# Dynamic import robot class based on configuration
from importlib import import_module
def get_robot_class():
    """Dynamically imports and returns the robot class from config."""
    module = import_module("utils.arm_pk")
    return getattr(module, HandToEyeCalibConfig.robot_class_pk)
RobotPK = get_robot_class()  # Expose the selected class


# Initial camera pose guess
camera_pose = np.eye(4)

# Initial rotation guess
rotation_matrix = R.from_euler('x', HandToEyeCalibConfig.initial_rotation[0], degrees=True).as_matrix() \
                @ R.from_euler('y', HandToEyeCalibConfig.initial_rotation[1], degrees=True).as_matrix() \
                @ R.from_euler('z', HandToEyeCalibConfig.initial_rotation[2], degrees=True).as_matrix()
camera_pose[0:3, 0:3] = rotation_matrix
# Initial translation guess
camera_pose[0:3, 3] = np.array([HandToEyeCalibConfig.initial_translation[0], HandToEyeCalibConfig.initial_translation[1], HandToEyeCalibConfig.initial_translation[2]])

lgr.info(f'Initial Camera Pose in Base = {camera_pose}')

# Initialize realsense camera
camera = Realsense(HandToEyeCalibConfig.serial_number)

# Initialize data saving paths
save_data_path = HandToEyeCalibConfig.save_data_path / HandToEyeCalibConfig.exp_name
save_init_pose_np_path = save_data_path / "init_camera_pose.npy"
if not save_data_path.exists():
    save_data_path.mkdir(parents=True)

# Setup simulation environment
robot_pk = RobotPK(
    urdf_path=HandToEyeCalibConfig.urdf_path
)
robot_pk_planner_cfg = MPlibConfig(
    urdf_path=HandToEyeCalibConfig.urdf_path,
    vis=False
)
robot_pk_planner = RobotMPlib(robot_pk_planner_cfg)
# Setup SAPIEN camera
sp_camera = robot_pk_planner.scene.add_camera(
        name="rw_camera", # real world camera
        width=camera.w,
        height=camera.h,
        fovy=camera.fov_y,
        near=0.001,
        far=10.0,
    )
sapien_cam_pose = convert_camera_pose_z_forward_to_sapien(camera_pose)
sp_camera.entity.set_pose(sapien.Pose(sapien_cam_pose))
sp_camera.set_focal_lengths(camera.intr.fx, camera.intr.fy)
sp_camera.set_principal_point(camera.intr.ppx, camera.intr.ppy)

sv = viser.ViserServer()
camera_wxyz = R.from_matrix(camera_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
camera_pos = camera_pose[:3, 3]
camera_control = sv.scene.add_transform_controls(
    f"camera_pose",
    opacity=0.75,
    disable_sliders=True,
    scale=0.25,
    line_width=2.5,
    wxyz=camera_wxyz,
    position=camera_pos,
)

joint_gui_handles = []

def update_robot_trimesh_camera_and_mask(joint_values):
    robot_pk_planner.sp_robot.set_qpos(joint_values)
    robot_pk_planner.scene.update_render()  # sync pose from SAPIEN to renderer
    sp_camera.take_picture()  # submit rendering jobs to the GPU
    rgba = sp_camera.get_picture("Color")  # [H, W, 4]
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    seg_labels = sp_camera.get_picture("Segmentation")  # [H, W, 4]
    actor_seg = seg_labels[..., 1].astype(np.uint8)  # actor-level
    colormap = sorted(set(ImageColor.colormap.values()))
    color_palette = np.array(
        [ImageColor.getrgb(color) for color in colormap], dtype=np.uint8
    )
    actor_and_bg_seg = actor_seg > 1  
    actor_and_bg_seg_img = actor_and_bg_seg.astype(np.uint8) * 255
    actor_and_bg_seg_img = np.stack([actor_and_bg_seg_img]*3, axis=-1)

    trimesh_dict = robot_pk.get_state_trimesh(
        joint_values,
        visual=True,
        collision=True,
    )
    visual_mesh = trimesh_dict["visual"]
    collision_mesh = trimesh_dict["collision"]
    sv.scene.add_mesh_trimesh("visual_mesh", visual_mesh)
    sv.scene.add_mesh_trimesh("collision_mesh", collision_mesh)
    camera_wxyz = R.from_matrix(camera_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
    camera_pos = camera_pose[:3, 3]
    sv.scene.add_camera_frustum("sp_camera_img", fov=camera.fov_x, aspect=camera.aspect_ratio, wxyz=camera_wxyz, position=camera_pos, image=rgba_img, scale=0.2)
    sv.scene.add_camera_frustum("sp_camera_seg", fov=camera.fov_x, aspect=camera.aspect_ratio, wxyz=camera_wxyz, position=camera_pos, image=color_palette[actor_seg], scale=0.2)
    sv.scene.add_camera_frustum("sp_camera_actor_and_bg_seg", fov=camera.fov_x, aspect=camera.aspect_ratio, wxyz=camera_wxyz, position=camera_pos, image=actor_and_bg_seg_img, scale=0.2)

    rtr_dict = camera.getCurrentData(pointcloud=True)
    rs_rgb = rtr_dict["rgb"]
    rs_pc = rtr_dict["pointcloud_o3d"]

    sv.scene.add_camera_frustum("rs_camera_img", fov=camera.fov_x, aspect=camera.aspect_ratio, wxyz=camera_wxyz, position=camera_pos, image=rs_rgb, scale=0.2)
    rs_pc_in_C_np = np.asarray(rs_pc.points) 
    rs_pc_in_B = camera_pose @ np.vstack([rs_pc_in_C_np.T, np.ones(rs_pc_in_C_np.shape[0])])
    rs_pc_in_B = rs_pc_in_B[:3].T
    
    
    sv.scene.add_point_cloud("rs_camera_pc", rs_pc_in_B, colors=np.asarray(rs_pc.colors), point_size=0.005)


def update_camera_pose(camera_wxyz, camera_pos):
    global camera_pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = R.from_quat(camera_wxyz[[1, 2, 3, 0]]).as_matrix()
    camera_pose[:3, 3] = camera_pos
    camera_pose_sapien = convert_camera_pose_z_forward_to_sapien(camera_pose)
    sp_camera.entity.set_pose(sapien.Pose(camera_pose_sapien))
    update_robot_trimesh_camera_and_mask([gui.value for gui in joint_gui_handles])

for joint_name, lower, upper, initial_angle in zip(
    robot_pk.actuated_joint_names, robot_pk.lower_joint_limits_np, robot_pk.upper_joint_limits_np, robot_pk.reference_joint_values_np
):
    lower = float(lower) if lower is not None else -np.pi
    upper = float(upper) if upper is not None else np.pi
    slider = sv.gui.add_slider(
        label=joint_name,
        min=lower,
        max=upper,
        step=0.05,
        initial_value=float(initial_angle),
    )
    slider.on_update(  # When sliders move, we update the URDF configuration.
        lambda _: update_robot_trimesh_camera_and_mask([gui.value for gui in joint_gui_handles])
    )
    joint_gui_handles.append(slider)

camera_control.on_update(
    lambda _: update_camera_pose(camera_control.wxyz, camera_control.position)
)
update_robot_trimesh_camera_and_mask([gui.value for gui in joint_gui_handles])

save_button = sv.gui.add_button("save_camera_pose",)
save_button.on_click(
    lambda _: np.save(save_init_pose_np_path, camera_pose)
)

while True:
    time.sleep(1)
    update_robot_trimesh_camera_and_mask([gui.value for gui in joint_gui_handles])

