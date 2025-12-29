import cv2
import time
import viser 
import numpy as np 
import open3d as o3d
from loguru import logger as lgr
from pathlib import Path
from enum import Enum, auto
from scipy.spatial.transform import Rotation as R
from rel import XARM6_IP, XARM7_IP
from rel.robots.pk_robot import XArm6WOEE, XArm7WOEE
from rel.robots.rw_robot import XArm6RealWorld, XArm7RealWorld
from rel.cameras.realsense import Realsense
try:
    from rel.cameras.orbbec import Orbbec
except:
    Orbbec = None
from rel import CAMERA_DATA_PATH, CAMERA_ASSETS_PATH


class VerificationState(Enum):
    WAITING_FOR_VERIFICATION = auto()
    VERIFIED_YES = auto()
    VERIFIED_NO = auto()


def update_viser_current_arm(sv:viser.ViserServer, arm_pk, current_joint_values, camera, init_X_BaseCamera):
    current_arm_mesh = arm_pk.get_state_trimesh(current_joint_values, visual=True, collision=False)["visual"]
    sv.scene.add_mesh_trimesh("current_arm_mesh", current_arm_mesh)

    # add the camera pose
    camera_wxyz = R.from_matrix(init_X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]]
    camera_pos = init_X_BaseCamera[:3, 3]
    rtr_dict = camera.getCurrentData(pointcloud=False)
    rs_rgb = rtr_dict["rgb"]

    sv.scene.add_camera_frustum("rs_camera_img", fov=camera.fov_x, aspect=camera.aspect_ratio, wxyz=camera_wxyz, position=camera_pos, image=rs_rgb, scale=0.2)


if __name__ == '__main__':
    ##########################################################################################
    # Hyperparameters
    ##########################################################################################
    # 0. set your hyperparameters
    serial_number = "233622079809"
    exp_name = "0509_excalib_capture00"
    robot_type = "xarm6"
    camera_type = "realsense"
    
    ## Camera
    # 1. setup camera
    if camera_type == "orbbec":
        camera = Orbbec(serial_number, use_color=True, use_depth=False)
    elif camera_type == "realsense":
        camera = Realsense(serial_number)
    else:
        raise ValueError
    
    # 2. load camera extrinsic params
    camera_data_path = CAMERA_DATA_PATH / serial_number
    init_X_BaseCamera_path = camera_data_path / exp_name / "init_X_BaseCamera.npy"
    init_X_BaseCamera = np.load(init_X_BaseCamera_path)
    init_X_CameraBase = np.linalg.inv(init_X_BaseCamera)
    lgr.info("init_X_BaseCamera: \n{}".format(init_X_BaseCamera))
    
    # 4. get camera infos
    H, W = camera.h, camera.w
    
    ## Robot
    # 4. setup the real world 
    if robot_type == "xarm6":
        xarm_rw = XArm6RealWorld(ip=XARM6_IP) # XARM6_IP
        # setup the digital twin xarm for vis 
        xarm_pk = XArm6WOEE()
    elif robot_type == "xarm7":
        xarm_rw = XArm7RealWorld(ip=XARM7_IP) # XARM7_IP
        # setup the digital twin xarm for vis  
        xarm_pk = XArm7WOEE()
    
    # 5. set the output path, set the exp_name to calibrate your own camera
    
    save_data_rel_dir_path = camera_data_path / exp_name
    
    # 6. setup the gui server
    sv = viser.ViserServer()
    button_verify_yes = sv.gui.add_button("verify_yes")
    buttion_exit = sv.gui.add_button("exit")

    ''' setup the gui server '''


    def set_verification_state(state):
        global verification_state
        verification_state = state
        lgr.info(f"Verification state: {verification_state}")

    def on_exit():
        xarm_rw.close()
        exit()


    n_collected_sample = 0

    def save_current_data():
        global n_collected_sample
        rt_dict = camera.getCurrentData(pointcloud=False)
        rgb_image = rt_dict["rgb"]

        # save the data
        sample_dir_path = save_data_rel_dir_path / f"{n_collected_sample:04d}"
        sample_dir_path.mkdir(parents=True, exist_ok=True)
        np.save(sample_dir_path / "joint_values.npy", xarm_rw.get_joint_values())
        cv2.imwrite(str(sample_dir_path / "rgb_image.jpg"), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        print(f"Saved data to: {sample_dir_path}")
        
        n_collected_sample += 1


    button_verify_yes.on_click(lambda _: save_current_data())
    buttion_exit.on_click(lambda _: on_exit())


    while True: 
        current_joint_values = xarm_rw.get_joint_values()
        update_viser_current_arm(sv, xarm_pk, current_joint_values, camera, init_X_BaseCamera)
        time.sleep(0.01)

