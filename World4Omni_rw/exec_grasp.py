# from rel.cameras.orbbec import Orbbec
# from model import GroundedGraspNet
from rel.robots.rw_robot import XArm7RealWorld
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle

import viser
import sys
from pathlib import Path

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

        server = viser.ViserServer()

        server.scene.add_frame(
            "cam_pose",
            wxyz=R_base_camera.as_quat()[[3, 0, 1, 2]],
            position=translation_base_camera,
        )

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
        input()

        # close server
        server.stop()
        
        # # convert grasp to end effector pose, [x,y,z,qx,qy,qz,qw]
        # rotation = torch.tensor(rotation)
        # quat = transforms.matrix_to_quaternion(rotation).numpy() # w,x,y,z
        # #  convert quat to x,y,z,w 
        # quat = np.roll(quat, -1)
        # ee_pose = np.concatenate([translation, quat], axis=0)
        
        return ee_pose, ee_pose_in, ee_pose_out
    

def main():
    basename = get_newest(RAW_DATA_DIR)
    grasp_path = f"{RAW_DATA_DIR}/{basename}/grasps_in_cam.npz"
    ## camera pose in base camera frame

    xarm = XArm7RealWorld()
    arm_init(xarm)

    best_grasp = np.load(grasp_path)['grasps'][2]  # (4,4) numpy array
    grasp_in_base_frame, ee_pose_in, ee_pose_out = postprocess_grasp(best_grasp)        # in and out is offset by gripper depth
    print(f"grasp_in_base_frame = {grasp_in_base_frame}")

    save_grasp_path = f"{RAW_DATA_DIR}/{basename}/grasp_pose_in_base.npz"       # (xyz, euler)
    np.savez(save_grasp_path, grasp_pose=grasp_in_base_frame)
    print(f"Saved grasp in base frame to {save_grasp_path}")

    input()
    xarm.arm.set_position(
        grasp_in_base_frame[0] * 1000, 
        grasp_in_base_frame[1] * 1000, 
        grasp_in_base_frame[2] * 1000, 
        grasp_in_base_frame[3],
        grasp_in_base_frame[4],
        grasp_in_base_frame[5],
        wait=True,
        is_radian=False,
        speed=100,
    )

if __name__ == "__main__":
    main()