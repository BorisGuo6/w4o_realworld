
import time
import click
import numpy as np 
from scipy.spatial.transform import Rotation as R

from rel import CAMERA_ASSETS_PATH
from rel.cameras.orbbec import Orbbec
from rel.cameras.realsense import Realsense
from rel.robots.pk_robot import XArm6WOEE, XArm7WOEE
from rel.robots.rw_robot import XArm6RealWorld, XArm7RealWorld
from rel.teleop.quest_to_arm import SingleArmQuestAgent
from rel.utils.keyboard_utils import KeyBoardCommand
from rel.utils.teleop_utils import precise_wait
from rel import CAMERA_DATA_PATH

if __name__ == "__main__":
    ##########################################################################################
    # Hyperparameters
    ##########################################################################################
    # 0. set your hyperparameters
    serial_number = 'CL8H74100BB'
    exp_name = "0426_excalib_capture00"
    robot_type = "xarm7"
    camera_type = "orbbec"
    which_hand = 'r'
    frequency = 20 
    command_latency = 0.01
    
    X_BaseCamera_path = CAMERA_DATA_PATH / serial_number/ exp_name / "X_BaseCamera.npy"
    ##########################################################################################
    # Main code
    ##########################################################################################
    # camera initialization
    if camera_type == "orbbec":
        camera = Orbbec(serial_number, use_color=False, use_depth=True)
    elif camera_type == "realsense":
        camera = Realsense(serial_number)
    else:
        raise ValueError
    X_BaseCamera = np.load(X_BaseCamera_path)

    # (Optional) set the camera intrinsics, you can just use the default one
    # camera.set_intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    # robot initialization
    assert robot_type in ['XArm6WOEE', 'XArm7WOEE'], "robot_type must be either XArm6WOEE or XArm7WOEE"
    if robot_type == 'XArm6WOEE':
        pk_arm = XArm6WOEE()
        rw_arm = XArm6RealWorld()
    elif robot_type == 'XArm7WOEE':
        pk_arm = XArm7WOEE()
        rw_arm = XArm7RealWorld()

    # question initialization
    quest = SingleArmQuestAgent(which_hand=which_hand)

    # status initialization
    dt = 1.0 / frequency
    frame_idx = 0
    stop = False
    is_recording = False
    abort = False

    # keyboard initialization
    keyboard = KeyBoardCommand()

    # data storage initialization
    depths = []
    pointclouds = []
    proprioceptions = []
    joint_name_list = pk_arm.actuated_joint_names.copy()
    link_name_list = sorted(list(pk_arm.link_visuals_dict.keys()).copy())
    
    t_start = time.monotonic()
    while not stop:
        # calculate timing
        t_cycle_end = t_start + (frame_idx + 1) * dt
        t_obs = t_cycle_end - command_latency

        # handle key presses
        press_events = keyboard.get_keyboard_to_robot()
        for key_stroke in press_events:
            if key_stroke == 'stop':
                # Exit program
                stop = True
                abort = True
            elif key_stroke == 'record':
                # Start recording
                is_recording = True
                print('Recording!')
            elif key_stroke == 'stop_record':
                # Stop recording
                is_recording = False
                stop = True
                print('Recording Stopped.')
            elif key_stroke == 'drop':
                # Delete the most recent recorded episode
                if click.confirm('Are you sure to drop an episode?'):
                    # TODO
                    is_recording = False
                    rw_arm.close()

        # get camera observation
        camera_obs = camera.getCurrentData(pointcloud=True,)
        current_depth = camera_obs['depth']  
        current_pointcloud = camera_obs['pointcloud_np']

        # get robot proprioception
        current_arm_joint_values = rw_arm.get_joint_values()  # arm joint values
        _, current_gripper_open_dis = rw_arm.arm.get_gripper_position()  # gripper open distance | 0 close, 850 open 
        current_proprioception = np.concatenate([current_arm_joint_values, [current_gripper_open_dis]])
        current_arm_link_status = pk_arm.pk_chain.forward_kinematics(
            th=pk_arm.ensure_tensor(current_arm_joint_values)
        )
        current_arm_link_poses = []
        for link_name in link_name_list:
            current_link_pose_np = current_arm_link_status[link_name].detach().cpu().numpy()[0]  # 4, 4
            current_arm_link_poses.append(current_link_pose_np)
        current_arm_link_poses = np.stack(current_arm_link_poses, axis=0)  # n, 4, 4

        # wait till send action command 
        precise_wait(t_obs)

        # process the teleop data 
        rw_arm_ee_pose = rw_arm.get_current_pose()
        action = quest.act({"X_WorldEE": rw_arm_ee_pose})
        if action is not None:
            X_WorldEENext, new_gripper_angle = action
        else:
            precise_wait(t_cycle_end)
            frame_idx += 1
            continue

        # send action command 
        pos_WorldEENext = X_WorldEENext[:3, 3]
        euler_WorldEENext = R.from_matrix(X_WorldEENext[:3, :3]).as_euler('xyz', degrees=True)
        rw_arm.arm.set_position(
            x=pos_WorldEENext[0] * 1000,
            y=pos_WorldEENext[1] * 1000,
            z=pos_WorldEENext[2] * 1000,
            roll=euler_WorldEENext[0],
            pitch=euler_WorldEENext[1],
            yaw=euler_WorldEENext[2],
            wait=False,
            speed=200,
        )
        assert new_gripper_angle is not None
        gripper = (1 - new_gripper_angle) * 850
        rw_arm.arm.set_gripper_position(gripper, wait=False)
        original_action = np.concatenate([pos_WorldEENext.reshape(-1), [gripper]])

        # store all the data
        if is_recording:
            depths.append(current_depth)
            pointclouds.append(current_pointcloud)
            proprioceptions.append(current_proprioception)


        # wait till next cycle
        precise_wait(t_cycle_end)
        frame_idx += 1
