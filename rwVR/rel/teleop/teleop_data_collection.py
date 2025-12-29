
import time
import click
import numpy as np 
from scipy.spatial.transform import Rotation as R

from rel import DATA_PATH
from rel.cameras.orbbec import Orbbec
from rel.cameras.realsense import Realsense
from rel.robots.rw_robot import XArm6RealWorld, XArm7RealWorld
from rel.teleop.quest_to_arm import SingleArmQuestAgent
from rel.utils.keyboard_utils import KeyBoardCommand
from rel.utils.teleop_utils import precise_wait
from rel.utils.pc_utils import fpsample_pc
from rel import CAMERA_DATA_PATH
from tqdm import tqdm


if __name__ == "__main__":
    ##########################################################################################
    # Hyperparameters
    ##########################################################################################
    # 0. set your hyperparameters
    serial_number = 'CL8H74100BB'
    robot_type = "xarm7"
    camera_type = "orbbec"
    which_hand = 'r'
    frequency = 10
    command_latency = 0.01
    init_joint_values = [0, -10, 0, 34.8, 0, 44.3, -90]
    exp_dir = DATA_PATH / "0428_9points_zixuan" / "raw_data"
    
    calib_name = "0428_excalib_capture00"
    X_BaseCamera_path = CAMERA_DATA_PATH / serial_number/ calib_name / "manual_X_BaseCamera.npy"
    X_BaseCamera = np.load(X_BaseCamera_path)
    bbox = [[0.4, 0.8], [-0.3, 0.3], [0.03, 2]]  # [[minx, maxx], [miny, maxy], [minz, maxz]]
    n_save_points = 512
    ##########################################################################################
    # Main code
    ##########################################################################################
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # camera initialization
    if camera_type == "orbbec":
        camera = Orbbec(serial_number, use_color=False, use_depth=True)
    elif camera_type == "realsense":
        camera = Realsense(serial_number)
    else:
        raise ValueError

    # robot initialization
    assert robot_type in ['xarm6', 'xarm7'], "robot_type must be either XArm6WOEE or XArm7WOEE"
    if robot_type == 'xarm6':
        rw_arm = XArm6RealWorld()
    elif robot_type == 'xarm7':
        rw_arm = XArm7RealWorld(is_radian=False)

    rw_arm.set_joint_values(init_joint_values, speed=30, is_radian=False, wait=True)
    rw_arm.arm.set_gripper_position(850, wait=True)
    rw_arm.arm.clean_error()
    rw_arm.arm.clean_warn()
    rw_arm.arm.motion_enable(True)
    time.sleep(0.1)
    rw_arm.arm.set_mode(7)
    time.sleep(0.1)
    rw_arm.arm.set_state(state=0)
    time.sleep(0.1)
    rw_arm.arm.set_collision_sensitivity(0)
    time.sleep(0.1)
    rw_arm.arm.set_gripper_enable(True)
    time.sleep(0.1)
    rw_arm.arm.set_gripper_mode(0)
    time.sleep(0.1)
    rw_arm.arm.set_gripper_speed(3000)
    time.sleep(0.1)
    
    # question initialization
    quest = SingleArmQuestAgent(which_hand=which_hand)

    # status initialization
    dt = 1.0 / frequency
    frame_idx = 0
    stop = False  # whether running 
    is_recording = False  # whether recording data
    abort = False  # whether to keep 

    # keyboard initialization
    keyboard = KeyBoardCommand()

    # data storage initialization
    depths = []
    pointclouds = []
    proprioceptions = []
    original_actions = []
    
    t_start = time.monotonic()
    while not stop:
        # calculate timing
        t_cycle_end = t_start + (frame_idx + 1) * dt
        t_obs = t_cycle_end - command_latency
        t1 = time.time()
        # handle key presses
        press_events = keyboard.get_keyboard_to_robot()
        for key_stroke in press_events:
            if key_stroke == 'stop':  # z
                # Exit program
                stop = True
                abort = True
            elif key_stroke == 'record':  # n 
                # Start recording
                is_recording = True
                rw_arm.arm.set_mode(0)
                rw_arm.arm.set_state(state=0)
                time.sleep(0.1)
                rw_arm.set_joint_values(init_joint_values, speed=30, is_radian=False, wait=True)
                rw_arm.arm.set_gripper_position(850, wait=True)
                rw_arm.arm.set_mode(7)
                rw_arm.arm.set_state(state=0)
                rw_arm.arm.set_gripper_mode(0)
                time.sleep(0.1)
                print('Recording!')
            elif key_stroke == 'stop_record':  # m
                # Stop recording
                is_recording = False
                stop = True
                print('Recording Stopped.')
            elif key_stroke == 'drop':
                # Delete the most recent recorded episode
                if click.confirm('Are you sure to drop an episode?'):
                    # TODO
                    is_recording = False
                    depths = []
                    pointclouds = []
                    proprioceptions = []
                    original_actions = []
                    print('Episode dropped! Ready to record new episode.')
        t2 = time.time()
        # get camera observation
        camera_obs = camera.getCurrentData(pointcloud=True,)
        current_depth = camera_obs['depth']  
        current_pointcloud = camera_obs['pointcloud_np']
        t3 = time.time()
        # get robot proprioception
        current_arm_joint_values = rw_arm.get_joint_values(is_radian=True)  # arm joint values
        _, current_gripper_open_dis = rw_arm.arm.get_gripper_position()  # gripper open distance | 0 close, 850 open 
        current_proprioception = np.concatenate([current_arm_joint_values, [current_gripper_open_dis]])
        t4 = time.time()
        # wait till send action command 
        precise_wait(t_obs)
        t5 = time.time()
        # process the teleop data 
        rw_arm_ee_pose = rw_arm.get_current_pose()
        action = quest.act({"X_WorldEE": rw_arm_ee_pose})
        t6 = time.time()
        if action is not None:
            X_WorldEENext, new_gripper_angle = action
        else:
            precise_wait(t_cycle_end)
            frame_idx += 1
            continue
        t7 = time.time()
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
            speed=100,
        )
        assert new_gripper_angle is not None
        gripper = (1 - new_gripper_angle) * 850  #  0 close, 850 open  
        if gripper > 0.5 * 850:
            gripper = 850 
        else:
            gripper = 0
        rw_arm.arm.set_gripper_position(gripper, wait=False)
        original_action = np.concatenate([pos_WorldEENext.reshape(-1), X_WorldEENext[:3, :3].reshape(-1), [gripper]])
        t8 = time.time()
        # store all the data
        if is_recording:
            depths.append(current_depth)
            pointclouds.append(current_pointcloud)
            proprioceptions.append(current_proprioception)
            original_actions.append(original_action)

        # wait till next cycle
        precise_wait(t_cycle_end)
        frame_idx += 1
        t9 = time.time()
        print(f"t2: {t2 - t1:.4f}, t3: {t3 - t2:.4f}, t4: {t4 - t3:.4f}, t5: {t5 - t4:.4f}, t6: {t6 - t5:.4f}, t7: {t7 - t6:.4f}, t8: {t8 - t7:.4f}, t9: {t9 - t8:.4f}")

    if not abort:
        assert len(depths) == len(pointclouds) == len(proprioceptions) == len(original_actions)
        # save the data
        current_idx = 0
        while (exp_dir / f"{current_idx}.npz").exists():
            current_idx += 1

        # post-process point cloud
        sampled_pcs = []
        for i in tqdm(range(len(pointclouds)), desc="Processing frames"):
            frame_pc = pointclouds[i]
            frame_pc = (X_BaseCamera[:3, :3] @ frame_pc.T + X_BaseCamera[:3, 3][:, np.newaxis]).T
            cropped_pc = frame_pc[
                (frame_pc[:, 0] > bbox[0][0]) & (frame_pc[:, 0] < bbox[0][1]) &
                (frame_pc[:, 1] > bbox[1][0]) & (frame_pc[:, 1] < bbox[1][1]) &
                (frame_pc[:, 2] > bbox[2][0]) & (frame_pc[:, 2] < bbox[2][1])
            ]  # (n, 3)
            sampled_pc = fpsample_pc(cropped_pc, n_save_points=n_save_points)
            sampled_pcs.append(sampled_pc)
        sampled_pcs = np.stack(sampled_pcs, axis=0)  # (T, n, 3)
        
        dict_to_save = {
            'pointclouds': np.array(sampled_pcs),
            'proprioceptions': np.array(proprioceptions),
            'original_actions': np.array(original_actions),
        }

        np.savez(exp_dir / f"{current_idx}.npz", **dict_to_save)
    else:
        print('Recording aborted!')
        
    camera.stop()
    exit()
