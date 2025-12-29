from typing import Dict
import numpy as np
from oculus_reader.reader import OculusReader
import viser 
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI
import time

trigger_state = {"l": False, "r": False}
X_WorldQuest = np.eye(4)
X_WorldQuest[:3, :3] = R.from_euler('X', [-90], degrees=True).as_matrix()
X_QuestWorld = np.linalg.inv(X_WorldQuest)


class SingleArmQuestAgent:
    
    def __init__(
        self,
        which_hand: str,
        translation_scaling_factor: float = 1.0,
    ) -> None:
        """Interact with the robot using the quest controller.

        leftTrig: press to start control (also record the current position as the home position)
        leftJS: a tuple of (x,y) for the joystick, only need y to control the gripper
        """
        self.which_hand = which_hand
        assert self.which_hand in ["l", "r"]

        self.oculus_reader = OculusReader()
        self.control_active = False
    
        self.X_WorldEERef = None 
        self.X_QuestHandleRef = None 
        self.translation_scaling_factor = translation_scaling_factor


    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Input:
            - obs 
                - X_WorldEE , np 4x4 
        
        """

        if self.which_hand == "l":
            pose_key = "l"
            trigger_key = "leftTrig"
            grip_key = "leftGrip"
            joystick_key = "leftJS"
        elif self.which_hand == "r":
            pose_key = "r"
            trigger_key = "rightTrig"
            grip_key = "rightGrip"
            joystick_key = "rightJS"
        else:
            raise ValueError(f"Unknown hand: {self.which_hand}")
        
        # check the trigger button state
        (
            pose_data,
            button_data,
        ) = self.oculus_reader.get_transformations_and_buttons()
        if len(pose_data) == 0 or len(button_data) == 0:
            print("no data, quest not yet ready")
            return None

        new_gripper_angle = [button_data[grip_key][0]]
        print("Gripper angle:", new_gripper_angle)
        X_WorldEECurr = obs["X_WorldEE"]
        X_QuestHandleCurr = pose_data[pose_key]

        global trigger_state
        
        trigger_state[self.which_hand] = button_data[trigger_key][0] > 0.5
        if trigger_state[self.which_hand]:
            is_first_frame_resum_control = not(self.control_active is True)
            if is_first_frame_resum_control:  
                # if the first frame to resume control
                self.control_active = True
                self.X_QuestHandleRef = X_QuestHandleCurr.copy()
                self.X_WorldEERef = X_WorldEECurr.copy()
                return None
            else:
                # if no the first frame to resume control
                dpos_in_Quest = X_QuestHandleCurr[:3, 3] - self.X_QuestHandleRef[:3, 3]
                drot_in_Quest = X_QuestHandleCurr[:3, :3] @ np.linalg.inv(self.X_QuestHandleRef[:3, :3])
                dpos_in_World = X_QuestWorld[:3, :3] @ dpos_in_Quest + X_QuestWorld[:3, 3]
                drot_in_World = X_QuestWorld[:3, :3] @ drot_in_Quest @ X_WorldQuest[:3, :3]

                X_WorldEENext = np.eye(4)
                X_WorldEENext[:3, 3] = dpos_in_World * self.translation_scaling_factor + self.X_WorldEERef[:3, 3]
                print(f'dpos_in_World:, {dpos_in_World}')
                print(f'X_WorldEENext[:3, 3]:, {X_WorldEENext[:3, 3]}')
                X_WorldEENext[:3, :3] = drot_in_World @ self.X_WorldEERef[:3, :3]
                return X_WorldEENext, new_gripper_angle[0]

        else:
            self.control_active = False
            self.X_QuestHandleRef = None
            self.X_WorldEERef = None 
            return None


class XArmQuestAgent:
    def __init__(self, ip="192.168.1.239"):
        
        self.arm = XArmAPI(ip)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        time.sleep(0.1)
        self.arm.set_mode(7)
        time.sleep(0.1)
        self.arm.set_state(state=0)
        time.sleep(0.1)
        self.arm.set_collision_sensitivity(0)
        time.sleep(0.1)
        self.arm.set_gripper_enable(True)
        time.sleep(0.1)
        self.arm.set_gripper_mode(0)
        time.sleep(0.1)
        self.arm.set_gripper_speed(3000)
        time.sleep(0.1)
        
        
    def get_current_pose(self):
        X_WorldEE = np.eye(4)
        _, current_pose = self.arm.get_position()
        X_WorldEE[:3, 3] = np.array(current_pose[:3]) / 1000
        current_euler = current_pose[3:6]
        X_WorldEE[:3, :3] = R.from_euler('xyz', current_euler, degrees=True).as_matrix()
        return X_WorldEE
    

if __name__ == "__main__":
    server = viser.ViserServer()
    
    agent = SingleArmQuestAgent(which_hand="r")
    arm = XArmQuestAgent()
    
    while True:
        time.sleep(0.04)
        
        default_X_WorldEE = arm.get_current_pose()
        pos_WorldEE = default_X_WorldEE[:3, 3]
        xyzw_WorldEE = R.from_matrix(default_X_WorldEE[:3, :3]).as_quat() 
        wxyz_WorldEE = xyzw_WorldEE[[3, 0, 1, 2]]
        server.scene.add_frame(
            position=pos_WorldEE,
            wxyz=wxyz_WorldEE,
            name="EECurr",
        )
        
        # X_WorldEENext, new_gripper_angle = agent.act({"X_WorldEE": default_X_WorldEE})
        action = agent.act({"X_WorldEE": default_X_WorldEE})
        # Extract the action
        if action is not None:
            X_WorldEENext, new_gripper_angle = action
        else:
            X_WorldEENext = None
            new_gripper_angle = None
            
        if X_WorldEENext is not None:
            pos_WorldEENext = X_WorldEENext[:3, 3]
            xyzw_WorldEENext = R.from_matrix(X_WorldEENext[:3, :3]).as_quat() 
            wxyz_WorldEENext = xyzw_WorldEENext[[3, 0, 1, 2]]
            server.scene.add_frame(
                position=pos_WorldEENext,
                wxyz=wxyz_WorldEENext,
                name="Oculus",
            )
            print(X_WorldEENext)
            
            # xarm control
            # pos_WorldEENext = pos_WorldEENext
            euler_WorldEENext = R.from_matrix(X_WorldEENext[:3, :3]).as_euler('xyz', degrees=True)
            arm.arm.set_position(
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
            arm.arm.set_gripper_position(gripper, wait=False)
        else:
            print("No action taken")