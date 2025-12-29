from rel.robots.rw_robot import XArm7RealWorld
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path('/home/world4omni/w4o')       # should be ~/w4o
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
sys.path.insert(0, str(PROJECT_ROOT))
from World4Omni_rw.tools.get_new import get_newest


def open_gripper(xarm):
    xarm.arm.set_gripper_position(850, wait=True)


def close_gripper(xarm):
    xarm.arm.set_gripper_position(0, wait=True)

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
    open_gripper(xarm)


def move_to(xarm, pose):        # pose in mm
    xarm.arm.set_position(
        pose[0], 
        pose[1], 
        pose[2], 
        pose[3],
        pose[4],
        pose[5],
        wait=True,
        is_radian=False,
        speed=100,
    )

def main():
    basename = get_newest(RAW_DATA_DIR)

    grasp_path = f"{RAW_DATA_DIR}/{basename}/grasp_pose_in_base.npz"
    goal_path = f"{RAW_DATA_DIR}/{basename}/goal_pose_in_base.npz"

    grasp_in_base = np.load(grasp_path)['grasp_pose']  # (xyz, euler(xyz))
    grasp_in_base[:3] = grasp_in_base[:3] * 1000  # to mm

    goal_in_base = np.load(goal_path)['goal_pose']  # (xyz, euler(xyz))
    goal_in_base[:3] = goal_in_base[:3] * 1000  # to mm

    xarm = XArm7RealWorld()
    arm_init(xarm)

    pre_grasp_in_base = grasp_in_base.copy()
    pre_grasp_in_base[2] += 20  # above
    move_to(xarm, pre_grasp_in_base)

    move_to(xarm, grasp_in_base)

    close_gripper(xarm)

    pre_goal_in_base = goal_in_base.copy()
    move_to(xarm, pre_goal_in_base)

    move_to(xarm, goal_in_base)

    open_gripper(xarm)

if __name__ == "__main__":
    main()