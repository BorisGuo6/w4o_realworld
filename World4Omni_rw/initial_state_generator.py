import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import TakeFrameOffHanger, OpenDoor, PutBooksOnBookshelf, ReachTarget, CloseBox, PutShoesInBox, PickAndLift, PickUpCup, OpenWineBottle, PushButton, TakePlateOffColoredDishRack, RemoveCups, InsertOntoSquarePeg, PlaceShapeInShapeSorter, PlugChargerInPowerSupply, PutKnifeInKnifeBlock, SlideCabinetOpenAndPlaceCups, StraightenRope

import os
import sys
from typing import Dict, Tuple

import torch
from pytorch3d import transforms
from model import AnyGrasp
import imageio
from PIL import Image
import open3d as o3d

# Fix the random seed for reproducibility for both numpy and torch
import random
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


obs_config = ObservationConfig()
obs_config.set_all(True)
env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()),
    obs_config=obs_config,
    headless=False)


for cur_task in [PutBooksOnBookshelf, OpenDoor, TakeFrameOffHanger, CloseBox, PutShoesInBox, OpenWineBottle, TakePlateOffColoredDishRack, RemoveCups, InsertOntoSquarePeg, PlaceShapeInShapeSorter, PlugChargerInPowerSupply, PutKnifeInKnifeBlock, SlideCabinetOpenAndPlaceCups, StraightenRope]:
    env.launch()
    task = env.get_task(cur_task)
    # breakpoint()
    print('Reset Episode')
    descriptions, obs = task.reset()
    Image.fromarray(obs.front_rgb).save(os.path.join('tmp', 'initial_state', f'{task.get_name()}_front.png'))
    Image.fromarray(obs.overhead_rgb).save(os.path.join('tmp', 'initial_state', f'{task.get_name()}_overhead.png'))
    Image.fromarray(obs.left_shoulder_rgb).save(os.path.join('tmp', 'initial_state', f'{task.get_name()}_left_shoulder.png'))
    Image.fromarray(obs.right_shoulder_rgb).save(os.path.join('tmp', 'initial_state', f'{task.get_name()}_right_shoulder.png'))
    Image.fromarray(obs.wrist_rgb).save(os.path.join('tmp', 'initial_state', f'{task.get_name()}_wrist.png'))
    print(f"Initial images of {task.get_name()} are saved.")
    env.shutdown()