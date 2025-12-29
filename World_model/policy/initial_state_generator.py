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
import imageio
from PIL import Image
import open3d as o3d
import pickle
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


for cur_task in [TakeFrameOffHanger, OpenWineBottle, TakePlateOffColoredDishRack, RemoveCups]:
# for cur_task in [OpenWineBottle]:
    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.launch()
    task = env.get_task(cur_task)
    # breakpoint()
    print('Reset Episode')
    descriptions, obs = task.reset()
    Image.fromarray(obs.front_rgb).save(os.path.join('tmp', 'initial_state_5seeds', f'{task.get_name()}_front_seed{seed}.png'))
    Image.fromarray(obs.overhead_rgb).save(os.path.join('tmp', 'initial_state_5seeds', f'{task.get_name()}_overhead_seed{seed}.png'))
    Image.fromarray(obs.left_shoulder_rgb).save(os.path.join('tmp', 'initial_state_5seeds', f'{task.get_name()}_left_shoulder_seed{seed}.png'))
    Image.fromarray(obs.right_shoulder_rgb).save(os.path.join('tmp', 'initial_state_5seeds', f'{task.get_name()}_right_shoulder_seed{seed}.png'))
    Image.fromarray(obs.wrist_rgb).save(os.path.join('tmp', 'initial_state_5seeds', f'{task.get_name()}_wrist_seed{seed}.png'))
    print(f"Initial images of {task.get_name()} are saved.")
    
    # Save obs.front_rgb, obs.front_depth, obs.front_point_cloud to tmp/initial_state/{task.get_name()}_front_dict.pkl
    observation = {
        'front_rgb': obs.front_rgb,
        'front_depth': obs.front_depth,
        'front_point_cloud': obs.front_point_cloud
    }
    # Save the observation dictionary to a file
    with open(os.path.join('tmp', 'initial_state_5seeds', f'{task.get_name()}_front_dict_seed{seed}.pkl'), 'wb') as f:
        pickle.dump(observation, f)
    
    env.shutdown()