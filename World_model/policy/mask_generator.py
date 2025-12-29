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


for cur_task in [TakePlateOffColoredDishRack]:
# for cur_task in [OpenWineBottle]:
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.launch()
    task = env.get_task(cur_task)
    # breakpoint()
    print('Reset Episode')
    descriptions, obs = task.reset()

    from grounded_sam import grounded_segmentation, plot_detections
    depth = obs.front_depth
    
    IMAGE_PATH = './feature_based_icp/pcd_data/02_plate_rgb.png'
    image = Image.open(IMAGE_PATH)
    # image = Image.fromarray(image)
    threshold = 0.3
    
    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"
    prompt = ['The plate on the dish rack.']
    image_array, detections = grounded_segmentation(
        image=image,
        labels=prompt,
        threshold=threshold,
        polygon_refinement=True,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )
    
        
    mask = detections[0].mask.astype(np.bool)  # mask of the first object
    Image.fromarray((mask * 255).astype(np.uint8)).save(f"tmp/{task.get_name()}_target_front_mask.png")
    print(f"Mask saved to {task.get_name()}_front_mask.png")
    env.shutdown()