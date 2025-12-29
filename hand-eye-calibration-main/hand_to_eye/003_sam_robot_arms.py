import cv2 
import numpy as np
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from utils.sam_prompt_drawer import SAMPromptDrawer

from config.config import HandToEyeCalibConfig


serial_number = HandToEyeCalibConfig.serial_number
exp_name = HandToEyeCalibConfig.exp_name
save_data_path = (HandToEyeCalibConfig.save_data_path / HandToEyeCalibConfig.exp_name).resolve()
sample_id_paths = sorted(save_data_path.glob("*/")) 

sample_img_paths = []
for p in sample_id_paths:
    if p.is_dir() and (p / "rgb_image.jpg").exists():
        sample_img_paths.append(p / "rgb_image.jpg")

# setup the prompt drawer
prompt_drawer = SAMPromptDrawer(window_name="Prompt Drawer", screen_scale=2.0, sam_checkpoint=HandToEyeCalibConfig.sam_path, device="cuda", model_type=HandToEyeCalibConfig.sam_type)

# Iterate through each image and obtain the mask
for img_path in sample_img_paths:
    print(f"Processing image: {img_path}")
    if img_path.exists():
        # Load the RGB image
        rgb_image = cv2.imread(img_path.as_posix())

        # Convert the image from BGR (OpenCV default) to RGB for SAMPromptDrawer
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Run the prompt drawer to obtain the mask
        prompt_drawer.reset()
        mask = prompt_drawer.run(rgb_image)

        # Save the mask as a numpy array
        if mask is not None:
            mask_path = img_path.parent / "mask.npy"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(mask_path.as_posix(), mask)
            print(f"Mask saved to: {mask_path}")
        else:
            print(f"No mask generated for: {img_path}")
    else:
        print(f"Image not found: {img_path}")