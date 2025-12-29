import cv2 
import numpy as np
from pathlib import Path
from rel import SAM_TYPE, SAM_PATH, CAMERA_DATA_PATH
from rel.utils.sam_prompt_drawer import SAMPromptDrawer


if __name__ == "__main__":
    ##########################################################################################
    # Hyperparameters
    ##########################################################################################
    # 0. set your hyperparameters
    serial_number = "317222073552" # 241122074374 CL8H74100BB
    exp_name = "0509_excalib_capture00"
    
    # 1. load camera extrinsic sample paths
    save_data_rel_dir_path = CAMERA_DATA_PATH / serial_number / exp_name
    sample_id_paths = list(save_data_rel_dir_path.glob("*"))
    sample_id_paths = sorted(sample_id_paths)
    sample_id_paths = [p for p in sample_id_paths if p.is_dir()]

    sample_img_paths = [p / "rgb_image.jpg" for p in sample_id_paths]

    # 2. setup the prompt drawer
    prompt_drawer = SAMPromptDrawer(window_name="Prompt Drawer", screen_scale=2.0, sam_checkpoint=SAM_PATH, device="cuda", model_type=SAM_TYPE)
    
    # 3. Iterate through each image and obtain the mask
    for img_path in sample_img_paths:
        print(f"Processing image: {img_path}")
        if img_path.exists():
            # Load the RGB image
            print(f"111")
            rgb_image = cv2.imread(str(img_path))

            # Convert the image from BGR (OpenCV default) to RGB for SAMPromptDrawer
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            print(f"222")

            # Run the prompt drawer to obtain the mask
            prompt_drawer.reset()
            print(f"333")
            
            mask = prompt_drawer.run(rgb_image)
            print(f"444")

            # Save the mask as a numpy array
            if mask is not None:
                mask_path = img_path.parent / "mask.npy"
                np.save(str(mask_path), mask)
                print(f"Mask saved to: {mask_path}")
            else:
                print(f"No mask generated for: {img_path}")
        else:
            print(f"Image not found: {img_path}")

