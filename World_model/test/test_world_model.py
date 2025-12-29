#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the modified World_model.py
Accepts a folder path, processes rgb_init.png, and saves the output as rgb_goal.png.
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.World_model import World_model

PROJECT_ROOT = Path('/home/world4omni/w4o') # should be ~/w4o
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
sys.path.insert(0, str(PROJECT_ROOT))
from World4Omni_rw.tools.get_new import get_newest

def test_world_model(img_dir_path: str):
    """
    Test the World_model function by processing an image from a given directory.
    
    Args:
        img_dir_path (str): The path to the directory containing 'rgb_init.png'.
    """
    
    # --- 1. Set up paths based on the input directory ---
    image_dir = Path(img_dir_path)
    image_path = image_dir / "rgb_init.png"
    dest_file_path = image_dir / "rgb_goal.png"
    
    # Check if the source image exists
    if not image_path.exists():
        print(f"❌ Error: Input image not found at '{image_path}'")
        sys.exit(1)
        
    # text_instruction = "move the tomato to pan"
    # text_instruction = "flip open the box"
    # text_instruction = "Place the pink cup horizontally above the bowl"
    # text_instruction = "move the Rubik's cube onto the orange cube"
    text_instruction = "place the duck onto the electronic scale"
    
    # Set GEMINI_API_KEY if not already set
    if not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = "AIzaSyCqX_zpTdFzrNt2IqejA1FWwa6_jxX11do"
    
    print("🧪 Testing World Model with mask generation...")
    print(f"📁 Source Image: {image_path}")
    print(f"📝 Instruction: {text_instruction}")
    print()
    
    # Start timing
    start_time = time.time()
    timing_info = {}
    
    try:
        # --- 2. Run the World Model ---
        print("⏱️  Starting World Model processing...")
        world_model_start = time.time()
        
        result_path = World_model(
            image=image_path,
            text=text_instruction,
            generate_masks=True,
            cleanup_intermediate=True,
            box_threshold=0.35,
            text_threshold=0.25,
            max_iterations=3,
            use_enhancer=True,
            use_reflector=True
        )
        
        world_model_end = time.time()
        timing_info["World Model Total"] = world_model_end - world_model_start
        
        print(f"\n✅ World Model processing completed successfully!")
        print(f"↪️  Temporary result file: {result_path}")

        # --- 3. Copy the result to the final destination ---
        shutil.copy(result_path, dest_file_path)
        print(f"✅ Final image saved to: {dest_file_path}")
        
        # Check if mask outputs were created
        mask_dir = Path("outputs/mask")
        if mask_dir.exists() and any(mask_dir.iterdir()):
            mask_files = list(mask_dir.glob("*"))
            print(f"🎭 Mask files created: {len(mask_files)}")
            for file in mask_files:
                print(f"  - {file.name}")
        else:
            print("⚠️ No mask directory or mask files found.")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    
    # --- 4. Print timing information ---
    total_time = time.time() - start_time
    timing_info["Total Test Time"] = total_time
    
    print(f"\n⏱️  Timing Information:")
    print(f"{'='*50}")
    for process, duration in timing_info.items():
        minutes = int(duration // 60)
        seconds = duration % 60
        if minutes > 0:
            print(f"  {process}: {minutes}m {seconds:.2f}s")
        else:
            print(f"  {process}: {seconds:.2f}s")
    print(f"{'='*50}")

if __name__ == "__main__":
    # --- Set up argument parser to accept --img-path ---
    parser = argparse.ArgumentParser(
        description="Run the World Model on rgb_init.png inside a specified folder."
    )
    parser.add_argument(
        "--img-path", 
        type=str, 
        default=None,
        help=(
            "包含 'rgb_init.png' 文件的目录路径。\n"
            f"如果未提供此参数，程序将自动从以下目录中寻找最新的数据文件夹：\n{RAW_DATA_DIR}"
        )
    )
    args = parser.parse_args()

    target_path = args.img_path
    
    if target_path:
        print(f"Using User provided: {target_path}")
    else:
        print(f"Default get newest in: {RAW_DATA_DIR}")
        basename = get_newest(RAW_DATA_DIR)
        
        if basename:
            target_path = RAW_DATA_DIR / basename
            print(f"Using: {target_path}")
        else:
            assert 0 

    test_world_model(target_path)