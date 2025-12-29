#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for different World_model.py modes
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.World_model import World_model

def test_world_model_modes():
    """Test the World_model function with different configurations."""
    
    # Set up paths
    image_path = Path("images/move_tomato_to_pan.png")
    text_instruction = "move the tomato to pan"
    
    # Set GEMINI_API_KEY if not already set
    if not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = "AIzaSyAIPxQUTVPKHG1A6U9QCviPiMvF8wz4lHY"
    
    print("🧪 Testing World Model with different modes...")
    print(f"📁 Image: {image_path}")
    print(f"📝 Instruction: {text_instruction}")
    print()
    
    # Start timing
    start_time = time.time()
    timing_info = {}
    
    # Test 1: Full mode (enhancer + reflector + masks)
    print("=" * 60)
    print("🔧 Test 1: Full mode (enhancer + reflector + masks)")
    print("=" * 60)
    try:
        test1_start = time.time()
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
        test1_end = time.time()
        timing_info["Test 1: Full mode"] = test1_end - test1_start
        print(f"✅ Full mode completed: {result_path}")
    except Exception as e:
        print(f"❌ Full mode failed: {e}")
    
    print("\n" + "=" * 60)
    print("🔧 Test 2: Simple mode (no enhancer, no reflector, no masks)")
    print("=" * 60)
    try:
        test2_start = time.time()
        result_path = World_model(
            image=image_path,
            text=text_instruction,
            generate_masks=False,
            cleanup_intermediate=True,
            use_enhancer=False,
            use_reflector=False
        )
        test2_end = time.time()
        timing_info["Test 2: Simple mode"] = test2_end - test2_start
        print(f"✅ Simple mode completed: {result_path}")
    except Exception as e:
        print(f"❌ Simple mode failed: {e}")
    
    print("\n" + "=" * 60)
    print("🔧 Test 3: Enhanced only (enhancer + no reflector + no masks)")
    print("=" * 60)
    try:
        test3_start = time.time()
        result_path = World_model(
            image=image_path,
            text=text_instruction,
            generate_masks=False,
            cleanup_intermediate=True,
            use_enhancer=True,
            use_reflector=False
        )
        test3_end = time.time()
        timing_info["Test 3: Enhanced only"] = test3_end - test3_start
        print(f"✅ Enhanced only mode completed: {result_path}")
    except Exception as e:
        print(f"❌ Enhanced only mode failed: {e}")
    
    print("\n" + "=" * 60)
    print("🔧 Test 4: Reflector only (no enhancer + reflector + no masks)")
    print("=" * 60)
    try:
        test4_start = time.time()
        result_path = World_model(
            image=image_path,
            text=text_instruction,
            generate_masks=False,
            cleanup_intermediate=True,
            use_enhancer=False,
            use_reflector=True
        )
        test4_end = time.time()
        timing_info["Test 4: Reflector only"] = test4_end - test4_start
        print(f"✅ Reflector only mode completed: {result_path}")
    except Exception as e:
        print(f"❌ Reflector only mode failed: {e}")
    
    # Calculate total time
    total_time = time.time() - start_time
    timing_info["Total Test Time"] = total_time
    
    # Print timing information
    print(f"\n⏱️  Timing Information:")
    print(f"{'='*60}")
    for process, duration in timing_info.items():
        minutes = int(duration // 60)
        seconds = duration % 60
        if minutes > 0:
            print(f"  {process}: {minutes}m {seconds:.2f}s")
        else:
            print(f"  {process}: {seconds:.2f}s")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_world_model_modes()
