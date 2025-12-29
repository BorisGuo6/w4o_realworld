#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import shutil
import os
import time
from pathlib import Path
from typing import Optional, List, Tuple
import json
from datetime import datetime

# Reuse existing Reflector logic
from scripts.Reflector import (
    run_enhancer,
    run_synthesis_without_enhance,
    validate_overlay_with_gemini,
    generate_revised_enhanced_text,
)

# Import extract_objects function
from scripts.extract_objects import extract_objects


def run_grounded_sam(image_path: Path, objects: List[str], output_dir: Path, 
                    box_threshold: float = 0.35, text_threshold: float = 0.25) -> Tuple[Path, Path]:
    """
    Run Grounded SAM on an image to generate masks for specified objects.
    
    Args:
        image_path: Path to the input image
        objects: List of object names to detect
        output_dir: Directory to save outputs
        box_threshold: GroundingDINO box threshold
        text_threshold: GroundingDINO text threshold
        
    Returns:
        Tuple of (mask_file_path, segmentation_file_path)
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert objects list to comma-separated string
    objects_str = ", ".join(objects)
    
    # Run test_grounded_sam.py
    cmd = [
        "python", 
        str(Path(__file__).parent.parent / "scripts" / "test_grounded_sam.py"),
        str(image_path),
        objects_str,
        "--output-dir", str(output_dir),
        "--box-threshold", str(box_threshold),
        "--text-threshold", str(text_threshold)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Grounded SAM completed for {image_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Grounded SAM failed: {e}")
        print(f"Error output: {e.stderr}")
        raise
    
    # Find the generated mask and segmentation files
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    mask_file = output_dir / f"segmentation_mask_{timestamp}.png"
    segmentation_file = output_dir / f"segmentation_{timestamp}.jpg"
    
    # If timestamp-based files don't exist, look for any mask/segmentation files
    if not mask_file.exists():
        mask_files = list(output_dir.glob("segmentation_mask_*.png"))
        if mask_files:
            mask_file = mask_files[-1]  # Get the most recent one
    
    if not segmentation_file.exists():
        seg_files = list(output_dir.glob("segmentation_*.jpg"))
        if seg_files:
            segmentation_file = seg_files[-1]  # Get the most recent one
    
    return mask_file, segmentation_file


def copy_to_mask_outputs(original_image: Path, edited_image: Path, 
                        original_mask: Path, edited_mask: Path) -> Path:
    """
    Copy images and masks to outputs/mask directory with organized structure.
    Each execution creates a timestamped subfolder with exactly 4 files:
    - original.png (original image)
    - edited.png (edited image) 
    - original_mask.png (original mask)
    - edited_mask.png (edited mask)
    
    Args:
        original_image: Path to original image
        edited_image: Path to edited image
        original_mask: Path to original mask
        edited_mask: Path to edited mask
        
    Returns:
        Path to the mask output directory
    """
    from datetime import datetime
    
    # Create timestamped subfolder in mask directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mask_output_dir = Path(__file__).parent.parent / "outputs" / "mask" / f"run_{timestamp}"
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy exactly 4 files with clean names
    shutil.copy2(original_image, mask_output_dir / "original.png")
    shutil.copy2(edited_image, mask_output_dir / "edited.png")
    shutil.copy2(original_mask, mask_output_dir / "original_mask.png")
    shutil.copy2(edited_mask, mask_output_dir / "edited_mask.png")
    
    print(f"📁 Mask outputs saved to: {mask_output_dir}")
    print(f"📁 Files: original.png, edited.png, original_mask.png, edited_mask.png")
    return mask_output_dir


def cleanup_intermediate_files(overlay_path: Optional[Path], run_dir: Optional[Path], 
                             keep_final: bool = True) -> None:
    """
    Clean up intermediate files from reflector process.
    
    Args:
        overlay_path: Path to overlay image (keep if keep_final=True)
        run_dir: Directory containing intermediate files
        keep_final: Whether to keep final results
    """
    if not run_dir or not run_dir.exists():
        return
    
    try:
        # List of patterns to clean up
        cleanup_patterns = [
            "*.png",  # All PNG files
            "*.jpg",  # All JPG files
            "*.json", # All JSON files
        ]
        
        files_removed = 0
        for pattern in cleanup_patterns:
            for file_path in run_dir.glob(pattern):
                # Skip final results if keep_final is True
                if keep_final and overlay_path and file_path.name in overlay_path.name:
                    continue
                file_path.unlink()
                files_removed += 1
        
        print(f"🧹 Cleaned up {files_removed} intermediate files from {run_dir}")
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to clean up intermediate files: {e}")


def World_model(image: str | Path, text: str, 
                generate_masks: bool = True, 
                cleanup_intermediate: bool = True,
                box_threshold: float = 0.35,
                text_threshold: float = 0.25,
                max_iterations: int = 3,
                use_enhancer: bool = True,
                use_reflector: bool = True) -> Path:
    """
    Given an input image and a text instruction, return the goal image.
    Uses iterative validation and reflection with Gemini like Reflector.py.
    Optionally generate masks for target objects and clean up intermediate files.

    Args:
        image: Path to the input image.
        text: Text instruction for editing.
        generate_masks: Whether to generate masks for target objects (default: True)
        cleanup_intermediate: Whether to clean up intermediate files (default: True)
        box_threshold: GroundingDINO box threshold for mask generation
        text_threshold: GroundingDINO text threshold for mask generation
        max_iterations: Maximum number of validation iterations (default: 3)
        use_enhancer: Whether to use instruction enhancement (default: True)
        use_reflector: Whether to use iterative validation and reflection (default: True)

    Returns:
        Path to the resulting image.
    """

    input_image = Path(image).expanduser().resolve()
    if not input_image.exists():
        raise FileNotFoundError(f"Image not found: {input_image}")

    # Check for API key if using reflector
    api_key = None
    if use_reflector:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set (required for reflector)")

    print(f"🎯 World Model processing: {input_image.name}")
    print(f"📝 Instruction: {text}")
    print(f"🔧 Use enhancer: {use_enhancer}")
    print(f"🔄 Use reflector: {use_reflector}")
    if use_reflector:
        print(f"🔄 Maximum iterations: {max_iterations}")

    # Step 1: Instruction enhancement (optional)
    if use_enhancer:
        print(f"\n🔧 Step 1: Enhancing instruction...")
        enhanced_instruction = run_enhancer(text)
    else:
        print(f"\n🔧 Step 1: Using original instruction (enhancer disabled)")
        enhanced_instruction = text
    
    current_instruction = enhanced_instruction
    final_edited_image = None
    overlay_path = None
    
    # Choose processing mode based on reflector setting
    if use_reflector:
        # Iterative improvement loop (same as Reflector)
        for iteration in range(1, max_iterations + 1):
            print(f"\n🔄 Iteration {iteration}/{max_iterations}")
            print(f"📝 Using instruction: {current_instruction}")
            
            # Step 2: Run synthesis with current instruction
            print(f"🎨 Step 2: Running synthesis...")
            overlay_path, edited_path, run_dir = run_synthesis_without_enhance(
                input_image=input_image,
                enhanced_instruction=current_instruction,
                prefix=f"world_model_iter_{iteration}",
            )
            
            if edited_path:
                final_edited_image = edited_path
            
            # Step 3: Validate with Gemini
            print(f"🔍 Step 3: Validating with Gemini...")
            is_valid, feedback = validate_overlay_with_gemini(
                text, current_instruction, overlay_path, api_key
            )
            
            if is_valid:
                print(f"✅ Validation successful! Image meets requirements.")
                break
            else:
                print(f"❌ Validation failed: {feedback}")
                
                if iteration < max_iterations:
                    print(f"🔧 Generating revised instruction...")
                    revised_instruction = generate_revised_enhanced_text(
                        text, current_instruction, feedback, api_key
                    )
                    current_instruction = revised_instruction
                else:
                    print(f"⚠️ Maximum iterations reached. Using best available result.")
        
        print(f"\n🎉 World Model iterative process completed!")
        print(f"📊 Total iterations: {iteration}")
    else:
        # Simple single-pass synthesis (no validation/reflection)
        print(f"\n🎨 Running single-pass synthesis...")
        print(f"📝 Using instruction: {current_instruction}")
        
        overlay_path, edited_path, run_dir = run_synthesis_without_enhance(
            input_image=input_image,
            enhanced_instruction=current_instruction,
            prefix="world_model",
        )
        
        if edited_path:
            final_edited_image = edited_path
        
        print(f"\n🎉 World Model synthesis completed!")

    result_path: Optional[Path] = final_edited_image if final_edited_image else overlay_path
    if result_path is None:
        raise RuntimeError("No output image produced by synthesis")
    
    print(f"📁 Final result: {result_path}")

    # Generate masks if requested
    if generate_masks:
        print("🔍 Generating masks for target objects...")
        
        # Extract target objects from the original text
        target_objects = extract_objects(text)
        print(f"🎯 Detected target objects: {target_objects}")
        
        if target_objects:
            # Create temporary mask directories in reflector folder
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Use reflector directory for intermediate files
            reflector_dir = Path(__file__).parent.parent / "outputs" / "reflector"
            original_mask_dir = reflector_dir / f"original_mask_{timestamp}"
            edited_mask_dir = reflector_dir / f"edited_mask_{timestamp}"
            
            try:
                # Run Grounded SAM on original image
                print("🔍 Processing original image...")
                original_mask, _ = run_grounded_sam(
                    input_image, target_objects, original_mask_dir, 
                    box_threshold, text_threshold
                )
                
                # Run Grounded SAM on edited image
                print("🔍 Processing edited image...")
                edited_mask, _ = run_grounded_sam(
                    result_path, target_objects, edited_mask_dir,
                    box_threshold, text_threshold
                )
                
                # Copy only the 4 essential files to organized mask directory
                mask_output_dir = copy_to_mask_outputs(
                    input_image, result_path,
                    original_mask, edited_mask
                )
                
                print(f"✅ Mask generation completed!")
                print(f"📁 Masks saved to: {mask_output_dir}")
                
            except Exception as e:
                print(f"⚠️ Warning: Mask generation failed: {e}")
                print("Continuing without masks...")
        else:
            print("⚠️ No target objects detected, skipping mask generation")

    # Clean up intermediate files if requested
    if cleanup_intermediate:
        print("🧹 Cleaning up intermediate files...")
        cleanup_intermediate_files(overlay_path, run_dir, keep_final=True)

    print(f"✅ World Model processing completed!")
    print(f"📁 Final result: {result_path}")
    
    return result_path


__all__ = ["World_model"]


