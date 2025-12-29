#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from google import genai
from google.genai import types


PROJECT_ROOT = Path("/home/world4omni/w4o/World_model").resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def run_enhancer(instruction: str) -> str:
    """Call enhancer.py to enhance the instruction prompt."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "enhancer.py"),
        instruction,
    ]
    print("[reflector] 🔧 Enhancing instruction prompt...")
    out = subprocess.check_output(cmd, text=True)
    
    # Parse the enhanced text from the output
    lines = out.strip().split('\n')
    for line in lines:
        if line.startswith("Enhanced text: "):
            enhanced = line.replace("Enhanced text: ", "").strip()
            print(f"[reflector] Enhanced instruction: {enhanced}")
            return enhanced
    
    # Fallback to original if parsing fails
    print("[reflector] Failed to parse enhanced text, using original instruction")
    return instruction


def run_synthesis_without_enhance(input_image: Path, enhanced_instruction: str, prefix: str) -> tuple[Path, Path]:
    """Run Synthesis.py with enhanced instruction (bypassing its internal enhancement)."""
    
    # Create a temporary modified Synthesis.py that skips enhancement
    temp_synthesis = create_temp_synthesis()
    
    try:
        # Create organized output directory structure
        reflector_dir = OUTPUT_ROOT / "reflector"
        os.makedirs(reflector_dir, exist_ok=True)
        
        # Run the temporary synthesis script
        cmd = [
            sys.executable,
            str(temp_synthesis),
            str(input_image),
            enhanced_instruction,
            "--out-prefix", prefix,
        ]
        
        # Default behavior: rely on Synthesis.py auto-selection.
        # Synthesis defaults to 2.5 and falls back to 2.0 when unavailable.
        
        print("[reflector] Running synthesis with enhanced instruction...")
        subprocess.run(cmd, check=True)
        
        # Find the generated files
        synthesis_dir = OUTPUT_ROOT / "synthesis"
        overlay_files = list(synthesis_dir.glob(f"{prefix}_overlay_*"))
        run_dirs = [d for d in synthesis_dir.iterdir() if d.is_dir() and prefix in d.name and "run_" in d.name]
        
        if not overlay_files:
            raise RuntimeError("No overlay image produced by synthesis")
        if not run_dirs:
            raise RuntimeError("No synthesis run directory found")
        
        overlay_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        overlay_path = overlay_files[0]
        run_dir = run_dirs[0]
        
        # Find the edited image from image_generation
        edit_img_dir = OUTPUT_ROOT / "image_generation" / "edit_img"
        edited_files = list(edit_img_dir.glob(f"{prefix}_edit_*"))
        if edited_files:
            edited_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            edited_path = edited_files[0]
        else:
            edited_path = None
        
        return overlay_path, edited_path, run_dir
        
    finally:
        # Clean up temporary file
        if temp_synthesis.exists():
            temp_synthesis.unlink()


def create_temp_synthesis() -> Path:
    """Create a temporary Synthesis.py that skips enhancement."""
    temp_file = SCRIPTS_DIR / "temp_synthesis.py"
    
    # Read original Synthesis.py
    with open(SCRIPTS_DIR / "Synthesis.py", 'r') as f:
        content = f.read()
    
    # Replace the enhancement call with direct usage
    modified_content = content.replace(
        "enhanced_instruction = run_enhance_prompt(args.instruction)",
        "enhanced_instruction = args.instruction  # Use instruction directly (already enhanced)"
    )
    modified_content = modified_content.replace(
        "print(\"[synthesis] 🔧 Enhancing instruction prompt...\")",
        "print(\"[synthesis] 🔧 Using pre-enhanced instruction...\")"
    )
    
    # Write temporary file
    with open(temp_file, 'w') as f:
        f.write(modified_content)
    
    return temp_file


def validate_overlay_with_gemini(original_text: str, enhanced_text: str, overlay_image_path: Path, api_key: str) -> tuple[bool, str]:
    """Use Gemini to validate if the overlay image meets the original instruction requirements."""
    
    client = genai.Client(api_key=api_key)
    
    # Read and encode the overlay image
    with open(overlay_image_path, "rb") as f:
        image_bytes = f.read()
    
    validation_prompt = f"""
You are an expert image analysis assistant. Your task is to evaluate whether an overlay image successfully demonstrates the requested image editing task.

**Original User Request:** {original_text}

**Enhanced Instruction:** {enhanced_text}

**Task:** Analyze the overlay image and determine if it successfully shows the requested changes.

**Evaluation Criteria:**
1. Does the overlay image show the main object(s) from the original instruction?
2. Are the objects positioned/arranged as requested in the instruction?
3. Is the visual result clear and matches the intent of the request?
4. Are there any obvious errors or missing elements?

**Response Format:**
- First line: "VALID: Yes" or "VALID: No"
- If valid, provide a brief confirmation
- If not valid, provide specific feedback on what needs to be improved

**Important:** Be strict but fair. Only mark as valid if the image clearly demonstrates the requested changes.
"""
    
    try:
        # Create the image part with proper MIME type detection
        image_part = types.Part(
            inline_data=types.Blob(
                mime_type="image/png" if overlay_image_path.suffix.lower() == ".png" else "image/jpeg",
                data=image_bytes
            )
        )
        
        # Create the text part
        text_part = types.Part(text=validation_prompt)
        
        # Create the content with both parts
        content = types.Content(
            role="user",
            parts=[text_part, image_part]
        )
        
        print(f"[reflector] Sending image to Gemini for validation: {overlay_image_path}")
        print(f"[reflector] Image size: {len(image_bytes)} bytes")
        print(f"[reflector] Image MIME type: {'image/png' if overlay_image_path.suffix.lower() == '.png' else 'image/jpeg'}")
        
        # Try different approaches for sending the image
        try:
            # Method 1: Try gemini-2.5-pro first (best for text analysis)
            print("[reflector] Trying method 1: gemini-2.5-pro")
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[content],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1024
                )
            )
            
            if hasattr(response, 'text') and response.text:
                result_text = response.text.strip()
                print(f"[reflector] Gemini validation result: {result_text}")
                
                # Parse the result
                if result_text.startswith("VALID: Yes"):
                    return True, "Image meets requirements"
                elif result_text.startswith("VALID: No"):
                    # Extract feedback for improvement
                    feedback = result_text.replace("VALID: No", "").strip()
                    return False, feedback
                else:
                    # Fallback parsing
                    if "valid" in result_text.lower() and "yes" in result_text.lower():
                        return True, "Image meets requirements"
                    else:
                        return False, "Image does not meet requirements"
            else:
                print("[reflector] No text response from Gemini, trying alternative method...")
                raise Exception("No text response")
                
        except Exception as e1:
            print(f"[reflector] Method 1 failed: {e1}, trying alternative approach...")
            
            # Method 2: Try gemini-2.0-flash as fallback (more reliable for images)
            try:
                print("[reflector] Trying method 2: gemini-2.0-flash")
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[content],
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=1024
                    )
                )
                
                if hasattr(response, 'text') and response.text:
                    result_text = response.text.strip()
                    print(f"[reflector] Gemini validation result (method 2): {result_text}")
                    
                    # Parse the result
                    if result_text.startswith("VALID: Yes"):
                        return True, "Image meets requirements"
                    elif result_text.startswith("VALID: No"):
                        feedback = result_text.replace("VALID: No", "").strip()
                        return False, feedback
                    else:
                        if "valid" in result_text.lower() and "yes" in result_text.lower():
                            return True, "Image meets requirements"
                        else:
                            return False, "Image does not meet requirements"
                else:
                    raise Exception("No text response from method 2")
                    
            except Exception as e2:
                print(f"[reflector] Method 2 also failed: {e2}")
                return False, f"Both validation methods failed: {e1}, {e2}"
            
    except Exception as e:
        print(f"[reflector] Gemini validation failed: {e}")
        return False, f"Validation error: {e}"


def generate_revised_enhanced_text(original_text: str, enhanced_text: str, feedback: str, api_key: str) -> str:
    """Generate a revised enhanced text based on Gemini's feedback."""
    
    client = genai.Client(api_key=api_key)
    
    revision_prompt = f"""
You are an expert prompt engineer for AI image editing. Your task is to improve an enhanced prompt based on feedback.

**Original User Request:** {original_text}

**Previous Enhanced Prompt:** {enhanced_text}

**Feedback for Improvement:** {feedback}

**Task:** Generate a revised and improved enhanced prompt that addresses the feedback.

**Guidelines:**
1. Keep the core structure of the enhanced prompt
2. Address the specific issues mentioned in the feedback
3. Make the instructions more precise and clear
4. Ensure the prompt will generate better results
5. Maintain the same format and style

**Response:** Return only the revised enhanced prompt without additional explanation.
"""
    
    try:
        print(f"[reflector] Generating revised instruction based on feedback: {feedback}")
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[types.Content(
                role="user",
                parts=[types.Part(text=revision_prompt)]
            )],
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048
            )
        )
        
        if hasattr(response, 'text') and response.text:
            revised_text = response.text.strip()
            print(f"[reflector] Revised enhanced text: {revised_text}")
            return revised_text
        else:
            print("[reflector] No revised text generated, using original enhanced text")
            print(f"[reflector] Response object: {response}")
            print(f"[reflector] Response attributes: {dir(response)}")
            if hasattr(response, 'candidates'):
                print(f"[reflector] Candidates: {response.candidates}")
            return enhanced_text
            
    except Exception as e:
        print(f"[reflector] Failed to generate revised text: {e}")
        return enhanced_text


def main():
    parser = argparse.ArgumentParser(description="Iterative image editing with Gemini validation")
    parser.add_argument("image", type=str, help="Input original image path")
    parser.add_argument("instruction", type=str, help="Text instruction for image editing")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum iteration attempts (default: 3)")
    # High performance flag removed. Default model selection uses 2.5 with fallback to 2.0.
    parser.add_argument("--out-prefix", type=str, default="reflector", help="Prefix for output files")
    args = parser.parse_args()

    input_image = Path(args.image).expanduser().resolve()
    if not input_image.exists():
        raise FileNotFoundError(f"Image not found: {input_image}")

    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.out_prefix}_{timestamp}"
    
    # Create output directory
    reflector_dir = OUTPUT_ROOT / "reflector"
    os.makedirs(reflector_dir, exist_ok=True)
    
    print(f"[reflector] 🎯 Starting iterative image editing process...")
    print(f"[reflector] 📁 Output directory: {reflector_dir}")
    print(f"[reflector] 🔄 Maximum iterations: {args.max_iterations}")
    
    # Step 1: Initial enhancement
    print(f"\n[reflector] 🔧 Step 1: Enhancing instruction...")
    enhanced_instruction = run_enhancer(args.instruction)
    
    current_instruction = enhanced_instruction
    final_edited_image = None
    
    # Iterative improvement loop
    for iteration in range(1, args.max_iterations + 1):
        print(f"\n[reflector] 🔄 Iteration {iteration}/{args.max_iterations}")
        print(f"[reflector] 📝 Using instruction: {current_instruction}")
        
        # Step 2: Run synthesis with current instruction
        print(f"[reflector] 🎨 Step 2: Running synthesis...")
        overlay_path, edited_path, run_dir = run_synthesis_without_enhance(
            input_image, current_instruction, f"{prefix}_iter_{iteration}"
        )
        
        if edited_path:
            final_edited_image = edited_path
        
        # Step 3: Validate with Gemini
        print(f"[reflector] 🔍 Step 3: Validating with Gemini...")
        is_valid, feedback = validate_overlay_with_gemini(
            args.instruction, current_instruction, overlay_path, api_key
        )
        
        if is_valid:
            print(f"[reflector] ✅ Validation successful! Image meets requirements.")
            break
        else:
            print(f"[reflector] ❌ Validation failed: {feedback}")
            
            if iteration < args.max_iterations:
                print(f"[reflector] 🔧 Generating revised instruction...")
                revised_instruction = generate_revised_enhanced_text(
                    args.instruction, current_instruction, feedback, api_key
                )
                current_instruction = revised_instruction
            else:
                print(f"[reflector] ⚠️ Maximum iterations reached. Using best available result.")
    
    # Final summary
    print(f"\n[reflector] 🎉 Iterative editing process completed!")
    print(f"[reflector] 📊 Total iterations: {iteration}")
    print(f"[reflector] 📁 Final overlay: {overlay_path}")
    
    if final_edited_image:
        print(f"[reflector] 🖼️ Final edited image: {final_edited_image}")
    else:
        print(f"[reflector] ⚠️ No edited image was generated")
    
    # Save final results
    final_summary = {
        "original_image": str(input_image),
        "original_instruction": args.instruction,
        "final_enhanced_instruction": current_instruction,
        "total_iterations": iteration,
        "final_overlay": str(overlay_path),
        "final_edited_image": str(final_edited_image) if final_edited_image else None,
        "validation_successful": is_valid,
        "timestamp": datetime.now().isoformat()
    }
    
    summary_file = reflector_dir / f"{prefix}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"[reflector] 📝 Summary saved to: {summary_file}")
    print(f"[reflector] 🎯 Process completed successfully!")


if __name__ == "__main__":
    main()
