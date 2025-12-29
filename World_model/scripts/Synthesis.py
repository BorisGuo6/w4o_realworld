import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pycocotools.mask as mask_util


PROJECT_ROOT = Path("/home/world4omni/w4o/World_model").resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
sys.path.append(str(PROJECT_ROOT))


def run_edit_image(input_image: Path, instruction: str, prefix: str) -> Path:
    """Call edit_img.py to generate an edited image. Return the first saved image path.
    Note: Requires GEMINI_API_KEY to be set in the environment.
    """
    # Create organized output directory structure
    synthesis_dir = OUTPUT_ROOT / "synthesis"
    os.makedirs(synthesis_dir, exist_ok=True)
    
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "edit_img.py"),
        str(input_image),
        instruction,
        "--prefix", prefix,
    ]
    
    print("[synthesis] Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)

    # Find the first image saved with the prefix in the new organized structure
    edit_img_dir = OUTPUT_ROOT / "image_generation" / "edit_img"
    candidates = sorted(edit_img_dir.glob(f"{prefix}_*.*"))
    if not candidates:
        raise RuntimeError("No edited image produced. Ensure GEMINI_API_KEY is set and edit_img.py ran successfully.")
    print("[synthesis] Edited image:", candidates[0])
    return candidates[0]


def run_extract_objects(instruction: str) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "extract_objects.py"),
        instruction,
        "--json",
    ]
    print("[synthesis] Extract objects:", " ".join(cmd))
    out = subprocess.check_output(cmd, text=True)
    objs = json.loads(out.strip())
    if not isinstance(objs, list):
        return []
    # Deduplicate and keep order
    seen = set()
    res = []
    for x in objs:
        k = str(x).strip().lower()
        if k and k not in seen:
            seen.add(k)
            res.append(k)
    print("[synthesis] Objects:", res)
    return res


def run_enhance_prompt(instruction: str) -> str:
    """Call enhancer.py to enhance the instruction prompt."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "enhancer.py"),
        instruction,
    ]
    print("[synthesis] Enhance prompt:", " ".join(cmd))
    out = subprocess.check_output(cmd, text=True)
    
    # Parse the enhanced text from the output
    lines = out.strip().split('\n')
    for line in lines:
        if line.startswith("Enhanced text: "):
            enhanced = line.replace("Enhanced text: ", "").strip()
            print(f"[synthesis] Enhanced instruction: {enhanced}")
            return enhanced
    
    # Fallback to original if parsing fails
    print("[synthesis] Failed to parse enhanced text, using original instruction")
    return instruction


def run_grounded_sam(input_image: Path, objects: list[str], output_dir: Path, box_th: float, text_th: float) -> Path:
    """Run Grounded SAM on the specified image to detect target objects."""
    items = ", ".join(objects)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "test_grounded_sam.py"),
        str(input_image),
        items,
        "--output-dir",
        str(output_dir),
        "--box-threshold",
        str(box_th),
        "--text-threshold",
        str(text_th),
    ]
    print("[synthesis] Run grounded SAM:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    result_json = output_dir / "segmentation_results.json"
    if not result_json.exists():
        raise RuntimeError("Grounded SAM results JSON not found:", result_json)
    return result_json


def rle_to_mask(rle_obj: dict, height: int, width: int) -> np.ndarray:
    rle_c = {
        "counts": rle_obj["counts"].encode("utf-8") if isinstance(rle_obj.get("counts"), str) else rle_obj.get("counts"),
        "size": [height, width],
    }
    m = mask_util.decode(rle_c)  # (H, W) uint8 {0,1}
    return m.astype(bool)


def overlay_objects_on_image(base_img: np.ndarray, source_img: np.ndarray, masks: list[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """Overlay objects from source image onto base image using masks with transparency."""
    out = base_img.copy()
    for m in masks:
        # Ensure mask size matches base image; resize if necessary
        if m.shape[:2] != base_img.shape[:2]:
            m_resized = cv2.resize(m.astype(np.uint8), (base_img.shape[1], base_img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            m_resized = m
        # Extract object from source image
        obj = np.zeros_like(source_img)
        obj[m_resized] = source_img[m_resized]
        # Alpha blend onto base image
        out[m_resized] = (alpha * obj[m_resized] + (1 - alpha) * out[m_resized]).astype(out.dtype)
    return out


def main():
    parser = argparse.ArgumentParser(description="Synthesize edited image with grounded objects overlaid from the edited image onto the original.")
    parser.add_argument("image", type=str, help="input image path")
    parser.add_argument("instruction", type=str, help="text instruction (will be passed to edit_img.py)")
    parser.add_argument("--alpha", type=float, default=0.5, help="overlay transparency for objects (0-1, higher = more visible)")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="GroundingDINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold")
    parser.add_argument("--out-prefix", type=str, default="reflection", help="prefix for edited image and outputs")
    # High performance flag removed. Default selection: prefer 2.5, fallback to 2.0.
    args = parser.parse_args()

    input_image = Path(args.image).expanduser().resolve()
    if not input_image.exists():
        raise FileNotFoundError(f"Image not found: {input_image}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    edit_prefix = f"{args.out_prefix}_edit_{timestamp}"
    
    # Create organized output directory structure
    synthesis_dir = OUTPUT_ROOT / "synthesis"
    run_dir = synthesis_dir / f"{args.out_prefix}_run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"[synthesis] 📁 Run directory: {run_dir}")

    # 1) Enhance the instruction prompt
    print("[synthesis] 🔧 Enhancing instruction prompt...")
    enhanced_instruction = run_enhance_prompt(args.instruction)
    
    # 2) Generate edited image with enhanced prompt
    edited_path = run_edit_image(input_image, enhanced_instruction, edit_prefix)

    # 3) Extract objects from instruction
    objects = run_extract_objects(args.instruction)
    if not objects:
        print("[synthesis] No objects extracted; skipping mask overlay. Using edited image as final overlay.")
        print("[synthesis] Edited image:", edited_path)
        # Save a copy of the edited image into the synthesis directory as overlay
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        overlay_filename = f"{args.out_prefix}_overlay_{timestamp_str}.png"
        out_path = synthesis_dir / overlay_filename
        try:
            edited_img_copy = cv2.imread(str(edited_path))
            if edited_img_copy is None:
                raise RuntimeError("Failed to read edited image for overlay copy.")
            cv2.imwrite(str(out_path), edited_img_copy)
            print(f"[synthesis] 🎯 Final overlay saved: {overlay_filename}")
            print(f"[synthesis] 📁 Overlay image: {out_path}")
        except Exception as e:
            print(f"[synthesis] ⚠️ Failed to save overlay copy: {e}")
        return

    # 4) Run grounded SAM on the EDITED image (not the original)
    print("[synthesis] 🔍 Running Grounded SAM on the edited image to detect target objects...")
    result_json = run_grounded_sam(edited_path, objects, run_dir, args.box_threshold, args.text_threshold)
    with open(result_json, "r") as f:
        result = json.load(f)
    ann = result.get("annotations", [])
    H = int(result.get("img_height", 0))
    W = int(result.get("img_width", 0))

    # Decode masks
    masks = []
    for a in ann:
        rle = a.get("segmentation")
        if not rle:
            continue
        try:
            masks.append(rle_to_mask(rle, H, W))
        except Exception:
            continue
    if not masks:
        print("[synthesis] No masks decoded; skipping overlay. Edited image only.")
        print("[synthesis] Edited image:", edited_path)
        return

    # 5) Overlay objects from the EDITED image onto the ORIGINAL image
    print("[synthesis] 🎨 Overlaying objects from edited image onto original image...")
    original_img = cv2.imread(str(input_image))
    edited_img = cv2.imread(str(edited_path))
    if original_img is None or edited_img is None:
        raise RuntimeError("Failed to read original or edited image.")

    # If sizes differ, resize edited image to original size for consistent overlay
    if edited_img.shape[:2] != original_img.shape[:2]:
        edited_img = cv2.resize(edited_img, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        print(f"[synthesis] Resized edited image from {result.get('img_height', 'unknown')}x{result.get('img_width', 'unknown')} to {original_img.shape[0]}x{original_img.shape[1]}")

    # Overlay objects from edited image onto original image
    blended = overlay_objects_on_image(original_img, edited_img, masks, alpha=float(args.alpha))

    # Save final overlay image with improved naming
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    overlay_filename = f"{args.out_prefix}_overlay_{timestamp_str}.png"
    out_path = synthesis_dir / overlay_filename
    cv2.imwrite(str(out_path), blended)
    
    # Find segmentation image path
    segmentation_filename = f"segmentation_{timestamp_str}.jpg"
    segmentation_path = run_dir / segmentation_filename
    
    print(f"[synthesis] 🎯 Final overlay saved: {overlay_filename}")
    print(f"[synthesis] 📁 Original image: {input_image}")
    print(f"[synthesis] 📁 Edited image: {edited_path}")
    print(f"[synthesis] 📁 Overlay image: {out_path}")
    print(f"[synthesis] 📄 Grounded SAM segmentation: {segmentation_path}")
    print(f"[synthesis] 🎉 Synthesis pipeline completed successfully!")
    print(f"[synthesis] 💡 The overlay shows objects from the edited image overlaid on the original image with {args.alpha} transparency")


if __name__ == "__main__":
    main()


