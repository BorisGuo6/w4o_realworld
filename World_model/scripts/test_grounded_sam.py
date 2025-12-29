#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import cv2
import torch
import numpy as np
import pycocotools.mask as mask_util
from pathlib import Path
from datetime import datetime

# Default path roots
PROJECT_ROOT = Path("/home/world4omni/w4o/World_model").resolve()
GSAM2_ROOT = PROJECT_ROOT / "3rdparty" / "Grounded-SAM-2"

# Inference hyperparameters (can be modified as needed)
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Import local modules from Grounded-SAM-2
sys.path.insert(0, str(GSAM2_ROOT))
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
from grounding_dino.groundingdino.util.inference import (  # noqa: E402
    load_model,
    load_image,
    predict,
)
from torchvision.ops import box_convert  # noqa: E402


def single_mask_to_rle(mask: np.ndarray):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _normalize_prompt(items: str) -> str:
    # Accept comma/space/period separators and normalize to "a. b. c."
    raw = [t.strip().lower() for t in items.replace("\n", " ").replace(",", ".").split(".")]
    toks = [t for t in raw if t]
    if not toks:
        return ""
    return ". ".join(toks) + "."


def parse_args():
    parser = argparse.ArgumentParser(description="Run Grounded SAM 2 on a single image")
    parser.add_argument("image", type=str, help="input image path")
    parser.add_argument("items", type=str, help="items for SAM to mask, e.g. 'tomato, pan' or 'tomato. pan.'")
    parser.add_argument("--output-dir", type=str, default=None, help="directory to save outputs (optional)")
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD, help="GroundingDINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD, help="GroundingDINO text threshold")
    return parser.parse_args()


def main():
    """Main function to run Grounded SAM 2 inference."""
    args = parse_args()

    img_path = Path(args.image).expanduser().resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    text_prompt = _normalize_prompt(args.items)
    
    # Create organized output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        # Default organized structure
        output_dir = PROJECT_ROOT / "outputs" / "grounded_sam" / "segmentation" / f"run_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Absolute paths for config and weights
    sam2_checkpoint = GSAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
    # Note: build_sam2's config_file requires Hydra relative config name, not filesystem absolute path
    sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    grounding_dino_cfg = GSAM2_ROOT / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    grounding_dino_ckpt = GSAM2_ROOT / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"

    # Build SAM2 predictor
    sam2_model = build_sam2(sam2_model_cfg, str(sam2_checkpoint), device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # Build GroundingDINO model
    grounding_model = load_model(
        model_config_path=str(grounding_dino_cfg),
        model_checkpoint_path=str(grounding_dino_ckpt),
        device=DEVICE,
    )

    # Load image and text prompt
    text = text_prompt
    image_source, image = load_image(str(img_path))
    sam2_predictor.set_image(image_source)

    # Detect bounding boxes
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
        device=DEVICE,
    )

    # Process to xyxy format and perform segmentation
    h, w, _ = image_source.shape
    if boxes.numel() == 0:
        # If no detections, output empty structure and return
        results_file = output_dir / "segmentation_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "image_path": str(img_path),
                "annotations": [],
                "box_format": "xyxy",
                "img_width": int(w),
                "img_height": int(h),
            }, f, indent=4)
        print("⚠️ No detections found. Empty results saved.")
        print(f"📄 Results saved to: {results_file}")
        return
    
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # Convert mask shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.numpy().tolist()
    class_names = labels
    class_ids = np.array(list(range(len(class_names))))

    # Visualize and save results
    img_bgr = cv2.imread(str(img_path))
    annotated = img_bgr.copy()
    # Draw bounding boxes
    for i, (x1, y1, x2, y2) in enumerate(input_boxes.astype(int)):
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[i]} {confidences[i]:.2f}"
        cv2.putText(annotated, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Overlay masks
    for i, m in enumerate(masks.astype(bool)):
        color = (0, 0, 255)
        overlay = annotated.copy()
        overlay[m] = (overlay[m] * 0.3 + np.array(color) * 0.7).astype(np.uint8)
        annotated = overlay

    # Save annotated images with improved naming
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    bounding_boxes_file = output_dir / f"bounding_boxes_{timestamp_str}.jpg"
    segmentation_file = output_dir / f"segmentation_{timestamp_str}.jpg"
    mask_file = output_dir / f"segmentation_mask_{timestamp_str}.png"
    
    cv2.imwrite(str(bounding_boxes_file), annotated)
    cv2.imwrite(str(segmentation_file), annotated)
    # Save a black-white mask image (union of all instance masks)
    try:
        union_mask = np.zeros((h, w), dtype=np.uint8)
        for m in masks.astype(bool):
            union_mask[m] = 255
        cv2.imwrite(str(mask_file), union_mask)
        print(f"🖤🤍 Binary mask saved: {mask_file.name}")
    except Exception as e:
        print(f"⚠️ Failed to save binary mask: {e}")
    
    print(f"🖼️ Bounding boxes saved: {bounding_boxes_file.name}")
    print(f"🎯 Segmentation saved: {segmentation_file.name}")

    # Save JSON results
    mask_rles = [single_mask_to_rle(mask) for mask in masks]
    # Normalize scores to float
    raw_scores = scores.tolist() if hasattr(scores, "tolist") else scores
    norm_scores = []
    for s in raw_scores:
        if isinstance(s, (list, tuple, np.ndarray)):
            s = s[0] if len(s) > 0 else 0.0
        try:
            norm_scores.append(float(s))
        except Exception:
            norm_scores.append(0.0)

    results = {
        "image_path": str(img_path),
        "annotations": [
            {
                "class_name": class_name,
                "bbox": box.tolist(),
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, norm_scores)
        ],
        "box_format": "xyxy",
        "img_width": int(w),
        "img_height": int(h),
    }
    
    results_file = output_dir / "segmentation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"📄 Results saved to: {results_file}")
    print(f"🎉 Successfully processed {len(class_names)} object(s): {', '.join(class_names)}")


if __name__ == "__main__":
    main()


