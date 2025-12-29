#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from io import BytesIO
import base64
from PIL import Image
import argparse
import time
import re
import sys
from datetime import datetime
from pathlib import Path
PROJECT_ROOT = Path("/home/world4omni/w4o/World_model")
sys.path.append(str(PROJECT_ROOT))
from google import genai
from google.genai import types
from google.genai.errors import ClientError


THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.normpath(os.path.join(THIS_DIR, ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

# Model configuration
HIGH_PERF_MODEL = "gemini-2.5-flash-image-preview"  # Preferred high-performance model
FALLBACK_MODEL = "gemini-2.0-flash-preview-image-generation"  # Fallback model if high-perf unavailable

def check_model_availability(client, model_name):
    """Check if a model is available via API."""
    try:
        models = list(client.models.list())
        model_names = [model.name for model in models]
        
        # Check both with and without 'models/' prefix
        if model_name in model_names:
            return True
        elif model_name.startswith('models/'):
            # If model_name already has prefix, check without it
            return model_name[7:] in model_names
        else:
            # If model_name doesn't have prefix, check with it
            return f"models/{model_name}" in model_names
            
    except Exception as e:
        print(f"⚠️ Warning: Could not check model availability: {e}")
        return False

def get_available_model(client):
    """Get the best available model, preferring high-performance over fallback."""
    if check_model_availability(client, HIGH_PERF_MODEL):
        print(f"🚀 High-performance model '{HIGH_PERF_MODEL}' is available")
        return HIGH_PERF_MODEL
    else:
        print(f"⚠️ High-performance model '{HIGH_PERF_MODEL}' not available, using fallback '{FALLBACK_MODEL}'")
        return FALLBACK_MODEL


def _decode_to_bytes(raw):
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("data:"):
            comma_idx = s.find(",")
            if comma_idx != -1:
                s = s[comma_idx + 1 :]
        try:
            return base64.b64decode(s, validate=False)
        except Exception:
            pass
        try:
            pad = "=" * (-len(s) % 4)
            return base64.urlsafe_b64decode(s + pad)
        except Exception:
            pass
    return None


def debug_print_response(response) -> None:
    print("--- Debug: Response Overview ---")
    candidates = getattr(response, "candidates", []) or []
    print(f"candidates: {len(candidates)}")
    for ci, cand in enumerate(candidates):
        role = getattr(getattr(cand, "content", None), "role", None)
        parts = getattr(getattr(cand, "content", None), "parts", []) or []
        print(f"  candidate[{ci}]: role={role}, parts={len(parts)}")
        for pi, part in enumerate(parts):
            has_text = bool(getattr(part, "text", None))
            inline = getattr(part, "inline_data", None)
            has_inline = inline is not None
            mime = getattr(inline, "mime_type", None) if has_inline else None
            data_len = len(getattr(inline, "data", b"")) if has_inline else 0
            text_preview = getattr(part, "text", "")
            if isinstance(text_preview, str):
                text_preview = text_preview[:160]
            print(
                f"    part[{pi}]: type={'text' if has_text else ('inline_data' if has_inline else 'unknown')}, "
                f"mime={mime}, data_len={data_len}, text='{text_preview}'"
            )
    if getattr(response, "text", None):
        print("text (top-level):", response.text[:200])
    fcalls = getattr(response, "function_calls", None)
    if fcalls:
        print("function_calls:", fcalls)
    print("--- End Debug ---")


def save_inline_images(response, prefix: str) -> int:
    """Save inline images from response with improved naming."""
    count = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create organized output directory structure
    edit_output_dir = os.path.join(OUTPUT_DIR, "image_generation", "edit_img")
    os.makedirs(edit_output_dir, exist_ok=True)
    
    for ci, cand in enumerate(getattr(response, "candidates", []) or []):
        content = getattr(cand, "content", None)
        if not content:
            continue
        for pi, part in enumerate(getattr(content, "parts", []) or []):
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                try:
                    raw_data = getattr(inline, "data", None)
                    img_bytes = _decode_to_bytes(raw_data)
                    if not img_bytes:
                        continue
                    try:
                        img = Image.open(BytesIO(img_bytes))
                    except Exception:
                        text_guess = img_bytes.decode("utf-8", errors="ignore").strip()
                        decoded = _decode_to_bytes(text_guess) if text_guess else None
                        if not decoded:
                            raise
                        img = Image.open(BytesIO(decoded))

                    mime_type = getattr(inline, "mime_type", None) or getattr(inline, "mime", None)
                    ext = "png"
                    if mime_type in ("image/jpeg", "image/jpg"):
                        ext = "jpg"
                    elif mime_type == "image/webp":
                        ext = "webp"

                    # Improved filename format: prefix_timestamp_candidate_part.ext
                    filename = f"{prefix}_{timestamp}_c{ci:02d}_p{pi:02d}.{ext}"
                    out_path = os.path.join(edit_output_dir, filename)
                    img.save(out_path)
                    print(f"✅ Saved edited image: {filename}")
                    count += 1
                except Exception as e:
                    print(f"❌ Failed to save image: {e}")
    return count



# Note: prompt_enhancer function has been moved to enhancer.py
# This script now expects pre-enhanced prompts as input


def extract_retry_delay(error_message: str) -> int:
    """Extract retry delay from error message. Returns delay in seconds, or -1 if not found."""
    # Try multiple patterns to handle different error message formats
    
    # Pattern 1: Full JSON format with quotes
    patterns = [
        r'"retryDelay":\s*"(\d+)([smhd])"',  # Original pattern
        r'retryDelay["\s]*:\s*["\s]*(\d+)([smhd])',  # More flexible pattern
        r'retryDelay["\s]*:\s*["\s]*(\d+)([smhd])["\s]*',  # Even more flexible
    ]
    
    for pattern in patterns:
        retry_match = re.search(pattern, error_message)
        if retry_match:
            try:
                value = int(retry_match.group(1))
                unit = retry_match.group(2)
                if unit == 's':
                    return value
                elif unit == 'm':
                    return value * 60
                elif unit == 'h':
                    return value * 3600
                elif unit == 'd':
                    return value * 86400
            except (ValueError, IndexError):
                continue
    
    # If no pattern matches, try to find any time-like pattern in the error message
    # This is a fallback for cases where the error message format is unexpected
    time_patterns = [
        r'(\d+)\s*seconds?',
        r'(\d+)\s*minutes?',
        r'(\d+)\s*hours?',
        r'(\d+)\s*days?',
        r'(\d+)s',
        r'(\d+)m',
        r'(\d+)h',
        r'(\d+)d',
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, error_message.lower())
        if match:
            try:
                value = int(match.group(1))
                if 'second' in pattern or pattern.endswith('s'):
                    return value
                elif 'minute' in pattern or pattern.endswith('m'):
                    return value * 60
                elif 'hour' in pattern or pattern.endswith('h'):
                    return value * 3600
                elif 'day' in pattern or pattern.endswith('d'):
                    return value * 86400
            except ValueError:
                continue
    
    return -1


def image_generator_with_retry(model, client, image, text, max_retries=3):
    """Generate image with automatic retry on quota limits."""
    for attempt in range(max_retries):
        try:
            # Build multimodal contents: instruction + image
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=text),
                        types.Part(inline_data=types.Blob(mime_type="image/png", data=image)),
                    ],
                )
            ]

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
            )

            debug_print_response(response)
            return response

        except ClientError as e:
            if e.code == 429:  # RESOURCE_EXHAUSTED
                error_message = str(e)
                retry_delay = extract_retry_delay(error_message)
                
                if retry_delay > 0 and retry_delay <= 60:  # Wait if delay <= 1 minute
                    print(f"⏳ Quota limit reached. Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    print(f"🔄 Retrying... (attempt {attempt + 2}/{max_retries})")
                    continue
                elif retry_delay > 60:
                    print(f"⏰ Quota limit reached. Suggested wait time: {retry_delay} seconds (>1 minute)")
                    print("💡 Consider upgrading your plan or waiting longer before retrying.")
                    break
                else:
                    print("⚠️ Quota limit reached but no retry delay found in error message.")
                    break
            else:
                # Re-raise non-quota errors
                raise
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying... (attempt {attempt + 2}/{max_retries})")
                time.sleep(2)  # Wait 2 seconds before retry
                continue
            else:
                raise
    
    # If we get here, all retries failed
    raise RuntimeError(f"Failed to generate image after {max_retries} attempts")


def image_generator(model, client, image, text):
    """Legacy function for backward compatibility."""
    return image_generator_with_retry(model, client, image, text)


def main():
    parser = argparse.ArgumentParser(description="Edit an image with a text instruction using Google GenAI SDK.")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("instruction", type=str, help="Text instruction")
    parser.add_argument("--model", help=f"Force specific model name (overrides auto-selection)")
    parser.add_argument("--high-perf", action="store_true", help=f"Force use of high-performance model: {HIGH_PERF_MODEL}")
    parser.add_argument("--fallback", action="store_true", help=f"Force use of fallback model: {FALLBACK_MODEL}")
    parser.add_argument("--prefix", default="edited_image", help="Output filename prefix")
    # Note: --no-enhance removed - enhancement should be done externally via enhancer.py
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts for quota limits (default: 3)")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        print("Warning: GEMINI_API_KEY is empty; please set it via environment variable.")
    client = genai.Client(api_key=api_key)

    # Determine which model to use
    if args.model:
        # User specified a specific model
        model_name = args.model
        print(f"🎯 Using user-specified model: {model_name}")
    elif args.high_perf:
        # User wants to force high-performance model
        model_name = HIGH_PERF_MODEL
        print(f"🚀 Using forced high-performance model: {model_name}")
    elif args.fallback:
        # User wants to force fallback model
        model_name = FALLBACK_MODEL
        print(f"📝 Using forced fallback model: {model_name}")
    else:
        # Auto-select best available model
        model_name = get_available_model(client)
        print(f"🤖 Auto-selected model: {model_name}")

    # Create organized output directory structure
    edit_output_dir = os.path.join(OUTPUT_DIR, "image_generation", "edit_img")
    os.makedirs(edit_output_dir, exist_ok=True)
    print(f"📁 Output directory: {edit_output_dir}")
    print(f"🤖 Using model: {model_name}")

    # Read input image bytes
    image_path = os.path.abspath(args.image)
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # Use the instruction directly (enhancement should be done externally via enhancer.py)
    instruction = args.instruction

    # Generate image with instruction and automatic retry
    try:
        response_image = image_generator_with_retry(model_name, client, img_bytes, instruction, args.max_retries)

        if getattr(response_image, "text", None):
            print("Text preview:", response_image.text[:200])

        saved = save_inline_images(response_image, prefix=args.prefix)
        print(f"🎉 Successfully saved {saved} edited image(s)")
        
    except Exception as e:
        print(f"❌ Failed to generate image: {e}")
        if "quota" in str(e).lower() or "429" in str(e):
            print("💡 This appears to be a quota limit issue. Consider:")
            print("   - Upgrading your Gemini API plan")
            print("   - Waiting longer before retrying")
            print("   - Using a different model with --model parameter")
        sys.exit(1)


if __name__ == "__main__":
    main()


