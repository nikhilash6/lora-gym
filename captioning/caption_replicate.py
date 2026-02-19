"""
Replicate Auto-Captioner for LoRA Training
============================================
Uses Replicate API to caption images and videos for LoRA training.
Costs credits from your Replicate account — no brutal rate limits.

Usage:
  1. python -m pip install replicate
  2. Set your Replicate API token:
       $env:REPLICATE_API_TOKEN = "r8_your_token_here"
  3. Edit the CONFIG section below
  4. python caption_replicate.py

Get your token at: https://replicate.com/account/api-tokens
"""

import os
import sys
import time
import base64
import mimetypes
import json
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests not installed.")
    print("Run: python -m pip install requests")
    sys.exit(1)

# =============================================================================
# ██████  CONFIG — EDIT THESE  ██████
# =============================================================================

# Replicate API token — set as environment variable
# In PowerShell: $env:REPLICATE_API_TOKEN = "r8_your_token_here"
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "INSERT_TOKEN_HERE")

# Dataset folders
IMAGES_DIR = r"datasets\annika\images"
VIDEOS_DIR = r"datasets\annika\videos"

# Character trigger word
CHARACTER_NAME = "annika"

# Model to use on Replicate — google/gemini-2.0-flash handles images + video
REPLICATE_MODEL = "google/gemini-2.5-flash"

# Skip files that already have a .txt caption
SKIP_EXISTING = True

# File extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

# =============================================================================
# Caption prompt (same proven prompt, works across models)
# =============================================================================

SYSTEM_PROMPT = f"""You are a captioning assistant for AI model training data. Your captions must follow a strict format.

RULES:
1. Every caption starts with "{CHARACTER_NAME}" followed by what they are doing
2. Include the camera view: "full body view", "medium shot", "close up", "upper body view", etc.
3. Include the camera angle: "straight on", "from above", "from below", "from the side", "three quarter view", etc.
4. After the initial line, describe ONLY:
   - Specific movements or poses (arms raised, head tilted, stepping forward)
   - Facial expressions (smiling, frowning, looking surprised)
   - Hair or accessory motion/position (hair blowing, scarf draped)
   - Emotions or states (laughing, shouting, looking pensive)
   - Environmental interaction (picking up an object, leaning on a wall)
5. Do NOT describe the character's physical appearance, clothing colors, body type, or features
6. Do NOT describe the background or setting unless the character directly interacts with it
7. Keep the caption to 1-3 sentences maximum
8. Use simple, direct language — no flowery prose
9. Write as a single continuous caption, not a list

EXAMPLES OF GOOD CAPTIONS:
"{CHARACTER_NAME} walking forward in a full body view, straight on, arms swinging naturally with a slight bounce in their step"
"{CHARACTER_NAME} sitting and turning to look over their shoulder in a medium shot, three quarter view, expression shifting from neutral to surprised"

EXAMPLES OF BAD CAPTIONS (do not do this):
"{CHARACTER_NAME} is a red character with blocky features wearing a green hat, walking through a forest" (describes appearance + setting)
"{CHARACTER_NAME}, a voxel-style figure, stands in a colorful meadow" (describes art style + setting)"""

VIDEO_USER_PROMPT = """Caption this video following the rules exactly. Focus on the motion and actions happening over time. Output ONLY the caption text, nothing else — no quotes, no labels, no explanation."""

IMAGE_USER_PROMPT = """Caption this image following the rules exactly. Describe the pose, framing, and any expression or gesture. Output ONLY the caption text, nothing else — no quotes, no labels, no explanation."""


# =============================================================================
# Script
# =============================================================================

def validate_config():
    if "INSERT" in REPLICATE_API_TOKEN:
        print("ERROR: Replicate API token not set.")
        print()
        print("In PowerShell:")
        print('  $env:REPLICATE_API_TOKEN = "r8_your_token_here"')
        print()
        print("Get your token at: https://replicate.com/account/api-tokens")
        sys.exit(1)

    found_any = False
    for label, path in [("IMAGES_DIR", IMAGES_DIR), ("VIDEOS_DIR", VIDEOS_DIR)]:
        if path and os.path.isdir(path):
            found_any = True
        elif path:
            print(f"WARNING: {label} not found: {path}")

    if not found_any:
        print("ERROR: Neither IMAGES_DIR nor VIDEOS_DIR exist.")
        sys.exit(1)


def file_to_data_uri(file_path):
    """
    Convert a file to a data URI (base64 encoded).
    Replicate accepts data URIs for file inputs — no need to host files anywhere.
    """
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type is None:
        # Fallback for common types
        ext = file_path.suffix.lower()
        mime_map = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".bmp": "image/bmp",
            ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
            ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
        }
        mime_type = mime_map.get(ext, "application/octet-stream")

    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{data}"


def collect_files():
    files = []
    if IMAGES_DIR and os.path.isdir(IMAGES_DIR):
        for f in sorted(Path(IMAGES_DIR).iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(("image", f))
    if VIDEOS_DIR and os.path.isdir(VIDEOS_DIR):
        for f in sorted(Path(VIDEOS_DIR).iterdir()):
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                files.append(("video", f))
    return files


def caption_file(file_path, file_type):
    """
    Send an image or video to Replicate via HTTP API.
    Uses data URI so the file is sent inline — no external hosting needed.
    """
    data_uri = file_to_data_uri(file_path)
    user_prompt = IMAGE_USER_PROMPT if file_type == "image" else VIDEO_USER_PROMPT

    # Replicate predictions API — create and wait for result
    response = requests.post(
        "https://api.replicate.com/v1/models/google/gemini-2.5-flash/predictions",
        headers={
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
            "Prefer": "wait",  # Blocks until prediction completes (up to 60s)
        },
        json={
            "input": {
                "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
                "media": data_uri,
            }
        },
        timeout=120,
    )

    if response.status_code == 422:
        # Model might use different input schema — try alternate field names
        response = requests.post(
            "https://api.replicate.com/v1/models/google/gemini-2.5-flash/predictions",
            headers={
                "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
                "Content-Type": "application/json",
                "Prefer": "wait",
            },
            json={
                "input": {
                    "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
                    "image": data_uri,
                }
            },
            timeout=120,
        )

    response.raise_for_status()
    result = response.json()

    # Handle different response shapes
    output = result.get("output", "")
    if isinstance(output, list):
        output = "".join(str(chunk) for chunk in output)

    if not output:
        # Some models put it in logs or other fields
        raise RuntimeError(f"Empty output. Status: {result.get('status')}. "
                           f"Error: {result.get('error', 'none')}")

    return output.strip()


def clean_caption(caption):
    # Remove wrapping quotes
    if caption.startswith('"') and caption.endswith('"'):
        caption = caption[1:-1]
    if caption.startswith("'") and caption.endswith("'"):
        caption = caption[1:-1]

    # Ensure it starts with character name
    if not caption.lower().startswith(CHARACTER_NAME.lower()):
        print(f"  WARNING: Caption missing trigger word, prepending '{CHARACTER_NAME}'")
        caption = f"{CHARACTER_NAME} {caption}"

    return caption


def main():
    validate_config()

    # Verify token works with a simple API call
    try:
        test = requests.get(
            "https://api.replicate.com/v1/account",
            headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"},
            timeout=10,
        )
        if test.status_code == 401:
            print("ERROR: Invalid Replicate API token.")
            print("Check your token at: https://replicate.com/account/api-tokens")
            sys.exit(1)
        print(f"Authenticated with Replicate.")
    except Exception as e:
        print(f"WARNING: Could not verify token: {e}")
        print("Proceeding anyway...")

    # Collect files
    all_files = collect_files()
    if not all_files:
        print("No images or videos found.")
        print(f"  IMAGES_DIR: {IMAGES_DIR}")
        print(f"  VIDEOS_DIR: {VIDEOS_DIR}")
        sys.exit(1)

    # Filter already-captioned
    if SKIP_EXISTING:
        to_process = [(t, f) for t, f in all_files if not f.with_suffix(".txt").exists()]
        skipped = len(all_files) - len(to_process)
        if skipped > 0:
            print(f"Skipping {skipped} already-captioned files")
    else:
        to_process = all_files

    n_images = sum(1 for t, _ in to_process if t == "image")
    n_videos = sum(1 for t, _ in to_process if t == "video")
    print(f"To caption: {n_images} images + {n_videos} videos = {len(to_process)} total")
    print(f"Character: {CHARACTER_NAME}")
    print(f"Model: {REPLICATE_MODEL}")
    print("-" * 50)

    success = 0
    failed = 0

    for i, (file_type, file_path) in enumerate(to_process, 1):
        caption_path = file_path.with_suffix(".txt")
        label = "IMG" if file_type == "image" else "VID"
        print(f"\n[{i}/{len(to_process)}] [{label}] {file_path.name}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"  Captioning...", end=" ", flush=True)
                caption = caption_file(file_path, file_type)
                caption = clean_caption(caption)

                caption_path.write_text(caption, encoding="utf-8")
                print(f"done.")
                print(f"  → {caption[:120]}{'...' if len(caption) > 120 else ''}")
                success += 1

                # Small delay to be polite to the API
                if i < len(to_process):
                    time.sleep(2)
                break

            except Exception as e:
                error_str = str(e)
                if attempt < max_retries - 1:
                    wait = 15 * (attempt + 1)
                    print(f"  Error, retrying in {wait}s... ({e})")
                    time.sleep(wait)
                else:
                    print(f"  FAILED: {e}")
                    failed += 1

    print("\n" + "=" * 50)
    print(f"Done! {success} captioned, {failed} failed, "
          f"{len(all_files) - len(to_process)} skipped")
    if IMAGES_DIR and os.path.isdir(IMAGES_DIR):
        print(f"Image captions in: {IMAGES_DIR}")
    if VIDEOS_DIR and os.path.isdir(VIDEOS_DIR):
        print(f"Video captions in: {VIDEOS_DIR}")


if __name__ == "__main__":
    main()
