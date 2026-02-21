"""Local Setup Script for Wan 2.2 LoRA Training
=============================================
Downloads models, clones musubi-tuner, and sets up the directory structure
for local training on a consumer GPU (RTX 3090/4090).

Python-based (no bash) for Windows compatibility.
Idempotent — skips already-downloaded files, safe to re-run.

Usage:
  python setup_local.py
  python setup_local.py --base_dir D:/my_training
  python setup_local.py --hf_token hf_xxxxx
  python setup_local.py --include_i2v
  python setup_local.py --comfyui_dir C:/ComfyUI
"""

import os
import sys
import subprocess
import shutil
import platform
import argparse

# =============================================================================
# ██████  CONFIG  ██████
# =============================================================================

# Default base directory (relative to where this script lives)
DEFAULT_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_training")

# T2V model files (~40GB total)
MODELS = {
    "T2V HIGH DiT": {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors",
        "dest_name": "wan2.2_t2v_high_noise_14B_fp16.safetensors",
        "size_gb": 14,
    },
    "T2V LOW DiT": {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors",
        "dest_name": "wan2.2_t2v_low_noise_14B_fp16.safetensors",
        "size_gb": 14,
    },
    "VAE": {
        "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "filename": "split_files/vae/wan_2.1_vae.safetensors",
        "dest_name": "wan_2.1_vae.safetensors",
        "size_gb": 0.35,
    },
    "T5 Text Encoder": {
        "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
        "filename": "models_t5_umt5-xxl-enc-bf16.pth",
        "dest_name": "models_t5_umt5-xxl-enc-bf16.pth",
        "size_gb": 10,
    },
}

# I2V models (optional, ~30GB extra)
I2V_MODELS = {
    "I2V HIGH DiT": {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
        "dest_name": "wan2.2_i2v_high_noise_14B_fp16.safetensors",
        "size_gb": 14,
    },
    "I2V LOW DiT": {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
        "dest_name": "wan2.2_i2v_low_noise_14B_fp16.safetensors",
        "size_gb": 14,
    },
}


# =============================================================================
# SETUP STEPS
# =============================================================================

def check_prerequisites():
    """Check that required tools are available."""
    print("Checking prerequisites...")

    # Python version
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(f"  ERROR: Python 3.10+ required (you have {major}.{minor})")
        sys.exit(1)
    print(f"  Python {major}.{minor} — OK")

    # Git
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True)
        print(f"  {result.stdout.strip()} — OK")
    except FileNotFoundError:
        print("  ERROR: git not found. Install from https://git-scm.com/")
        sys.exit(1)

    # GPU (informational)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                print(f"  GPU: {line.strip()}")
        else:
            print("  WARNING: nvidia-smi failed — CUDA may not be set up")
    except FileNotFoundError:
        print("  WARNING: nvidia-smi not found — CUDA drivers may not be installed")

    print()


def check_disk_space(base_dir):
    """Check available disk space."""
    try:
        target = os.path.abspath(base_dir)
        if platform.system() == "Windows":
            drive = os.path.splitdrive(target)[0]
            if not drive:
                drive = "C:"
            total, used, free = shutil.disk_usage(drive + "\\")
        else:
            # Use parent dir if base_dir doesn't exist yet
            check_path = target
            while not os.path.exists(check_path):
                check_path = os.path.dirname(check_path)
            total, used, free = shutil.disk_usage(check_path)

        free_gb = free / (1024**3)
        print(f"  Available disk space: {free_gb:.1f} GB")
        if free_gb < 80:
            print(f"  WARNING: At least 80 GB free recommended (40GB models + working space)")
            print(f"  You have {free_gb:.1f} GB. Downloads may fail if space runs out.")
        else:
            print(f"  Disk space OK (need ~80 GB, have {free_gb:.1f} GB)")
    except Exception as e:
        print(f"  Could not check disk space: {e}")

    print()


def setup_musubi_tuner(base_dir):
    """Clone or update musubi-tuner."""
    musubi_dir = os.path.join(base_dir, "musubi-tuner")

    print("[1/4] Setting up musubi-tuner...")
    if os.path.exists(os.path.join(musubi_dir, ".git")):
        print("  Already cloned — pulling latest...")
        subprocess.run(["git", "pull"], cwd=musubi_dir, check=True)
    else:
        print("  Cloning musubi-tuner...")
        os.makedirs(base_dir, exist_ok=True)
        subprocess.run([
            "git", "clone", "https://github.com/kohya-ss/musubi-tuner.git", musubi_dir
        ], check=True)

    print("  Done\n")
    return musubi_dir


def install_dependencies(musubi_dir):
    """Install Python dependencies."""
    print("[2/4] Installing Python dependencies...")
    print("  This may take a few minutes on first run.")

    # Install musubi-tuner requirements
    req_file = os.path.join(musubi_dir, "requirements.txt")
    if os.path.exists(req_file):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req_file], check=False)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", musubi_dir], check=False)

    # Additional dependencies needed for training
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers>=4.46.0", "huggingface_hub", "hf_transfer",
        "pydantic==1.10.13", "albumentations==1.4.3", "accelerate",
    ], check=True)

    print("  Done\n")


def scan_comfyui_models(comfyui_dir, models_dir):
    """
    Scan a ComfyUI installation for reusable model files.
    Symlinks (or copies on Windows if symlinks fail) compatible files into models_dir.
    Returns a set of dest_names that were found and linked.
    """
    print(f"  Scanning ComfyUI directory: {comfyui_dir}")
    found = set()

    # Map of dest_name -> list of (comfyui_subpath, is_compatible) to search
    search_map = {
        # VAE — same file everywhere
        "wan_2.1_vae.safetensors": [
            ("models/vae/wan_2.1_vae.safetensors", True),
        ],
        # T5 — safetensors format works (musubi-tuner supports both .pth and .safetensors)
        "models_t5_umt5-xxl-enc-bf16.pth": [
            ("models/text_encoders/umt5-xxl-enc-bf16.safetensors", True),
            ("models/text_encoders/umt5_xxl_fp16.safetensors", True),
        ],
        # DiTs — only fp16/bf16 versions work. fp8_scaled does NOT work.
        "wan2.2_t2v_high_noise_14B_fp16.safetensors": [
            ("models/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors", True),
            ("models/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", False),
        ],
        "wan2.2_t2v_low_noise_14B_fp16.safetensors": [
            ("models/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors", True),
            ("models/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", False),
        ],
    }

    for dest_name, candidates in search_map.items():
        dest_path = os.path.join(models_dir, dest_name)
        if os.path.exists(dest_path):
            continue  # Already have this file

        for subpath, compatible in candidates:
            src = os.path.join(comfyui_dir, subpath)
            if not os.path.exists(src):
                continue

            if not compatible:
                size_gb = os.path.getsize(src) / (1024**3)
                print(f"  SKIP: {os.path.basename(src)} ({size_gb:.1f} GB) — fp8 version, training needs fp16")
                continue

            size_gb = os.path.getsize(src) / (1024**3)
            # Use symlink if possible, copy as fallback
            try:
                os.symlink(src, dest_path)
                print(f"  LINKED: {dest_name} -> {src} ({size_gb:.1f} GB)")
            except OSError:
                shutil.copy2(src, dest_path)
                print(f"  COPIED: {dest_name} <- {src} ({size_gb:.1f} GB)")
            found.add(dest_name)
            break

    return found


def download_models(base_dir, hf_token=None, include_i2v=False, comfyui_dir=None):
    """Download model files from HuggingFace, reusing ComfyUI files where possible."""
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Scan ComfyUI for reusable models
    already_have = set()
    if comfyui_dir:
        already_have = scan_comfyui_models(comfyui_dir, models_dir)
        if already_have:
            print(f"  Reused {len(already_have)} model(s) from ComfyUI\n")
        else:
            print(f"  No compatible models found in ComfyUI (fp8 DiTs can't be used for training)\n")

    models_to_download = dict(MODELS)
    if include_i2v:
        models_to_download.update(I2V_MODELS)

    # Filter out models we already have
    remaining = {k: v for k, v in models_to_download.items()
                 if v["dest_name"] not in already_have
                 and not os.path.exists(os.path.join(models_dir, v["dest_name"]))}

    if not remaining:
        print(f"[3/4] All models already present in {models_dir}")
        print()
        return

    total_gb = sum(m["size_gb"] for m in remaining.values())
    print(f"[3/4] Downloading models to {models_dir}")
    print(f"  Download size: ~{total_gb:.0f} GB ({len(remaining)} file(s))")
    print(f"  Models: {', '.join(remaining.keys())}")
    if not include_i2v:
        print(f"  (I2V models skipped — use --include_i2v to download them too)")
    print()

    # Enable fast downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    try:
        from huggingface_hub import hf_hub_download
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
    except ImportError:
        print("  ERROR: huggingface_hub not installed.")
        print("  Run: pip install huggingface_hub hf_transfer")
        sys.exit(1)

    for name, info in models_to_download.items():
        dest = os.path.join(models_dir, info["dest_name"])

        if os.path.exists(dest):
            size_gb = os.path.getsize(dest) / (1024**3)
            print(f"  [CACHED] {info['dest_name']} ({size_gb:.1f} GB)")
            continue

        print(f"  [DOWNLOADING] {name} (~{info['size_gb']} GB)...")
        try:
            tmp_dir = os.path.join(models_dir, "hf_tmp")
            hf_hub_download(
                repo_id=info["repo_id"],
                filename=info["filename"],
                local_dir=tmp_dir,
            )
            # Move from HF's nested structure to flat models dir
            downloaded_path = os.path.join(tmp_dir, info["filename"])
            shutil.move(downloaded_path, dest)
            # Clean up temp dir
            shutil.rmtree(tmp_dir, ignore_errors=True)

            size_gb = os.path.getsize(dest) / (1024**3)
            print(f"  Done: {info['dest_name']} ({size_gb:.1f} GB)")
        except Exception as e:
            print(f"  ERROR downloading {name}: {e}")
            print(f"  You may need to set --hf_token or run: huggingface-cli login")

    print()


def create_directories(base_dir):
    """Create the directory structure."""
    print("[4/4] Creating directory structure...")

    dirs = [
        os.path.join(base_dir, "datasets"),
        os.path.join(base_dir, "outputs"),
        os.path.join(base_dir, "lightning_loras"),
        os.path.join(base_dir, "merged_dits"),
        os.path.join(base_dir, "resume_checkpoints"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  {d}")

    print("  Done\n")


def print_summary(base_dir):
    """Print setup summary and next steps."""
    models_dir = os.path.join(base_dir, "models")

    print("=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    print()
    print("  Directory structure:")
    print(f"    {base_dir}/")
    print(f"      musubi-tuner/        — training framework")
    print(f"      models/              — model weights")
    print(f"      datasets/            — put your dataset here")
    print(f"      outputs/             — training output")
    print(f"      lightning_loras/     — cached Lightning LoRAs")
    print(f"      merged_dits/         — cached merged models")
    print(f"      resume_checkpoints/  — for resuming training")
    print()

    # List downloaded models
    if os.path.exists(models_dir):
        print("  Downloaded models:")
        for f in sorted(os.listdir(models_dir)):
            fp = os.path.join(models_dir, f)
            if os.path.isfile(fp):
                size_gb = os.path.getsize(fp) / (1024**3)
                print(f"    {f} ({size_gb:.1f} GB)")
    print()
    print("  Next steps:")
    print(f"    1. Copy your dataset into {os.path.join(base_dir, 'datasets', 'your_character')}/")
    print(f"       (images/ and videos/ folders, each file with a matching .txt caption)")
    print(f"    2. Edit wan22-dataset-config-local.toml — update paths to your dataset")
    print(f"    3. Run training:")
    print(f"       python train_local_t2v_lightning.py --noise_level high")
    print(f"       python train_local_t2v_lightning.py --noise_level low")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Set up local environment for Wan 2.2 LoRA training",
    )
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR,
                        help=f"Base directory for all files (default: ./local_training/)")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--include_i2v", action="store_true",
                        help="Also download I2V model weights (~30GB extra)")
    parser.add_argument("--comfyui_dir", default=None,
                        help="Path to your ComfyUI installation. Reuses compatible models "
                             "(VAE, T5) to save download time. NOTE: ComfyUI's fp8_scaled DiTs "
                             "can't be used for training — fp16 DiTs are still downloaded.")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    comfyui_dir = os.path.abspath(args.comfyui_dir) if args.comfyui_dir else None

    print("=" * 60)
    print("  Wan 2.2 LoRA Training — Local Setup")
    print("=" * 60)
    print(f"  Base directory: {base_dir}")
    if comfyui_dir:
        print(f"  ComfyUI directory: {comfyui_dir}")
    print()

    check_prerequisites()
    check_disk_space(base_dir)

    musubi_dir = setup_musubi_tuner(base_dir)
    install_dependencies(musubi_dir)
    download_models(base_dir, hf_token=hf_token, include_i2v=args.include_i2v,
                    comfyui_dir=comfyui_dir)
    create_directories(base_dir)
    print_summary(base_dir)


if __name__ == "__main__":
    main()
