"""
Wan 2.2 I2V 14B LoRA Training — Lightning-Optimized
=====================================================
Merges a Lightning (speed) LoRA into the base DiT weights BEFORE training,
so the character LoRA is optimized against the Lightning-modified model.

This produces a character LoRA that's tuned for the Lightning denoising path,
not the vanilla one. The tradeoff: this LoRA is coupled to the specific
Lightning LoRA you merge — it won't perform the same without Lightning loaded.

Usage:
  1. Set LIGHTNING_LORA config below (HF repo + filename for each expert)
  2. Upload dataset:  python -m modal run train_wan22_i2v_lightning.py::add_to_volume_datasets
  3. Train high:      python -m modal run train_wan22_i2v_lightning.py::run_high
  4. Train low:       python -m modal run train_wan22_i2v_lightning.py::run_low

Resume from checkpoint:
  1. Upload a checkpoint:
     python -m modal run train_wan22_i2v_lightning.py::upload_resume_checkpoint --checkpoint path/to/lora.safetensors
  2. Train as normal — the script auto-detects the checkpoint:
     python -m modal run train_wan22_i2v_lightning.py::run_high
  3. Verify it's on the volume:
     python -m modal volume ls kohya-volume /resume_checkpoints/
  4. To clear (start fresh):
     python -m modal volume rm kohya-volume -r /resume_checkpoints/

At inference: load Lightning LoRA + your character LoRA together in ComfyUI.
"""

import os
import datetime
from pathlib import Path

from modal import (
    App,
    Image,
    Secret,
    Volume,
)

# =============================================================================
# ██████  CONFIG  ██████
# =============================================================================

OUTPUT_NAME = "annika-wan22-i2v-lightning"

# --- Lightning LoRA to merge into base DiT ---
# Set these to the HF repo and filename for your Lightning LoRA.
# High-noise and low-noise experts need their own Lightning LoRAs.
# Set to None to skip merge for that expert (trains against vanilla DiT).
LIGHTNING_LORA = {
    "high": {
        "repo_id": "lightx2v/Wan2.2-Lightning",
        "filename": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
    },
    "low": {
        "repo_id": "lightx2v/Wan2.2-Lightning",
        "filename": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
    },
}
LIGHTNING_MERGE_STRENGTH = 1.0    # 1.0 = full merge, <1.0 = partial (try 0.8 if full is too aggressive)

# --- Shared Hyperparameters ---
LR_SCHEDULER        = "polynomial"
LR_SCHEDULER_POWER  = "2"
MIN_LR_RATIO        = "0.01"
OPTIMIZER           = "adamw8bit"
SEED                = "42"
DISCRETE_FLOW_SHIFT = "5.0"      # I2V standard

# --- Per-Expert Hyperparameters ---
# High and low noise experts can have different training configs.
# The high-noise expert handles the initial denoising (coarse structure),
# while the low-noise expert handles refinement (fine detail).
EXPERT_CONFIG = {
    "high": {
        "learning_rate": "1e-4",
        "network_dim":   "8",
        "network_alpha": "8",
        "max_epochs":    "50",
        "save_every":    "5",
    },
    "low": {
        "learning_rate": "8e-5",
        "network_dim":   "42",
        "network_alpha": "42",
        "max_epochs":    "50",
        "save_every":    "5",
    },
}

# =============================================================================
# Modal setup
# =============================================================================

app = App(name="wan22-i2v-lightning-low")
dataset_volume = Volume.from_name("datasets", create_if_missing=True)


@app.local_entrypoint()
def add_to_volume_datasets():
    import pathlib
    # Use the directory where this script lives, not the CWD
    script_dir = pathlib.Path(__file__).parent.resolve()
    datasets_path = script_dir / "datasets"
    
    if not datasets_path.is_dir():
        print(f"ERROR: datasets folder not found at {datasets_path}")
        print(f"Script directory: {script_dir}")
        return "Failed"
    
    print(f"Uploading from: {datasets_path}")
    with dataset_volume.batch_upload(force=True) as batch:
        batch.put_directory(str(datasets_path), "/datasets")
    return "Success"


# --- Resume Checkpoint Upload ---
# Upload a .safetensors LoRA checkpoint to the Modal volume so training
# can resume from it. The checkpoint goes into /outputs/resume_checkpoints/
# and the train() function auto-detects the latest file there.
#
# Usage:
#   python -m modal run train_wan22_i2v_lightning.py::upload_resume_checkpoint --checkpoint path/to/lora.safetensors
#
@app.local_entrypoint()
def upload_resume_checkpoint(checkpoint: str):
    """Upload a LoRA .safetensors checkpoint for resume training."""
    import pathlib
    ckpt_path = pathlib.Path(checkpoint).resolve()
    
    if not ckpt_path.exists():
        print(f"ERROR: File not found: {ckpt_path}")
        return "Failed"
    if not ckpt_path.suffix == ".safetensors":
        print(f"WARNING: Expected .safetensors file, got: {ckpt_path.suffix}")
        print(f"Uploading anyway...")
    
    dest = f"/resume_checkpoints/{ckpt_path.name}"
    print(f"Uploading checkpoint to Modal volume:")
    print(f"  Local:  {ckpt_path}")
    print(f"  Remote: kohya-volume:{dest}")
    print(f"  Size:   {ckpt_path.stat().st_size / (1024*1024):.1f} MB")
    
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(ckpt_path), dest)
    
    print(f"✓ Uploaded! Train will auto-detect this checkpoint.")
    print(f"  To verify: python -m modal volume ls kohya-volume /resume_checkpoints/")
    return "Success"


image = (
    Image.debian_slim(python_version="3.10")
    .env({"HF_HUB_CACHE": "/cache/cache/"})
    .apt_install("git")
    .apt_install("ffmpeg")
    .apt_install("python3-opencv")
    .pip_install("transformers>=4.46.0")
    .pip_install("huggingface_hub")
    .run_commands("git clone https://github.com/kohya-ss/musubi-tuner.git")
    .workdir("musubi-tuner")
    .run_commands(
        "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
        "pip3 install -r requirements.txt || pip3 install -e .",
    )
    .pip_install("pydantic==1.10.13", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install("albumentations==1.4.3")
    .add_local_file(
        "wan21-dataset-config.toml",
        "/dataset-config.toml"
    )
)

volume = Volume.from_name("kohya-volume", create_if_missing=True)
cache_vol = Volume.from_name("hf-hub-cache", create_if_missing=True)
OUTPUT_DIR = "/outputs"


# =============================================================================
# LoRA → Base Model Merge Logic
# =============================================================================
# Merges LoRA delta weights directly into the DiT safetensors so the
# character LoRA trains against the Lightning-modified denoising path.
#
# Formula per layer: W_merged = W_base + strength * (alpha/rank) * (up @ down)
#
# Runs entirely on CPU to avoid OOM — the 14B DiT is ~28GB in fp16.
# =============================================================================

MERGE_SCRIPT = r'''
import sys
import os
import json
import torch
from safetensors.torch import load_file, save_file

def merge_lora_into_dit(dit_path, lora_path, output_path, strength=1.0):
    """
    Merge a LoRA's learned deltas into the base DiT weights.

    Key mapping strategy:
      LoRA keys follow patterns like:
        lora_unet_<module_path_with_dots_as_separators>.lora_down.weight
        lora_unet_<module_path_with_dots_as_separators>.lora_up.weight
        lora_unet_<module_path_with_dots_as_separators>.alpha

      The base model key is derived by stripping the lora_ prefix/suffix
      and reconstructing the dotted path + .weight

    Handles multiple naming conventions (kohya, musubi-tuner, ComfyUI format).
    """
    print(f"Loading base DiT: {dit_path}")
    print(f"  (this is ~28GB for 14B fp16 — loading on CPU)")
    base_sd = load_file(dit_path, device="cpu")
    print(f"  Loaded {len(base_sd)} base model keys")
    
    # === DIAGNOSTIC: Show sample keys from both sides ===
    base_key_list = sorted(base_sd.keys())
    print(f"\n  DIAGNOSTIC — First 10 base model keys:")
    for k in base_key_list[:10]:
        print(f"    {k}")
    print(f"  DIAGNOSTIC — Last 5 base model keys:")
    for k in base_key_list[-5:]:
        print(f"    {k}")

    print(f"Loading Lightning LoRA: {lora_path}")
    lora_sd = load_file(lora_path, device="cpu")
    print(f"  Loaded {len(lora_sd)} LoRA keys")
    
    lora_key_list = sorted(lora_sd.keys())
    print(f"\n  DIAGNOSTIC — First 10 LoRA keys:")
    for k in lora_key_list[:10]:
        print(f"    {k}")
    # Show a sample down key specifically
    down_keys = [k for k in lora_key_list if ".lora_down.weight" in k]
    if down_keys:
        print(f"\n  DIAGNOSTIC — Sample LoRA down key: {down_keys[0]}")
        module = down_keys[0].replace(".lora_down.weight", "")
        print(f"  DIAGNOSTIC — Derived module path: {module}")
        print(f"  DIAGNOSTIC — Looking for: {module}.weight")
        stripped = module.replace("diffusion_model.", "", 1) if module.startswith("diffusion_model.") else module
        print(f"  DIAGNOSTIC — Stripped to: {stripped}.weight")
        print(f"  DIAGNOSTIC — Is '{stripped}.weight' in base? {f'{stripped}.weight' in base_sd}")
        print(f"  DIAGNOSTIC — Is '{module}.weight' in base? {f'{module}.weight' in base_sd}")

    # --- Build set of base model keys for matching ---
    base_keys = set(base_sd.keys())

    # --- Find all LoRA down/up pairs ---
    lora_pairs = {}  # base_key -> {"down": tensor, "up": tensor, "alpha": float}
    unmapped_keys = []

    for key in lora_sd:
        if ".lora_down.weight" not in key:
            continue

        down_key = key
        up_key = key.replace(".lora_down.weight", ".lora_up.weight")
        alpha_key = key.replace(".lora_down.weight", ".alpha")

        if up_key not in lora_sd:
            print(f"  WARNING: Found down without up: {key}")
            continue

        # --- Derive the base model key ---
        # Strip LoRA-specific suffixes first
        module_path = key.replace(".lora_down.weight", "")

        # Try multiple prefix variations — ComfyUI LoRAs use "diffusion_model."
        # but the Comfy-Org repackaged DiTs often store keys without that prefix
        candidates = []
        
        # As-is with .weight
        candidates.append(f"{module_path}.weight")
        # As-is without .weight
        candidates.append(module_path)
        
        # Strip common prefixes and try again
        for prefix in ["diffusion_model.", "lora_unet_", "lora_te_"]:
            if module_path.startswith(prefix):
                stripped = module_path[len(prefix):]
                candidates.append(f"{stripped}.weight")
                candidates.append(stripped)
        
        # Try adding diffusion_model. prefix (reverse case)
        if not module_path.startswith("diffusion_model."):
            candidates.append(f"diffusion_model.{module_path}.weight")
            candidates.append(f"diffusion_model.{module_path}")

        base_key = None
        for candidate in candidates:
            if candidate in base_keys:
                base_key = candidate
                break
        
        if base_key is None:
            unmapped_keys.append(key)
            if len(unmapped_keys) <= 3:  # Only show first 3 in detail
                print(f"  WARNING: Cannot map LoRA key: {key}")
                print(f"    All candidates tried: {candidates}")
            continue

        alpha = lora_sd[alpha_key].item() if alpha_key in lora_sd else None
        lora_pairs[base_key] = {
            "down": lora_sd[down_key],
            "up": lora_sd[up_key],
            "alpha": alpha,
        }

    print(f"\n  Found {len(lora_pairs)} LoRA layer pairs to merge")
    if unmapped_keys:
        print(f"  WARNING: {len(unmapped_keys)} LoRA keys could not be mapped to base model")
    if len(lora_pairs) == 0:
        print("  ERROR: No LoRA pairs could be mapped to base model keys!")
        print("  This likely means the LoRA uses a different key naming format.")
        print("  Aborting merge — training will use vanilla DiT.")
        return False

    # --- Apply deltas ---
    merged_count = 0
    for base_key, pair in lora_pairs.items():
        down = pair["down"].float()   # [rank, in_features] or [rank, in, k, k, k]
        up = pair["up"].float()       # [out_features, rank] or [out, rank, k, k, k]
        alpha = pair["alpha"]
        rank = down.shape[0]

        # Compute scale factor
        if alpha is not None:
            scale = strength * (alpha / rank)
        else:
            # No alpha stored — assume alpha == rank (scale factor = 1.0 * strength)
            scale = strength

        # Compute delta based on tensor dimensions
        base_weight = base_sd[base_key].float()

        if down.dim() == 2 and up.dim() == 2:
            # Standard linear: delta = up @ down
            delta = (up @ down) * scale
        elif down.dim() == 5 and up.dim() == 5:
            # Conv3d: need to handle kernel dimensions
            # down: [rank, in_c, kT, kH, kW], up: [out_c, rank, 1, 1, 1] (typical)
            # or both could have full kernel dims
            if up.shape[2:] == (1, 1, 1):
                # up is pointwise — reshape to 2D, multiply, reshape back
                down_2d = down.reshape(rank, -1)            # [rank, in_c * kT * kH * kW]
                up_2d = up.reshape(up.shape[0], rank)       # [out_c, rank]
                delta_2d = (up_2d @ down_2d) * scale        # [out_c, in_c * kT * kH * kW]
                delta = delta_2d.reshape(base_weight.shape)
            else:
                # Both have full kernels — more complex, use einsum
                # This is rare but handle it
                down_2d = down.reshape(rank, -1)
                up_2d = up.reshape(up.shape[0], -1)
                # This won't work for non-pointwise up — fall back to skip
                print(f"  WARNING: Complex Conv3d LoRA shape for {base_key}, skipping")
                continue
        elif down.dim() == 3 and up.dim() == 3:
            # Conv1d or similar: down [rank, in, k], up [out, rank, 1]
            down_2d = down.reshape(rank, -1)
            up_2d = up.reshape(up.shape[0], rank)
            delta_2d = (up_2d @ down_2d) * scale
            delta = delta_2d.reshape(base_weight.shape)
        else:
            print(f"  WARNING: Unexpected tensor dims for {base_key}: "
                  f"down={list(down.shape)}, up={list(up.shape)}. Skipping.")
            continue

        # Apply delta in the base weight's dtype
        base_sd[base_key] = (base_weight + delta).to(base_sd[base_key].dtype)
        merged_count += 1

    print(f"  Merged {merged_count}/{len(lora_pairs)} layers (strength={strength})")

    # --- Save merged model ---
    print(f"  Saving merged DiT to: {output_path}")
    save_file(base_sd, output_path)
    merged_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"  Saved ({merged_size_gb:.1f} GB)")

    # Free memory
    del base_sd, lora_sd
    return True


if __name__ == "__main__":
    dit_path = sys.argv[1]
    lora_path = sys.argv[2]
    output_path = sys.argv[3]
    strength = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    success = merge_lora_into_dit(dit_path, lora_path, output_path, strength)
    sys.exit(0 if success else 1)
'''


# =============================================================================
# Training
# =============================================================================

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={OUTPUT_DIR: volume, "/cache": cache_vol, "/datasets": dataset_volume},
    timeout=60 * 60 * 24,
    secrets=[Secret.from_name("my-huggingface-secret")],
)
def train(noise_level: str):
    import subprocess
    import os
    import glob
    import threading
    import huggingface_hub

    # === Quick check: dataset volume has data ===
    import itertools
    img_count = len(list(itertools.chain(
        Path("/datasets").rglob("*.png"),
        Path("/datasets").rglob("*.webp"),
        Path("/datasets").rglob("*.jpg"),
    )))
    vid_count = len(list(Path("/datasets").rglob("*.mp4")))
    print(f"Dataset volume: {img_count} images, {vid_count} videos")
    if img_count == 0 and vid_count == 0:
        print("WARNING: No data found! Run add_to_volume_datasets first.")

    is_high = noise_level == "high"
    label = "HIGH-NOISE" if is_high else "LOW-NOISE"
    lightning_cfg = LIGHTNING_LORA.get(noise_level, {})

    # --- Per-expert hyperparameters ---
    expert = EXPERT_CONFIG[noise_level]
    LEARNING_RATE = expert["learning_rate"]
    NETWORK_DIM   = expert["network_dim"]
    NETWORK_ALPHA = expert["network_alpha"]
    MAX_EPOCHS    = expert["max_epochs"]
    SAVE_EVERY    = expert["save_every"]

    # ----------------------------------------------------------
    # Background thread: commits the volume every 5 minutes
    # ----------------------------------------------------------
    _stop_committer = threading.Event()

    def _periodic_commit(interval_seconds=300):
        while not _stop_committer.is_set():
            _stop_committer.wait(interval_seconds)
            if _stop_committer.is_set():
                break
            try:
                n_files = len(glob.glob(os.path.join(OUTPUT_DIR, "**/*.safetensors"), recursive=True))
                if n_files > 0:
                    volume.commit()
                    print(f"[auto-commit] Volume committed ({n_files} safetensors on disk)")
            except Exception as e:
                print(f"[auto-commit] WARNING: commit failed: {e}")

    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()

    # Timestamped output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{OUTPUT_NAME}-{noise_level}-{timestamp}"
    run_output_dir = f"{OUTPUT_DIR}/{OUTPUT_NAME}"
    os.makedirs(run_output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Wan 2.2 I2V 14B — {label} Expert (Lightning-Optimized)")
    print(f"  Output: {name}")
    print(f"  Dir:    {run_output_dir}")
    print(f"  LR: {LEARNING_RATE} | Scheduler: {LR_SCHEDULER}")
    print(f"  Dim: {NETWORK_DIM} | Alpha: {NETWORK_ALPHA}")
    print(f"  Flow Shift: {DISCRETE_FLOW_SHIFT}")
    print(f"  Lightning merge strength: {LIGHTNING_MERGE_STRENGTH}")
    print(f"  Resume dir: {OUTPUT_DIR}/resume_checkpoints/")
    print("=" * 60)

    # Detect repo structure
    if os.path.exists("src/musubi_tuner/wan_train_network.py"):
        SCRIPT_PREFIX = "src/musubi_tuner/"
        print("Detected new musubi-tuner repo structure")
    else:
        SCRIPT_PREFIX = ""
        print("Detected classic musubi-tuner repo structure")

    # =================================================================
    # Resume from checkpoint (optional)
    # =================================================================
    # Checks /outputs/resume_checkpoints/ for a .safetensors file.
    # Upload one via:  python -m modal run <script>::upload_resume_checkpoint --checkpoint path/to/file
    # Or it can also pick up checkpoints from a previous training run's output dir.
    #
    # Uses --network_weights to load LoRA weights WITHOUT optimizer state.
    # This avoids the known musubi-tuner bug where LR scheduling breaks
    # on full resume (--resume), causing a sustained loss spike.
    # =================================================================
    resume_weights = None
    resume_dir = f"{OUTPUT_DIR}/resume_checkpoints"
    os.makedirs(resume_dir, exist_ok=True)

    import glob as _glob
    candidates = sorted(
        _glob.glob(f"{resume_dir}/*.safetensors"),
        key=os.path.getmtime,
        reverse=True,  # newest first
    )
    if candidates:
        resume_weights = candidates[0]
        size_mb = os.path.getsize(resume_weights) / (1024 * 1024)
        print(f"\n  RESUME: Found checkpoint in {resume_dir}/:")
        print(f"    {resume_weights}")
        print(f"    File size: {size_mb:.1f} MB")
        if len(candidates) > 1:
            print(f"    ({len(candidates)} files found — using newest by modification time)")
        print(f"    NOTE: Loading weights only (no optimizer state) — LR schedule starts fresh.")
    else:
        print(f"\n  No resume checkpoint found. Starting from scratch.")
        print(f"  (To resume: python -m modal run <script>::upload_resume_checkpoint --checkpoint path/to/file)")

    # -----------------------------------------------------------------
    # Download models (cached on hf-hub-cache volume between runs)
    # -----------------------------------------------------------------
    # Try local cache first to avoid network round-trips.
    # If not cached yet, download and commit the volume so next run is instant.
    comfy_22_repo = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
    comfy_21_repo = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
    wan_repo = "Wan-AI/Wan2.1-I2V-14B-720P"

    dit_file = (
        "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
        if is_high else
        "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
    )

    def cached_download(repo_id, filename):
        """Try local cache first, then download. Returns local path."""
        try:
            path = huggingface_hub.hf_hub_download(
                repo_id=repo_id, filename=filename, local_files_only=True
            )
            print(f"  [CACHED] {repo_id}/{filename}")
            return path
        except Exception:
            print(f"  [DOWNLOADING] {repo_id}/{filename} ...")
            path = huggingface_hub.hf_hub_download(
                repo_id=repo_id, filename=filename, local_files_only=False
            )
            return path

    try:
        needs_commit = False

        # Check if DiT is already cached
        try:
            huggingface_hub.hf_hub_download(repo_id=comfy_22_repo, filename=dit_file, local_files_only=True)
        except Exception:
            needs_commit = True  # Will need to commit after downloading

        dit_path = cached_download(comfy_22_repo, dit_file)
        vae_path = cached_download(comfy_21_repo, "split_files/vae/wan_2.1_vae.safetensors")
        t5_path = cached_download(wan_repo, "models_t5_umt5-xxl-enc-bf16.pth")
        
        print(f"DiT ({label}): {dit_path}")
        print(f"VAE: {vae_path}")
        print(f"T5:  {t5_path}")

        # Commit cache volume after downloads so next run is instant
        if needs_commit:
            print("  Committing model cache to volume (first run only)...")
            cache_vol.commit()
            print("  Cache volume committed — future runs will skip downloads.")

    except Exception as e:
        print(f"Error downloading models: {e}")
        _stop_committer.set()
        return

    # =================================================================
    # Lightning LoRA Merge
    # Downloads the Lightning LoRA and bakes it into the DiT weights.
    # Training then runs against the merged model, so the character
    # LoRA learns to complement Lightning's modified denoising path.
    # =================================================================
    has_lightning = (
        lightning_cfg.get("repo_id")
        and lightning_cfg.get("filename")
        and "INSERT" not in lightning_cfg.get("repo_id", "INSERT")
    )

    if has_lightning:
        print(f"\n{'='*60}")
        print(f"  LIGHTNING MERGE — {label} expert")
        print(f"  Repo: {lightning_cfg['repo_id']}")
        print(f"  File: {lightning_cfg['filename']}")
        print(f"  Strength: {LIGHTNING_MERGE_STRENGTH}")
        print(f"{'='*60}")

        # Check if we already have a merged DiT cached on the volume
        # Uses a deterministic name so repeat runs skip the merge entirely
        merged_cache_dir = "/cache/merged_dits"
        os.makedirs(merged_cache_dir, exist_ok=True)
        merge_tag = f"wan22_i2v_{noise_level}_lightning_s{LIGHTNING_MERGE_STRENGTH}"
        merged_dit_path = f"{merged_cache_dir}/{merge_tag}.safetensors"

        if os.path.exists(merged_dit_path):
            print(f"  [CACHED] Merged DiT already exists: {merged_dit_path}")
            print(f"  Skipping merge — using cached version.")
            dit_path = merged_dit_path
        else:
            # Need to merge — download Lightning LoRA first
            try:
                lightning_lora_path = cached_download(
                    lightning_cfg["repo_id"],
                    lightning_cfg["filename"],
                )
            except Exception as e:
                print(f"  ERROR downloading Lightning LoRA: {e}")
                print(f"  Falling back to vanilla DiT — training will proceed without merge.")
                lightning_lora_path = None

            if lightning_lora_path:
                # Write merge script to temp
                with open("/tmp/merge_lora.py", "w") as f:
                    f.write(MERGE_SCRIPT)

                result = subprocess.run(
                    [
                        "python", "/tmp/merge_lora.py",
                        dit_path,
                        lightning_lora_path,
                        merged_dit_path,
                        str(LIGHTNING_MERGE_STRENGTH),
                    ],
                    capture_output=False,
                )

                if result.returncode == 0 and os.path.exists(merged_dit_path):
                    print(f"  Merge successful. Caching to volume for future runs...")
                    cache_vol.commit()
                    print(f"  Cached. Training against merged DiT.")
                    dit_path = merged_dit_path
                else:
                    print(f"  WARNING: Merge failed (exit code {result.returncode}).")
                    print(f"  Training will proceed against vanilla DiT.")
    else:
        print(f"\n  Lightning LoRA not configured for {label} expert.")
        print(f"  Set LIGHTNING_LORA['{noise_level}'] in script config.")
        print(f"  Training against vanilla DiT.\n")

    # -----------------------------------------------------------------
    # Step 1: Cache latents (uses vanilla VAE — not affected by merge)
    # -----------------------------------------------------------------
    print("\nStep 1: Caching latents...")
    subprocess.run([
        "python",
        f"{SCRIPT_PREFIX}wan_cache_latents.py",
        "--dataset_config", "/dataset-config.toml",
        "--vae", vae_path,
        "--vae_cache_cpu",
        "--i2v",
    ], check=True)

    # -----------------------------------------------------------------
    # Step 2: Cache text encoder outputs
    # -----------------------------------------------------------------
    print("\nStep 2: Caching text encoder outputs...")
    subprocess.run([
        "python",
        f"{SCRIPT_PREFIX}wan_cache_text_encoder_outputs.py",
        "--dataset_config", "/dataset-config.toml",
        "--t5", t5_path,
        "--batch_size", "16",
        "--fp8_t5",
    ], check=True)

    # -----------------------------------------------------------------
    # Step 3: Train against (possibly merged) DiT
    # -----------------------------------------------------------------
    print(f"\nStep 3: Training {label} expert...")
    print(f"  DiT path: {dit_path}")
    if "/cache/merged_dits/" in dit_path:
        print(f"  ** Training against Lightning-merged weights **")

    # I2V boundary: 900, not 875
    if is_high:
        min_ts, max_ts = "900", "1000"
    else:
        min_ts, max_ts = "0", "900"

    train_cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "fp16",
        f"{SCRIPT_PREFIX}wan_train_network.py",
        "--task", "i2v-A14B",
        "--dit", dit_path,              # <-- Points to merged DiT if merge succeeded
        "--vae", vae_path,
        "--t5", t5_path,
        "--dataset_config", "/dataset-config.toml",
        "--sdpa",
        "--mixed_precision", "fp16",
        "--fp8_base",
        "--fp8_scaled",                 # Valid for Wan 2.2
        "--vae_cache_cpu",
        # --- Wan 2.2 I2V specific ---
        "--min_timestep", min_ts,
        "--max_timestep", max_ts,
        "--preserve_distribution_shape",
        # --- Optimizer ---
        "--optimizer_type", OPTIMIZER,
        "--optimizer_args", "weight_decay=0.01",
        # --- Learning Rate ---
        "--learning_rate", LEARNING_RATE,
        # --- LR Scheduler ---
        "--lr_scheduler", LR_SCHEDULER,
        "--lr_scheduler_min_lr_ratio", MIN_LR_RATIO,
        "--lr_scheduler_power", LR_SCHEDULER_POWER,
        # --- Memory ---
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        # --- LoRA Config ---
        "--network_module", "networks.lora_wan",
        "--network_dim", NETWORK_DIM,
        "--network_alpha", NETWORK_ALPHA,
        # --- Timestep / Flow ---
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", DISCRETE_FLOW_SHIFT,
        # --- Training Duration ---
        "--max_train_epochs", MAX_EPOCHS,
        "--save_every_n_epochs", SAVE_EVERY,
        # --- Output ---
        "--seed", SEED,
        "--output_dir", run_output_dir,
        "--output_name", name,
        # --- Logging ---
        "--log_with", "tensorboard",
        "--logging_dir", f"{run_output_dir}/logs",
    ]

    # --- Resume: inject --network_weights if we have a checkpoint ---
    if resume_weights:
        train_cmd += ["--network_weights", resume_weights]
        print(f"\n  >> --network_weights {resume_weights}")

    print(f"Training command:\n{' '.join(train_cmd)}")
    result = subprocess.run(train_cmd)
    print(f"\nTraining exit code: {result.returncode}")

    if result.returncode != 0:
        print(f"ERROR: Training failed with exit code {result.returncode}")

    # ----------------------------------------------------------
    # Stop auto-commit and final commit
    # ----------------------------------------------------------
    _stop_committer.set()

    print(f"\nTraining complete. Contents of {run_output_dir}:")
    for root, dirs, files in os.walk(run_output_dir):
        level = root.replace(run_output_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            filepath = os.path.join(root, f)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"{subindent}{f} ({size_mb:.1f} MB)")

    volume.commit()
    print("Volume committed successfully!")


@app.local_entrypoint()
def run_high():
    # .spawn() dispatches to Modal's cloud and returns immediately.
    # Training continues even if your terminal/PC disconnects.
    # Monitor progress at https://modal.com/apps — look for the function logs.
    call = train.spawn("high")
    print(f"Training dispatched! Function call ID: {call.object_id}")
    print(f"Monitor at: https://modal.com/apps")
    print(f"Your terminal is free — training runs independently in Modal's cloud.")


@app.local_entrypoint()
def run_low():
    call = train.spawn("low")
    print(f"Training dispatched! Function call ID: {call.object_id}")
    print(f"Monitor at: https://modal.com/apps")
    print(f"Your terminal is free — training runs independently in Modal's cloud.")
