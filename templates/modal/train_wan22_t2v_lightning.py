"""
Wan 2.2 T2V A14B LoRA Training — Lightning-Optimized
=====================================================
Merges a Lightning (speed) LoRA into the base DiT weights BEFORE training,
so the character LoRA is optimized against the Lightning-modified model.

Usage:
  1. Set LIGHTNING_LORA config below (HF repo + filename for each expert)
  2. Upload dataset:  python -m modal run train_wan22_t2v_lightning.py::add_to_volume_datasets
  3. Train high:      python -m modal run train_wan22_t2v_lightning.py::run_high
  4. Train low:       python -m modal run train_wan22_t2v_lightning.py::run_low

Resume from checkpoint:
  1. Upload a checkpoint:
     python -m modal run train_wan22_t2v_lightning.py::upload_resume_checkpoint --checkpoint path/to/lora.safetensors
  2. Train as normal — the script auto-detects the checkpoint:
     python -m modal run train_wan22_t2v_lightning.py::run_high

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

OUTPUT_NAME = "annika-wan22-t2v-lightning"

LIGHTNING_LORA = {
    "high": {
        "repo_id": "lightx2v/Wan2.2-Lightning",
        "filename": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
    },
    "low": {
        "repo_id": "lightx2v/Wan2.2-Lightning",
        "filename": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
    },
}
LIGHTNING_MERGE_STRENGTH = 1.0

# --- Shared Hyperparameters ---
LR_SCHEDULER        = "polynomial"
LR_SCHEDULER_POWER  = "2"
MIN_LR_RATIO        = "0.01"
OPTIMIZER           = "adamw8bit"
SEED                = "42"
DISCRETE_FLOW_SHIFT = "3.0"      # T2V standard

# --- Per-Expert Hyperparameters ---
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

app = App(name="wan22-t2v-lightning")
dataset_volume = Volume.from_name("datasets", create_if_missing=True)


@app.local_entrypoint()
def add_to_volume_datasets():
    import pathlib
    script_dir = pathlib.Path(__file__).parent.resolve()
    datasets_path = script_dir / "datasets"

    if not datasets_path.is_dir():
        print(f"ERROR: datasets folder not found at {datasets_path}")
        return "Failed"

    print(f"Uploading from: {datasets_path}")
    with dataset_volume.batch_upload(force=True) as batch:
        batch.put_directory(str(datasets_path), "/datasets")
    return "Success"


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

    dest = f"/resume_checkpoints/{ckpt_path.name}"
    print(f"Uploading checkpoint to Modal volume:")
    print(f"  Local:  {ckpt_path}")
    print(f"  Remote: kohya-volume:{dest}")
    print(f"  Size:   {ckpt_path.stat().st_size / (1024*1024):.1f} MB")

    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(ckpt_path), dest)

    print(f"Uploaded! Train will auto-detect this checkpoint.")
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

MERGE_SCRIPT = r'''
import sys
import os
import torch
from safetensors.torch import load_file, save_file

def merge_lora_into_dit(dit_path, lora_path, output_path, strength=1.0):
    print(f"Loading base DiT: {dit_path}")
    print(f"  (this is ~28GB for 14B fp16 — loading on CPU)")
    base_sd = load_file(dit_path, device="cpu")
    print(f"  Loaded {len(base_sd)} base model keys")

    base_key_list = sorted(base_sd.keys())
    print(f"\n  First 10 base model keys:")
    for k in base_key_list[:10]:
        print(f"    {k}")

    print(f"Loading Lightning LoRA: {lora_path}")
    lora_sd = load_file(lora_path, device="cpu")
    print(f"  Loaded {len(lora_sd)} LoRA keys")

    down_keys = [k for k in sorted(lora_sd.keys()) if ".lora_down.weight" in k]
    if down_keys:
        print(f"\n  Sample LoRA down key: {down_keys[0]}")

    base_keys = set(base_sd.keys())
    lora_pairs = {}
    unmapped_keys = []

    for key in lora_sd:
        if ".lora_down.weight" not in key:
            continue

        down_key = key
        up_key = key.replace(".lora_down.weight", ".lora_up.weight")
        alpha_key = key.replace(".lora_down.weight", ".alpha")

        if up_key not in lora_sd:
            continue

        module_path = key.replace(".lora_down.weight", "")
        candidates = [f"{module_path}.weight", module_path]
        for prefix in ["diffusion_model.", "lora_unet_", "lora_te_"]:
            if module_path.startswith(prefix):
                stripped = module_path[len(prefix):]
                candidates.extend([f"{stripped}.weight", stripped])
        if not module_path.startswith("diffusion_model."):
            candidates.extend([f"diffusion_model.{module_path}.weight", f"diffusion_model.{module_path}"])

        base_key = None
        for candidate in candidates:
            if candidate in base_keys:
                base_key = candidate
                break

        if base_key is None:
            unmapped_keys.append(key)
            if len(unmapped_keys) <= 3:
                print(f"  WARNING: Cannot map: {key}")
            continue

        alpha = lora_sd[alpha_key].item() if alpha_key in lora_sd else None
        lora_pairs[base_key] = {"down": lora_sd[down_key], "up": lora_sd[up_key], "alpha": alpha}

    print(f"\n  Found {len(lora_pairs)} LoRA layer pairs to merge")
    if unmapped_keys:
        print(f"  WARNING: {len(unmapped_keys)} keys could not be mapped")
    if len(lora_pairs) == 0:
        print("  ERROR: No LoRA pairs mapped — aborting merge")
        return False

    merged_count = 0
    for base_key, pair in lora_pairs.items():
        down = pair["down"].float()
        up = pair["up"].float()
        alpha = pair["alpha"]
        rank = down.shape[0]
        scale = strength * (alpha / rank) if alpha is not None else strength
        base_weight = base_sd[base_key].float()

        if down.dim() == 2 and up.dim() == 2:
            delta = (up @ down) * scale
        elif down.dim() == 5 and up.dim() == 5:
            if up.shape[2:] == (1, 1, 1):
                down_2d = down.reshape(rank, -1)
                up_2d = up.reshape(up.shape[0], rank)
                delta = (up_2d @ down_2d).reshape(base_weight.shape) * scale
            else:
                continue
        elif down.dim() == 3 and up.dim() == 3:
            down_2d = down.reshape(rank, -1)
            up_2d = up.reshape(up.shape[0], rank)
            delta = (up_2d @ down_2d).reshape(base_weight.shape) * scale
        else:
            continue

        base_sd[base_key] = (base_weight + delta).to(base_sd[base_key].dtype)
        merged_count += 1

    print(f"  Merged {merged_count}/{len(lora_pairs)} layers (strength={strength})")
    print(f"  Saving merged DiT to: {output_path}")
    save_file(base_sd, output_path)
    merged_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"  Saved ({merged_size_gb:.1f} GB)")
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

    expert = EXPERT_CONFIG[noise_level]
    LEARNING_RATE = expert["learning_rate"]
    NETWORK_DIM   = expert["network_dim"]
    NETWORK_ALPHA = expert["network_alpha"]
    MAX_EPOCHS    = expert["max_epochs"]
    SAVE_EVERY    = expert["save_every"]

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

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{OUTPUT_NAME}-{noise_level}-{timestamp}"
    run_output_dir = f"{OUTPUT_DIR}/{OUTPUT_NAME}"
    os.makedirs(run_output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Wan 2.2 T2V A14B — {label} Expert (Lightning-Optimized)")
    print(f"  Output: {name}")
    print(f"  Dir:    {run_output_dir}")
    print(f"  LR: {LEARNING_RATE} | Scheduler: {LR_SCHEDULER}")
    print(f"  Dim: {NETWORK_DIM} | Alpha: {NETWORK_ALPHA}")
    print(f"  Flow Shift: {DISCRETE_FLOW_SHIFT}")
    print(f"  Lightning merge strength: {LIGHTNING_MERGE_STRENGTH}")
    print("=" * 60)

    if os.path.exists("src/musubi_tuner/wan_train_network.py"):
        SCRIPT_PREFIX = "src/musubi_tuner/"
    else:
        SCRIPT_PREFIX = ""

    # Resume
    resume_weights = None
    resume_dir = f"{OUTPUT_DIR}/resume_checkpoints"
    os.makedirs(resume_dir, exist_ok=True)
    import glob as _glob
    candidates = sorted(_glob.glob(f"{resume_dir}/*.safetensors"), key=os.path.getmtime, reverse=True)
    if candidates:
        resume_weights = candidates[0]
        print(f"\n  RESUME: {resume_weights}")

    # Download models
    comfy_22_repo = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
    comfy_21_repo = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
    wan_repo = "Wan-AI/Wan2.1-I2V-14B-720P"

    dit_file = (
        "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
        if is_high else
        "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
    )

    def cached_download(repo_id, filename):
        try:
            path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
            print(f"  [CACHED] {repo_id}/{filename}")
            return path
        except Exception:
            print(f"  [DOWNLOADING] {repo_id}/{filename} ...")
            return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=False)

    try:
        needs_commit = False
        try:
            huggingface_hub.hf_hub_download(repo_id=comfy_22_repo, filename=dit_file, local_files_only=True)
        except Exception:
            needs_commit = True

        dit_path = cached_download(comfy_22_repo, dit_file)
        vae_path = cached_download(comfy_21_repo, "split_files/vae/wan_2.1_vae.safetensors")
        t5_path = cached_download(wan_repo, "models_t5_umt5-xxl-enc-bf16.pth")

        if needs_commit:
            cache_vol.commit()
    except Exception as e:
        print(f"Error downloading models: {e}")
        _stop_committer.set()
        return

    # Lightning merge
    has_lightning = lightning_cfg.get("repo_id") and lightning_cfg.get("filename")
    if has_lightning:
        print(f"\n{'='*60}")
        print(f"  LIGHTNING MERGE — {label} expert")
        print(f"{'='*60}")

        merged_cache_dir = "/cache/merged_dits"
        os.makedirs(merged_cache_dir, exist_ok=True)
        merge_tag = f"wan22_t2v_{noise_level}_lightning_s{LIGHTNING_MERGE_STRENGTH}"
        merged_dit_path = f"{merged_cache_dir}/{merge_tag}.safetensors"

        if os.path.exists(merged_dit_path):
            print(f"  [CACHED] Merged DiT: {merged_dit_path}")
            dit_path = merged_dit_path
        else:
            try:
                lightning_lora_path = cached_download(lightning_cfg["repo_id"], lightning_cfg["filename"])
            except Exception as e:
                print(f"  ERROR downloading Lightning LoRA: {e}")
                lightning_lora_path = None

            if lightning_lora_path:
                with open("/tmp/merge_lora.py", "w") as f:
                    f.write(MERGE_SCRIPT)
                result = subprocess.run([
                    "python", "/tmp/merge_lora.py",
                    dit_path, lightning_lora_path, merged_dit_path, str(LIGHTNING_MERGE_STRENGTH),
                ])
                if result.returncode == 0 and os.path.exists(merged_dit_path):
                    cache_vol.commit()
                    dit_path = merged_dit_path
                else:
                    print(f"  WARNING: Merge failed. Using vanilla DiT.")

    # Step 1: Cache latents
    print("\nStep 1: Caching latents...")
    subprocess.run([
        "python", f"{SCRIPT_PREFIX}wan_cache_latents.py",
        "--dataset_config", "/dataset-config.toml",
        "--vae", vae_path,
        "--vae_cache_cpu",
        # NOTE: No --i2v for T2V
    ], check=True)

    # Step 2: Cache text encoder
    print("\nStep 2: Caching text encoder outputs...")
    subprocess.run([
        "python", f"{SCRIPT_PREFIX}wan_cache_text_encoder_outputs.py",
        "--dataset_config", "/dataset-config.toml",
        "--t5", t5_path,
        "--batch_size", "16",
        "--fp8_t5",
    ], check=True)

    # Step 3: Train
    print(f"\nStep 3: Training {label} expert...")
    # T2V boundary: 875
    if is_high:
        min_ts, max_ts = "875", "1000"
    else:
        min_ts, max_ts = "0", "875"

    train_cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "fp16",
        f"{SCRIPT_PREFIX}wan_train_network.py",
        "--task", "t2v-A14B",
        "--dit", dit_path,
        "--vae", vae_path,
        "--t5", t5_path,
        "--dataset_config", "/dataset-config.toml",
        "--sdpa",
        "--mixed_precision", "fp16",
        "--fp8_base",
        "--fp8_scaled",
        "--vae_cache_cpu",
        "--min_timestep", min_ts,
        "--max_timestep", max_ts,
        "--preserve_distribution_shape",
        "--optimizer_type", OPTIMIZER,
        "--optimizer_args", "weight_decay=0.01",
        "--learning_rate", LEARNING_RATE,
        "--lr_scheduler", LR_SCHEDULER,
        "--lr_scheduler_min_lr_ratio", MIN_LR_RATIO,
        "--lr_scheduler_power", LR_SCHEDULER_POWER,
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        "--network_module", "networks.lora_wan",
        "--network_dim", NETWORK_DIM,
        "--network_alpha", NETWORK_ALPHA,
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", DISCRETE_FLOW_SHIFT,
        "--max_train_epochs", MAX_EPOCHS,
        "--save_every_n_epochs", SAVE_EVERY,
        "--seed", SEED,
        "--output_dir", run_output_dir,
        "--output_name", name,
        "--log_with", "tensorboard",
        "--logging_dir", f"{run_output_dir}/logs",
    ]

    if resume_weights:
        train_cmd += ["--network_weights", resume_weights]

    print(f"Training command:\n{' '.join(train_cmd)}")
    result = subprocess.run(train_cmd)
    print(f"\nTraining exit code: {result.returncode}")

    _stop_committer.set()
    volume.commit()
    print("Volume committed successfully!")


@app.local_entrypoint()
def run_high():
    call = train.spawn("high")
    print(f"Training dispatched! Function call ID: {call.object_id}")
    print(f"Monitor at: https://modal.com/apps")


@app.local_entrypoint()
def run_low():
    call = train.spawn("low")
    print(f"Training dispatched! Function call ID: {call.object_id}")
    print(f"Monitor at: https://modal.com/apps")
