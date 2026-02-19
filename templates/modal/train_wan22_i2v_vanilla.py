"""
Wan 2.2 I2V A14B LoRA Training — Vanilla (No Lightning)
========================================================
Trains against the base DiT weights without any Lightning merge.
Use this for standard 30-50 step workflows.

Usage:
  1. Upload dataset:  python -m modal run train_wan22_i2v_vanilla.py::add_to_volume_datasets
  2. Train high:      python -m modal run train_wan22_i2v_vanilla.py::run_high
  3. Train low:       python -m modal run train_wan22_i2v_vanilla.py::run_low

At inference: load ONLY your character LoRA in ComfyUI (no Lightning needed).
"""

import os
import datetime
from pathlib import Path

from modal import App, Image, Secret, Volume

# =============================================================================
# ██████  CONFIG  ██████
# =============================================================================

OUTPUT_NAME = "annika-wan22-i2v-vanilla"

LR_SCHEDULER        = "polynomial"
LR_SCHEDULER_POWER  = "2"
MIN_LR_RATIO        = "0.01"
OPTIMIZER           = "adamw8bit"
SEED                = "42"
DISCRETE_FLOW_SHIFT = "5.0"      # I2V standard

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

app = App(name="wan22-i2v-vanilla")
dataset_volume = Volume.from_name("datasets", create_if_missing=True)


@app.local_entrypoint()
def add_to_volume_datasets():
    import pathlib
    script_dir = pathlib.Path(__file__).parent.resolve()
    datasets_path = script_dir / "datasets"
    if not datasets_path.is_dir():
        print(f"ERROR: datasets folder not found at {datasets_path}")
        return "Failed"
    with dataset_volume.batch_upload(force=True) as batch:
        batch.put_directory(str(datasets_path), "/datasets")
    return "Success"


@app.local_entrypoint()
def upload_resume_checkpoint(checkpoint: str):
    import pathlib
    ckpt_path = pathlib.Path(checkpoint).resolve()
    if not ckpt_path.exists():
        print(f"ERROR: File not found: {ckpt_path}")
        return "Failed"
    dest = f"/resume_checkpoints/{ckpt_path.name}"
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(ckpt_path), dest)
    print(f"Uploaded to kohya-volume:{dest}")
    return "Success"


image = (
    Image.debian_slim(python_version="3.10")
    .env({"HF_HUB_CACHE": "/cache/cache/"})
    .apt_install("git", "ffmpeg", "python3-opencv")
    .pip_install("transformers>=4.46.0", "huggingface_hub")
    .run_commands("git clone https://github.com/kohya-ss/musubi-tuner.git")
    .workdir("musubi-tuner")
    .run_commands(
        "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
        "pip3 install -r requirements.txt || pip3 install -e .",
    )
    .pip_install("pydantic==1.10.13", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install("albumentations==1.4.3")
    .add_local_file("wan21-dataset-config.toml", "/dataset-config.toml")
)

volume = Volume.from_name("kohya-volume", create_if_missing=True)
cache_vol = Volume.from_name("hf-hub-cache", create_if_missing=True)
OUTPUT_DIR = "/outputs"


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={OUTPUT_DIR: volume, "/cache": cache_vol, "/datasets": dataset_volume},
    timeout=60 * 60 * 24,
    secrets=[Secret.from_name("my-huggingface-secret")],
)
def train(noise_level: str):
    import subprocess, glob, threading
    import huggingface_hub, itertools

    img_count = len(list(itertools.chain(Path("/datasets").rglob("*.png"), Path("/datasets").rglob("*.webp"), Path("/datasets").rglob("*.jpg"))))
    vid_count = len(list(Path("/datasets").rglob("*.mp4")))
    print(f"Dataset volume: {img_count} images, {vid_count} videos")

    is_high = noise_level == "high"
    label = "HIGH-NOISE" if is_high else "LOW-NOISE"
    expert = EXPERT_CONFIG[noise_level]

    _stop_committer = threading.Event()
    def _periodic_commit(interval_seconds=300):
        while not _stop_committer.is_set():
            _stop_committer.wait(interval_seconds)
            if _stop_committer.is_set(): break
            try:
                n = len(glob.glob(os.path.join(OUTPUT_DIR, "**/*.safetensors"), recursive=True))
                if n > 0: volume.commit()
            except Exception: pass
    threading.Thread(target=_periodic_commit, daemon=True).start()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{OUTPUT_NAME}-{noise_level}-{timestamp}"
    run_output_dir = f"{OUTPUT_DIR}/{OUTPUT_NAME}"
    os.makedirs(run_output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Wan 2.2 I2V A14B — {label} Expert — VANILLA")
    print(f"  Output: {name}")
    print("=" * 60)

    SCRIPT_PREFIX = "src/musubi_tuner/" if os.path.exists("src/musubi_tuner/wan_train_network.py") else ""

    # Resume
    resume_weights = None
    resume_dir = f"{OUTPUT_DIR}/resume_checkpoints"
    os.makedirs(resume_dir, exist_ok=True)
    candidates = sorted(glob.glob(f"{resume_dir}/*.safetensors"), key=os.path.getmtime, reverse=True)
    if candidates:
        resume_weights = candidates[0]
        print(f"  RESUME: {resume_weights}")

    # Download models
    comfy_22 = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
    comfy_21 = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"

    def cached_download(repo_id, filename):
        try:
            return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
        except Exception:
            return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)

    dit_file = f"split_files/diffusion_models/wan2.2_i2v_{'high' if is_high else 'low'}_noise_14B_fp16.safetensors"
    dit_path = cached_download(comfy_22, dit_file)
    vae_path = cached_download(comfy_21, "split_files/vae/wan_2.1_vae.safetensors")
    t5_path = cached_download("Wan-AI/Wan2.1-I2V-14B-720P", "models_t5_umt5-xxl-enc-bf16.pth")
    cache_vol.commit()

    # Cache latents
    subprocess.run(["python", f"{SCRIPT_PREFIX}wan_cache_latents.py",
        "--dataset_config", "/dataset-config.toml", "--vae", vae_path, "--vae_cache_cpu", "--i2v"], check=True)

    # Cache text encoder
    subprocess.run(["python", f"{SCRIPT_PREFIX}wan_cache_text_encoder_outputs.py",
        "--dataset_config", "/dataset-config.toml", "--t5", t5_path, "--batch_size", "16", "--fp8_t5"], check=True)

    # Train — I2V boundary: 900
    min_ts, max_ts = ("900", "1000") if is_high else ("0", "900")
    train_cmd = [
        "accelerate", "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", "fp16",
        f"{SCRIPT_PREFIX}wan_train_network.py",
        "--task", "i2v-A14B",
        "--dit", dit_path, "--vae", vae_path, "--t5", t5_path,
        "--dataset_config", "/dataset-config.toml",
        "--sdpa", "--mixed_precision", "fp16", "--fp8_base", "--fp8_scaled", "--vae_cache_cpu",
        "--min_timestep", min_ts, "--max_timestep", max_ts, "--preserve_distribution_shape",
        "--optimizer_type", OPTIMIZER, "--optimizer_args", "weight_decay=0.01",
        "--learning_rate", expert["learning_rate"],
        "--lr_scheduler", LR_SCHEDULER, "--lr_scheduler_min_lr_ratio", MIN_LR_RATIO, "--lr_scheduler_power", LR_SCHEDULER_POWER,
        "--gradient_checkpointing", "--max_data_loader_n_workers", "2", "--persistent_data_loader_workers",
        "--network_module", "networks.lora_wan", "--network_dim", expert["network_dim"], "--network_alpha", expert["network_alpha"],
        "--timestep_sampling", "shift", "--discrete_flow_shift", DISCRETE_FLOW_SHIFT,
        "--max_train_epochs", expert["max_epochs"], "--save_every_n_epochs", expert["save_every"],
        "--seed", SEED, "--output_dir", run_output_dir, "--output_name", name,
        "--log_with", "tensorboard", "--logging_dir", f"{run_output_dir}/logs",
    ]
    if resume_weights:
        train_cmd += ["--network_weights", resume_weights]

    result = subprocess.run(train_cmd)
    print(f"\nTraining exit code: {result.returncode}")
    _stop_committer.set()
    volume.commit()


@app.local_entrypoint()
def run_high():
    call = train.spawn("high")
    print(f"Training dispatched! ID: {call.object_id}")

@app.local_entrypoint()
def run_low():
    call = train.spawn("low")
    print(f"Training dispatched! ID: {call.object_id}")
