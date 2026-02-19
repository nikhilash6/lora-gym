"""
Wan 2.1 T2V 1.3B LoRA Training — Modal Edition
================================================
Smallest Wan model (~2.84 GB bf16). No fp8 needed.
Can run on smaller/cheaper GPUs.

Usage:
  1. Upload dataset:  python -m modal run train_wan21_t2v_1_3b.py::add_to_volume_datasets
  2. Train:           python -m modal run train_wan21_t2v_1_3b.py::run
"""

import os
import datetime
from pathlib import Path

from modal import App, Image, Secret, Volume

# =============================================================================
# ██████  CONFIG  ██████
# =============================================================================

OUTPUT_NAME         = "annika-wan21-t2v-1.3b"
LEARNING_RATE       = "8e-5"
LR_SCHEDULER        = "polynomial"
LR_SCHEDULER_POWER  = "2"
MIN_LR_RATIO        = "0.01"
OPTIMIZER           = "adamw8bit"
NETWORK_DIM         = "16"       # Smaller model = smaller LoRA rank
NETWORK_ALPHA       = "16"
MAX_EPOCHS          = "50"
SAVE_EVERY          = "5"
SEED                = "42"
DISCRETE_FLOW_SHIFT = "3.0"

# =============================================================================
# Modal setup
# =============================================================================

app = App(name="wan21-t2v-1_3b")
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
    gpu="A10G",                    # 1.3B model fits on smaller GPUs
    volumes={OUTPUT_DIR: volume, "/cache": cache_vol, "/datasets": dataset_volume},
    timeout=60 * 60 * 24,
    secrets=[Secret.from_name("my-huggingface-secret")],
)
def train():
    import subprocess, glob, threading
    import huggingface_hub, itertools

    img_count = len(list(itertools.chain(Path("/datasets").rglob("*.png"), Path("/datasets").rglob("*.webp"), Path("/datasets").rglob("*.jpg"))))
    vid_count = len(list(Path("/datasets").rglob("*.mp4")))
    print(f"Dataset volume: {img_count} images, {vid_count} videos")

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
    name = f"{OUTPUT_NAME}-{timestamp}"
    run_output_dir = f"{OUTPUT_DIR}/{OUTPUT_NAME}"
    os.makedirs(run_output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Wan 2.1 T2V 1.3B")
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
    def cached_download(repo_id, filename):
        try:
            return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
        except Exception:
            return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)

    dit_path = cached_download("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors")
    vae_path = cached_download("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors")
    t5_path = cached_download("Wan-AI/Wan2.1-I2V-14B-720P", "models_t5_umt5-xxl-enc-bf16.pth")
    cache_vol.commit()

    # Cache latents
    subprocess.run(["python", f"{SCRIPT_PREFIX}wan_cache_latents.py",
        "--dataset_config", "/dataset-config.toml", "--vae", vae_path, "--vae_cache_cpu"], check=True)

    # Cache text encoder
    subprocess.run(["python", f"{SCRIPT_PREFIX}wan_cache_text_encoder_outputs.py",
        "--dataset_config", "/dataset-config.toml", "--t5", t5_path, "--batch_size", "16", "--fp8_t5"], check=True)

    # Train — single model, no timestep splitting, no fp8_base (small model)
    train_cmd = [
        "accelerate", "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16",
        f"{SCRIPT_PREFIX}wan_train_network.py",
        "--task", "t2v-1.3B",
        "--dit", dit_path, "--vae", vae_path, "--t5", t5_path,
        "--dataset_config", "/dataset-config.toml",
        "--sdpa", "--mixed_precision", "bf16", "--vae_cache_cpu",
        # NOTE: No --fp8_base — model is small enough without it
        "--optimizer_type", OPTIMIZER, "--optimizer_args", "weight_decay=0.01",
        "--learning_rate", LEARNING_RATE,
        "--lr_scheduler", LR_SCHEDULER, "--lr_scheduler_min_lr_ratio", MIN_LR_RATIO, "--lr_scheduler_power", LR_SCHEDULER_POWER,
        "--gradient_checkpointing", "--max_data_loader_n_workers", "2", "--persistent_data_loader_workers",
        "--network_module", "networks.lora_wan", "--network_dim", NETWORK_DIM, "--network_alpha", NETWORK_ALPHA,
        "--timestep_sampling", "shift", "--discrete_flow_shift", DISCRETE_FLOW_SHIFT,
        "--max_train_epochs", MAX_EPOCHS, "--save_every_n_epochs", SAVE_EVERY,
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
def run():
    call = train.spawn()
    print(f"Training dispatched! ID: {call.object_id}")
    print(f"Monitor at: https://modal.com/apps")
