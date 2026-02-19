"""
Wan 2.2 T2V A14B LoRA Training — RunPod Edition (Vanilla / No Lightning)
=========================================================================
Trains a character identity LoRA directly against the base Wan 2.2 DiT
weights — no Lightning LoRA merge. Use this when you want a LoRA that
works with the standard 30-50 step workflow (not Lightning's 4-step).

Usage:
  # Train high-noise expert
  cd /workspace/musubi-tuner
  python /workspace/train_runpod_t2v_vanilla.py --noise_level high

  # Train low-noise expert
  python /workspace/train_runpod_t2v_vanilla.py --noise_level low

  # Train both sequentially
  python /workspace/train_runpod_t2v_vanilla.py --noise_level high
  python /workspace/train_runpod_t2v_vanilla.py --noise_level low

  # Custom config
  python /workspace/train_runpod_t2v_vanilla.py --noise_level high --lr 5e-5 --epochs 30 --dim 32

  # Resume from a specific checkpoint
  python /workspace/train_runpod_t2v_vanilla.py --noise_level high --resume_from /workspace/outputs/my-lora-e25.safetensors

At inference: load ONLY your character LoRA in ComfyUI (no Lightning LoRA needed).

Model downloads required (add to setup_runpod.sh or download manually):
  Comfy-Org/Wan_2.2_ComfyUI_Repackaged:
    split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors
    split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors
  Comfy-Org/Wan_2.1_ComfyUI_repackaged:
    split_files/vae/wan_2.1_vae.safetensors
  Wan-AI/Wan2.1-I2V-14B-720P:
    models_t5_umt5-xxl-enc-bf16.pth
"""

import os
import sys
import subprocess
import datetime
import argparse
import glob
from pathlib import Path

# =============================================================================
# ██████  CONFIG  ██████
# =============================================================================
# These are defaults — override any of them via command-line arguments.

OUTPUT_NAME = "annika-wan22-t2v-vanilla"

# --- Shared Hyperparameters ---
LR_SCHEDULER        = "polynomial"
LR_SCHEDULER_POWER  = "2"
MIN_LR_RATIO        = "0.01"
OPTIMIZER           = "adamw8bit"
SEED                = "42"
DISCRETE_FLOW_SHIFT = "3.0"       # T2V standard

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

# --- Paths on RunPod ---
WORKSPACE          = "/workspace"
MODELS_DIR         = f"{WORKSPACE}/models"
DATASETS_DIR       = f"{WORKSPACE}/datasets"
OUTPUTS_DIR        = f"{WORKSPACE}/outputs"
DATASET_CONFIG     = f"{WORKSPACE}/wan21-dataset-config.toml"
MUSUBI_DIR         = f"{WORKSPACE}/musubi-tuner"
RESUME_DIR         = f"{WORKSPACE}/resume_checkpoints"

# --- Model files (pre-downloaded by setup_runpod.sh) ---
MODEL_PATHS = {
    "dit_high": f"{MODELS_DIR}/wan2.2_t2v_high_noise_14B_fp16.safetensors",
    "dit_low":  f"{MODELS_DIR}/wan2.2_t2v_low_noise_14B_fp16.safetensors",
    "vae":      f"{MODELS_DIR}/wan_2.1_vae.safetensors",
    "t5":       f"{MODELS_DIR}/models_t5_umt5-xxl-enc-bf16.pth",
}


# =============================================================================
# Main Training Function
# =============================================================================

def train(args):
    noise_level = args.noise_level
    is_high = noise_level == "high"
    label = "HIGH-NOISE" if is_high else "LOW-NOISE"

    # --- Resolve hyperparameters (CLI overrides > per-expert defaults) ---
    expert = EXPERT_CONFIG[noise_level]
    lr          = args.lr or expert["learning_rate"]
    scheduler   = args.scheduler or LR_SCHEDULER
    power       = args.power or LR_SCHEDULER_POWER
    min_lr      = args.min_lr or MIN_LR_RATIO
    optimizer   = args.optimizer or OPTIMIZER
    dim         = args.dim or expert["network_dim"]
    alpha       = args.alpha or expert["network_alpha"]
    epochs      = args.epochs or expert["max_epochs"]
    save_every  = args.save_every or expert["save_every"]
    seed        = args.seed or SEED
    flow_shift  = args.flow_shift or DISCRETE_FLOW_SHIFT
    output_name = args.output_name or OUTPUT_NAME

    # --- Validate prerequisites ---
    dit_key = "dit_high" if is_high else "dit_low"
    dit_path = MODEL_PATHS[dit_key]
    vae_path = MODEL_PATHS["vae"]
    t5_path  = MODEL_PATHS["t5"]

    missing = []
    for name, path in [("DiT", dit_path), ("VAE", vae_path), ("T5", t5_path)]:
        if not os.path.exists(path):
            missing.append(f"  {name}: {path}")
    if missing:
        print("ERROR: Model files not found. Run setup_runpod.sh first.")
        for m in missing:
            print(m)
        sys.exit(1)

    if not os.path.exists(DATASET_CONFIG):
        print(f"ERROR: Dataset config not found: {DATASET_CONFIG}")
        print(f"Upload wan21-dataset-config.toml to /workspace/")
        sys.exit(1)

    # --- Detect musubi-tuner repo structure ---
    if os.path.exists(f"{MUSUBI_DIR}/src/musubi_tuner/wan_train_network.py"):
        SCRIPT_PREFIX = "src/musubi_tuner/"
        print("Detected new musubi-tuner repo structure")
    else:
        SCRIPT_PREFIX = ""
        print("Detected classic musubi-tuner repo structure")

    # --- Timestamped output ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{output_name}-{noise_level}-{timestamp}"
    run_output_dir = f"{OUTPUTS_DIR}/{output_name}"
    os.makedirs(run_output_dir, exist_ok=True)

    # =================================================================
    # Resume from checkpoint (optional)
    # =================================================================
    resume_weights = None
    os.makedirs(RESUME_DIR, exist_ok=True)

    if args.resume_from:
        if os.path.exists(args.resume_from):
            resume_weights = args.resume_from
            print(f"\n  RESUME: Using explicitly specified checkpoint:")
            print(f"    {resume_weights}")
        else:
            print(f"\n  WARNING: --resume_from path not found: {args.resume_from}")
            print(f"  Training will start from scratch.")
    else:
        candidates = sorted(
            glob.glob(f"{RESUME_DIR}/*.safetensors"),
            key=os.path.getmtime,
            reverse=True,
        )
        if candidates:
            resume_weights = candidates[0]
            print(f"\n  RESUME: Auto-detected checkpoint in {RESUME_DIR}/:")
            print(f"    {resume_weights}")
            if len(candidates) > 1:
                print(f"    ({len(candidates)} files found — using newest by modification time)")
        else:
            print(f"\n  No resume checkpoint found. Starting from scratch.")
            print(f"  (To resume: drop a .safetensors in {RESUME_DIR}/ or use --resume_from)")

    if resume_weights:
        size_mb = os.path.getsize(resume_weights) / (1024 * 1024)
        print(f"    File size: {size_mb:.1f} MB")
        print(f"    NOTE: Loading weights only (no optimizer state) — LR schedule starts fresh.")

    print("=" * 60)
    print(f"  Wan 2.2 T2V A14B — {label} Expert — VANILLA (RunPod)")
    print(f"  Training against base DiT (no Lightning merge)")
    print(f"  Output: {name}")
    print(f"  Dir:    {run_output_dir}")
    print(f"  LR: {lr} | Scheduler: {scheduler}")
    print(f"  Dim: {dim} | Alpha: {alpha}")
    print(f"  Epochs: {epochs} | Flow Shift: {flow_shift}")
    if resume_weights:
        print(f"  ** RESUMING from: {os.path.basename(resume_weights)} **")
    print("=" * 60)

    # =================================================================
    # Step 1: Cache latents
    # =================================================================
    print("\n" + "=" * 60)
    print("  Step 1: Caching latents...")
    print("=" * 60)
    subprocess.run([
        "python",
        f"{SCRIPT_PREFIX}wan_cache_latents.py",
        "--dataset_config", DATASET_CONFIG,
        "--vae", vae_path,
        "--vae_cache_cpu",
        # NOTE: No --i2v flag for T2V models
    ], check=True)

    # =================================================================
    # Step 2: Cache text encoder outputs
    # =================================================================
    print("\n" + "=" * 60)
    print("  Step 2: Caching text encoder outputs...")
    print("=" * 60)
    subprocess.run([
        "python",
        f"{SCRIPT_PREFIX}wan_cache_text_encoder_outputs.py",
        "--dataset_config", DATASET_CONFIG,
        "--t5", t5_path,
        "--batch_size", "16",
        "--fp8_t5",
    ], check=True)

    # =================================================================
    # Step 3: Train LoRA
    # =================================================================
    print("\n" + "=" * 60)
    print(f"  Step 3: Training {label} expert...")
    print(f"  DiT: {dit_path}")
    print(f"  ** Training against vanilla (unmodified) DiT weights **")
    print("=" * 60)

    # T2V boundary: 875
    if is_high:
        min_ts, max_ts = "875", "1000"
    else:
        min_ts, max_ts = "0", "875"

    train_cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "fp16",       # Wan 2.2 weights are fp16
        f"{SCRIPT_PREFIX}wan_train_network.py",
        "--task", "t2v-A14B",              # Wan 2.2 T2V MoE architecture
        "--dit", dit_path,
        "--vae", vae_path,
        "--t5", t5_path,
        "--dataset_config", DATASET_CONFIG,
        "--sdpa",
        "--mixed_precision", "fp16",       # Wan 2.2 = fp16 (NOT bf16)
        "--fp8_base",
        "--fp8_scaled",                    # Valid for Wan 2.2 only
        "--vae_cache_cpu",
        # --- Wan 2.2 T2V specific ---
        "--min_timestep", min_ts,
        "--max_timestep", max_ts,
        "--preserve_distribution_shape",
        # --- Optimizer ---
        "--optimizer_type", optimizer,
        "--optimizer_args", "weight_decay=0.01",
        # --- Learning Rate ---
        "--learning_rate", lr,
        # --- LR Scheduler ---
        "--lr_scheduler", scheduler,
        "--lr_scheduler_min_lr_ratio", min_lr,
        "--lr_scheduler_power", power,
        # --- Memory ---
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        # --- LoRA Config ---
        "--network_module", "networks.lora_wan",
        "--network_dim", dim,
        "--network_alpha", alpha,
        # --- Timestep / Flow ---
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", flow_shift,
        # --- Training Duration ---
        "--max_train_epochs", epochs,
        "--save_every_n_epochs", save_every,
        # --- Output ---
        "--seed", seed,
        "--output_dir", run_output_dir,
        "--output_name", name,
        # --- Logging ---
        "--log_with", "tensorboard",
        "--logging_dir", f"{run_output_dir}/logs",
    ]

    if resume_weights:
        train_cmd += ["--network_weights", resume_weights]
        print(f"\n  >> --network_weights {resume_weights}")

    print(f"\nTraining command:\n{' '.join(train_cmd)}\n")
    result = subprocess.run(train_cmd)
    print(f"\nTraining exit code: {result.returncode}")

    if result.returncode != 0:
        print(f"ERROR: Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"{'='*60}")
    print(f"\nContents of {run_output_dir}:")
    for root, dirs, files in os.walk(run_output_dir):
        level = root.replace(run_output_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in sorted(files):
            filepath = os.path.join(root, f)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"{subindent}{f} ({size_mb:.1f} MB)")

    print(f"\nTo download results to your PC:")
    print(f"  scp -P PORT -i ~/.ssh/KEY root@HOST:{run_output_dir}/*.safetensors .")


# =============================================================================
# CLI Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Wan 2.2 T2V A14B LoRA Training — RunPod Edition (Vanilla / No Lightning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train high-noise expert with defaults
  python train_runpod_t2v_vanilla.py --noise_level high

  # Train low-noise with custom learning rate
  python train_runpod_t2v_vanilla.py --noise_level low --lr 5e-5

  # Resume from a specific checkpoint
  python train_runpod_t2v_vanilla.py --noise_level high --resume_from /workspace/outputs/my-lora-e25.safetensors

  # Full custom config
  python train_runpod_t2v_vanilla.py --noise_level high --lr 5e-5 --dim 32 --alpha 32 --epochs 30
        """
    )

    parser.add_argument("--noise_level", required=True, choices=["high", "low"],
                        help="Which noise expert to train: 'high' or 'low'")
    parser.add_argument("--lr", default=None, help="Learning rate (default: 8e-5)")
    parser.add_argument("--scheduler", default=None, help=f"LR scheduler (default: {LR_SCHEDULER})")
    parser.add_argument("--power", default=None, help=f"Scheduler power (default: {LR_SCHEDULER_POWER})")
    parser.add_argument("--min_lr", default=None, help=f"Min LR ratio (default: {MIN_LR_RATIO})")
    parser.add_argument("--optimizer", default=None, help=f"Optimizer (default: {OPTIMIZER})")
    parser.add_argument("--dim", default=None, help="LoRA rank/dim (default: 24)")
    parser.add_argument("--alpha", default=None, help="LoRA alpha (default: 24)")
    parser.add_argument("--epochs", default=None, help="Max epochs (default: 50)")
    parser.add_argument("--save_every", default=None, help="Save interval (default: 5)")
    parser.add_argument("--seed", default=None, help=f"Random seed (default: {SEED})")
    parser.add_argument("--flow_shift", default=None, help=f"Discrete flow shift (default: {DISCRETE_FLOW_SHIFT})")
    parser.add_argument("--output_name", default=None, help=f"Output name prefix (default: {OUTPUT_NAME})")
    parser.add_argument("--resume_from", default=None,
                        help="Path to a .safetensors LoRA checkpoint to resume from. "
                             f"If not specified, auto-checks {RESUME_DIR}/")

    return parser.parse_args()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    if not os.path.exists(MUSUBI_DIR):
        print(f"ERROR: musubi-tuner not found at {MUSUBI_DIR}")
        print(f"Run setup_runpod.sh first.")
        sys.exit(1)

    os.chdir(MUSUBI_DIR)
    print(f"Working directory: {os.getcwd()}")

    args = parse_args()
    train(args)
