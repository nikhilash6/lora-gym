# Local Training Guide — Wan LoRA Training with Musubi-Tuner

## What This Guide Covers

Step-by-step instructions for training Wan 2.2 LoRA models on your own GPU using musubi-tuner. This covers installing prerequisites, running the setup script, preparing your dataset, and launching training — all on Windows or Linux.

Training is slower than cloud A100s (~2x) but produces identical results and costs nothing beyond electricity.

---

## Step 1: Pick Your Template Script

All 16 training templates have local versions. Pick the one that matches your model and workflow:

### Wan 2.2 (Dual Experts — train high + low separately)

| Script | Model | Use Case |
|---|---|---|
| `train_local_t2v_lightning.py` | Wan 2.2 T2V A14B | T2V, optimized for Lightning speed LoRA |
| `train_local_t2v_vanilla.py` | Wan 2.2 T2V A14B | T2V, standard 30-50 step workflows |
| `train_local_i2v_lightning.py` | Wan 2.2 I2V A14B | I2V, optimized for Lightning speed LoRA |
| `train_local_i2v_vanilla.py` | Wan 2.2 I2V A14B | I2V, standard 30-50 step workflows |

### Wan 2.1 (Single Model — one training run)

| Script | Model | Use Case |
|---|---|---|
| `train_local_wan21_t2v_14b.py` | Wan 2.1 T2V 14B | Text-to-video, full quality |
| `train_local_wan21_t2v_1_3b.py` | Wan 2.1 T2V 1.3B | Text-to-video, lightweight prototyping |
| `train_local_wan21_i2v_720p.py` | Wan 2.1 I2V 14B 720p | Image-to-video, full quality |
| `train_local_wan21_i2v_480p.py` | Wan 2.1 I2V 14B 480p | Image-to-video, lower res |

**Vanilla vs Lightning:** Vanilla trains against the stock model weights — use if you don't plan to use a Lightning/speed-up LoRA at inference. Lightning merges a speed LoRA into the base weights before training, so your character LoRA is tuned specifically for the Lightning denoising path. At inference, load Lightning + your character LoRA together in ComfyUI.

**Wan 2.2 vs 2.1:** Wan 2.2 uses a Mixture-of-Experts architecture — you train two LoRAs per concept (high-noise + low-noise) and load both at inference. Wan 2.1 is a single model, one training run.

**Note:** `setup_local.py` downloads Wan 2.2 T2V models by default. For I2V, add `--include_i2v`. For Wan 2.1 models, download them manually from HuggingFace or use the `--dit` flag to point to existing files.

---

## Quick Reference — Returning Users

If you've done this before and just need the commands:

```bash
# 1. Run setup (skips already-downloaded files)
python setup_local.py

# 2. Edit dataset config — update paths to your dataset
#    (wan22-dataset-config-local.toml)

# 3. Train both experts
python train_local_t2v_lightning.py --noise_level high
python train_local_t2v_lightning.py --noise_level low
```

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 24 GB (RTX 3090/4090) | 48 GB (A6000, RTX 6000 Ada) |
| System RAM | 48 GB | 64 GB |
| Disk Space | 80 GB free | 120 GB free |

**24GB GPUs (RTX 3090/4090):** Image training works out of the box. Video training may require additional memory flags — add `--blocks_to_swap 20` to offload transformer blocks to CPU if you hit OOM. See the [Troubleshooting](#troubleshooting) section.

**48GB GPUs (A6000, RTX 6000 Ada):** Everything works without extra flags. This is the most comfortable local training experience.

**Why 48GB+ system RAM?** The Lightning merge loads the full ~28GB DiT model on CPU. With 48GB RAM you'll have enough headroom. If you have less RAM, use `--skip_lightning` to skip the merge. You can still use Lightning LoRA at inference — the weights will be slightly mismatched which may reduce character consistency, but Lightning speed still works.

**Training speed:** Expect ~35-40 seconds/step on RTX 4090/A6000. A 50-epoch run on a typical dataset takes 12-20 hours.

---

## Full Step-by-Step Setup (First Time)

### Step 2: Install Prerequisites

You need Python 3.10+, Git, and CUDA-compatible PyTorch.

**Windows:**
1. Install [Python 3.10+](https://www.python.org/downloads/) (check "Add to PATH" during install)
2. Install [Git](https://git-scm.com/download/win)
3. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (12.1+ recommended)
4. Install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Linux:**
```bash
# Python and Git are usually pre-installed
# Install CUDA toolkit from your distro's package manager, then:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify your setup:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.get_device_name(0)}')"
```

### Step 3: Run the Setup Script

```bash
python setup_local.py
```

This will:
1. Clone musubi-tuner from GitHub
2. Install Python dependencies
3. Download ~40GB of model weights from HuggingFace (T2V only)
4. Create the directory structure

**Already have ComfyUI?** Save ~10GB of downloads by reusing compatible models:
```bash
python setup_local.py --comfyui_dir C:/path/to/ComfyUI
```

The setup script reuses your VAE and T5 encoder but still downloads the fp16 DiT weights (ComfyUI's fp8 versions can't be used for training).

**Custom install location:**
```bash
python setup_local.py --base_dir D:/my_training
```

### Step 4: Prepare Your Dataset

Create a folder structure under `local_training/datasets/`:

```
local_training/datasets/your_character/
  images/
    photo_001.png
    photo_001.txt
    photo_002.jpg
    photo_002.txt
    ...
  videos/
    clip_001.mp4
    clip_001.txt
    clip_002.mp4
    clip_002.txt
    ...
```

**Every image and video MUST have a matching `.txt` caption file** with the same base name. If captions are missing, musubi-tuner silently skips those files.

Use the captioning tools in `captioning/` to auto-generate captions, or write them manually.

### Step 5: Edit the Dataset Config

Open `wan22-dataset-config-local.toml` and update the paths to point to your dataset:

```toml
[[datasets]]
image_directory = "C:/path/to/local_training/datasets/your_character/images"
cache_directory = "C:/path/to/local_training/datasets/your_character/images/cache"
```

**Use forward slashes** (`/`) even on Windows — TOML and Python handle them fine.

**Tip:** The config has three `[[datasets]]` entries — one for images, two for videos at different resolutions. If you only have images, you can delete or comment out the video entries.

### Step 6: Launch Training

Train both experts (run sequentially — each uses your full GPU):

```bash
# High-noise expert (composition/motion)
python train_local_t2v_lightning.py --noise_level high

# Low-noise expert (texture/identity)
python train_local_t2v_lightning.py --noise_level low
```

The script handles everything automatically:
1. Validates your dataset config (checks that paths exist)
2. Downloads and merges the Lightning LoRA into the DiT (~5-10 min on CPU)
3. Caches VAE latents (~5 min for a typical dataset)
4. Caches T5 text encoder outputs (~2 min)
5. Trains the LoRA (~12-20 hours for 50 epochs)

**Custom hyperparameters:**
```bash
python train_local_t2v_lightning.py --noise_level high --lr 5e-5 --dim 32 --epochs 30
```

**Resume from a checkpoint:**
```bash
# Explicit path
python train_local_t2v_lightning.py --noise_level high --resume_from ./outputs/my-lora-e25.safetensors

# Auto-detect: drop a .safetensors in local_training/resume_checkpoints/
python train_local_t2v_lightning.py --noise_level high
```

### Step 7: Use Your LoRA in ComfyUI

When training completes, checkpoints are saved to `local_training/outputs/`. Copy the `.safetensors` files to your ComfyUI `models/loras/` folder.

At inference, load **two** LoRAs in ComfyUI:
1. The Lightning speed LoRA (from `lightx2v/Wan2.2-Lightning` on HuggingFace)
2. Your trained character LoRA

---

## Troubleshooting

### "ERROR: Model files not found"
Run `setup_local.py` first. If you already ran it, check that `--base_dir` matches between setup and training.

### Out of memory (OOM) during training
The training script uses `--fp8_base` and `--gradient_checkpointing` to reduce VRAM usage. On 48GB cards this is sufficient. On 24GB cards (3090/4090), video training may OOM. Try these in order:
1. Add `--blocks_to_swap 20` (offloads transformer blocks to CPU — slower but uses less VRAM)
2. Increase to `--blocks_to_swap 36` if still OOMing (max 39 for 14B)
3. Reduce resolution in the TOML config (try `[544, 960]` instead of `[720, 1280]`)
4. Reduce `target_frames` for video datasets

### Out of memory during Lightning merge
The merge runs on CPU and needs ~40-50GB system RAM. If it fails:
```bash
python train_local_t2v_lightning.py --noise_level high --skip_lightning
```
This trains against the vanilla DiT instead of Lightning-merged weights. You can still use Lightning LoRA at inference — the weights will be slightly mismatched which may reduce character consistency, but Lightning speed still works.

### "UnicodeEncodeError" on Windows
The script sets `PYTHONIOENCODING=utf-8` automatically. If you still see encoding errors, set the environment variable manually:
```bash
set PYTHONIOENCODING=utf-8
python train_local_t2v_lightning.py --noise_level high
```

### "accelerate not found"
The script uses `python -m accelerate` instead of the bare `accelerate` command to avoid PATH issues. If you still get errors:
```bash
pip install accelerate
```

### "ValueError: No training items found"
The caches are missing or empty. Check:
- **Path mismatch** — TOML paths must exactly match your actual folder names (case-sensitive on Linux)
- **Missing captions** — every image/video needs a `.txt` with the same base name
- **Forward slashes** — use `/` not `\` in TOML paths, even on Windows

### Training seems stuck at 0%
For Lightning scripts, the LoRA merge runs on CPU before GPU training starts. This takes 5-10 minutes for a 14B model. Look for "Loading base DiT" in the logs — that's the merge step.

---

## Time and Cost Estimates

| GPU | VRAM | Training Speed | 50 Epochs (typical dataset) | Cost |
|---|---|---|---|---|
| A6000 | 48GB | ~36-39 s/step | ~12-16 hours | Electricity only |
| RTX 4090 | 24GB | ~35-40 s/step | ~12-16 hours | Electricity only |
| RTX 3090 | 24GB | ~45-55 s/step | ~16-22 hours | Electricity only |

For comparison, an A100-80GB on RunPod does ~18 s/step (~6-10 hours for 50 epochs, ~$10-16/expert).

Total for both Wan 2.2 experts locally: **free** (just time and electricity).

**Note:** 24GB cards may need `--blocks_to_swap` for video training, which adds ~10-20% overhead to step time due to CPU offloading.
