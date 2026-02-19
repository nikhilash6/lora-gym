# RunPod Training Guide — Wan LoRA Training with Musubi-Tuner

## What This Guide Covers

Step-by-step instructions for training Wan 2.1 and 2.2 LoRA models on RunPod using musubi-tuner. This covers setting up a pod, picking a template script, uploading your dataset, running training, and downloading checkpoints.

---

## Step 1: Pick Your Template Script

Each script is a self-contained RunPod training template. Pick the one that matches your model and workflow. All scripts are in `runpod_py_templates.zip`.

### Wan 2.1 (Single Model — one training run)

| Script | Model | Use Case |
|---|---|---|
| `train_runpod_wan21_t2v_14b.py` | Wan 2.1 T2V 14B | Text-to-video, full quality |
| `train_runpod_wan21_t2v_1_3b.py` | Wan 2.1 T2V 1.3B | Text-to-video, lightweight/cheap prototyping |
| `train_runpod_wan21_i2v_480p.py` | Wan 2.1 I2V 14B 480p | Image-to-video, lower res |
| `train_runpod_wan21_i2v_720p.py` | Wan 2.1 I2V 14B 720p | Image-to-video, full quality |

### Wan 2.2 (Dual Experts — train high + low separately)

| Script | Model | Use Case |
|---|---|---|
| `train_runpod_t2v_vanilla.py` | Wan 2.2 T2V A14B | T2V, standard 30-50 step workflows |
| `train_runpod_t2v_lightning.py` | Wan 2.2 T2V A14B | T2V, optimized for Lightning speed LoRA |
| `train_runpod_i2v_vanilla.py` | Wan 2.2 I2V A14B | I2V, standard 30-50 step workflows |
| `train_runpod_i2v_lightning.py` | Wan 2.2 I2V A14B | I2V, optimized for Lightning speed LoRA |

**Vanilla vs Lightning:** Vanilla trains against the stock model weights — use if you don't plan to use a Lightning/speed-up LoRA at inference. Lightning merges a speed LoRA into the base weights before training, so your character LoRA is tuned specifically for the Lightning denoising path. At inference, load Lightning + your character LoRA together.

**Key rule:** The model you train on MUST match the model you inference with in ComfyUI. A LoRA trained on T2V won't work correctly on I2V, and vice versa.

---

## Quick Reference — Returning Users

If you've done this before and just need the commands:

```bash
# 1. SSH into your pod
ssh root@<address> -p <port> -i ~/.ssh/id_ed25519

# 2. Fix filenames with spaces (MUST do before caching)
cd /workspace/datasets/annika/images/
for f in *\ *; do mv "$f" "$(echo $f | tr ' ' '_')"; done
cd /workspace/datasets/annika/videos/
for f in *\ *; do mv "$f" "$(echo $f | tr ' ' '_')"; done

# 3. Cache latents
cd /workspace/musubi-tuner
python src/musubi_tuner/wan_cache_latents.py \
  --dataset_config /workspace/wan22-dataset-config-runpod.toml \
  --vae /workspace/models/wan_2.1_vae.safetensors \
  --batch_size 1

# 4. Cache text encoder
python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
  --dataset_config /workspace/wan22-dataset-config-runpod.toml \
  --t5 /workspace/models/models_t5_umt5-xxl-enc-bf16.pth \
  --fp8_t5 --batch_size 1

# 5. Launch training in tmux
tmux new -s train
python /workspace/<your-script>.py --noise_level high   # Wan 2.2
python /workspace/<your-script>.py                       # Wan 2.1
```

---

## Full Step-by-Step Setup (First Time)

### Step 2: Create a RunPod Account

1. Go to [runpod.io](https://www.runpod.io/) and sign up
2. Add credits (Community Cloud is cheapest — about $1.64/hr for A100-80GB)
3. Set up SSH keys for secure access (see Step 3)

### Step 3: Set Up SSH Keys (One-Time, on Your PC)

You need SSH keys to connect to your pod from PowerShell. Open PowerShell and run:

```powershell
# Check if you already have a key
cat ~/.ssh/id_ed25519.pub
```

If that shows a key starting with `ssh-ed25519`, you're good — skip to copying it.

If you get an error ("file not found"), generate a new key:

```powershell
ssh-keygen -t ed25519
```

Press Enter for all prompts (default location, no passphrase is fine).

Now copy your public key:

```powershell
cat ~/.ssh/id_ed25519.pub
```

Copy the entire output. Go to RunPod → Settings → SSH Public Keys → paste it in and save.

### Step 4: Deploy a Pod

1. Go to RunPod → **GPU Cloud** → **Deploy**
2. Pick **A100-80GB** (required for 14B models)
3. Set **Container Disk** to **50 GB**
4. Set **Volume Disk** to **200 GB** (stores models + outputs persistently)
5. Under **Environment Variables**, add:
   - Variable name: `HF_TOKEN`
   - Variable value: your HuggingFace access token (get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
6. Click **Deploy**
7. Wait for the pod to show a green "Running" status

### Step 5: Connect via SSH

On your pod's page in RunPod, click **Connect** → copy the SSH command. It looks like:

```powershell
ssh root@<some-address> -p <port> -i ~/.ssh/id_ed25519
```

Paste that into PowerShell. Type `yes` if asked about the host fingerprint.

### Step 6: Upload Your Files

Open **Jupyter Lab** from the pod's Connect menu (it opens in your browser). Use the file browser on the left to:

1. Navigate to `/workspace/`
2. Upload these files by dragging them in:
   - `setup_runpod.sh` — the setup script
   - Your chosen training script from `runpod_py_templates.zip` (e.g., `train_runpod_i2v_lightning.py`)
   - `wan22-dataset-config-runpod.toml` — rename to `wan22-dataset-config-runpod.toml` on the pod (keep the same name)
3. Create folder `datasets/annika/` in Jupyter
4. Inside that, create `images/` and `videos/` folders
5. Upload your images (with matching `.txt` caption files) into `images/`
6. Upload your videos (with matching `.txt` caption files) into `videos/`

**Important:** Every image and video MUST have a matching `.txt` file with the same name. For example, `photo_001.png` needs `photo_001.txt` next to it. If captions are missing, training will silently skip those files.

### Step 7: Run the Setup Script

Back in your SSH terminal:

```bash
bash /workspace/setup_runpod.sh
```

This downloads all model weights (~60GB), installs musubi-tuner, and sets up the directory structure. Takes about 10-15 minutes depending on connection speed.

### Step 8: Fix Dataset Filenames

**Some filenames may contain spaces.** Linux commands and musubi-tuner can break on spaces in filenames. Fix them by replacing spaces with underscores:

```bash
cd /workspace/datasets/annika/images/
for f in *\ *; do mv "$f" "$(echo $f | tr ' ' '_')"; done
cd /workspace/datasets/annika/videos/
for f in *\ *; do mv "$f" "$(echo $f | tr ' ' '_')"; done
```

**Run this on EVERY new pod** — each pod gets its own copy of the dataset.

### Step 9: Delete Unneeded Models (Save Disk Space)

The setup script downloads all model variants. Delete the ones you don't need:

For **T2V training** — delete I2V models:
```bash
rm /workspace/models/wan2.2_i2v_high_noise_14B_fp16.safetensors
rm /workspace/models/wan2.2_i2v_low_noise_14B_fp16.safetensors
```

For **I2V training** — delete T2V models:
```bash
rm /workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors
rm /workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors
```

Check free space: `df -h /workspace`

### Step 10: Build the Caches

The training scripts can run caching automatically, BUT if caching fails silently (e.g., bad paths, missing files), training will crash with:

```
ValueError: No training items found in the dataset
```

**Run the caching steps manually first** so you can verify they work:

**Cache latents** (encodes your images and videos):
```bash
cd /workspace/musubi-tuner
python src/musubi_tuner/wan_cache_latents.py \
  --dataset_config /workspace/wan22-dataset-config-runpod.toml \
  --vae /workspace/models/wan_2.1_vae.safetensors \
  --batch_size 1
```

You should see encoding passes with item counts. If you see `0it` for any dataset, check paths and captions.

**Cache text encoder** (encodes your caption text):
```bash
python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
  --dataset_config /workspace/wan22-dataset-config-runpod.toml \
  --t5 /workspace/models/models_t5_umt5-xxl-enc-bf16.pth \
  --fp8_t5 --batch_size 1
```

Total caching time: ~7 minutes for a typical dataset.

### Step 11: Launch Training

Start a tmux session so training survives if your SSH connection drops:

```bash
tmux new -s train
```

Then launch training:

**Wan 2.1 (single run):**
```bash
python /workspace/train_runpod_wan21_i2v_720p.py
```

**Wan 2.2 (two runs — one per expert):**
```bash
python /workspace/train_runpod_i2v_lightning.py --noise_level high
python /workspace/train_runpod_i2v_lightning.py --noise_level low
```

You can override config from the command line:
```bash
python /workspace/train_runpod_i2v_lightning.py --noise_level high --lr 5e-5 --epochs 30 --dim 32
```

**tmux basics:**
- Detach (leave training running): press `Ctrl+B`, then `D`
- Reattach later: `tmux attach -t train`
- You can close PowerShell after detaching — training keeps running on the pod

---

## Downloading Checkpoints

When training finishes (or you want to grab intermediate checkpoints), download them to your PC.

**Option A: From Jupyter Lab** — open Jupyter from the pod page, navigate to `/workspace/outputs/`, right-click files → Download.

**Option B: Using SCP from PowerShell:**
```powershell
scp -P <port> -i ~/.ssh/id_ed25519 root@<address>:/workspace/outputs/*.safetensors ~/Downloads/
```

---

## Troubleshooting

### "ValueError: No training items found in the dataset"
The caches are missing or empty. Run the manual caching steps from Step 10. Common causes:
- **Path mismatch** — the TOML points to a different folder name or capitalization than your actual folders
- **Missing captions** — every image/video needs a `.txt` file with the same base name

### "File not found" or garbled filenames
Filenames with spaces cause issues. Run the space-fix commands from Step 8.

### Terminal disconnected mid-training
If you used tmux, training is still running. SSH back in and reattach:
```bash
tmux attach -t train
```

### Training seems stuck / GPU at 0%
For Lightning scripts, the LoRA merge runs on CPU before GPU training starts. This takes 5-10 minutes for a 14B model. Look for "Loading base DiT" in the logs.

### Disk full
Delete unneeded model files (Step 9) or old checkpoints. Check space with `df -h /workspace`.

---

## Cost Estimate

| GPU | RunPod Community Rate | Training Time (50 epochs) | Cost per Expert |
|---|---|---|---|
| A100-80GB | ~$1.64/hr | ~6-10 hours | ~$10-16 |

Total for both Wan 2.2 experts: **~$20-32**.
