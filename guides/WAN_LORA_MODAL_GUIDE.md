# Modal Training Guide — Wan LoRA Training with Musubi-Tuner

## What This Guide Covers

Step-by-step instructions for training Wan 2.1 and 2.2 LoRA models on Modal (cloud GPU) using musubi-tuner. Covers picking the right template script, setting up your dataset, running training, and downloading checkpoints.

---

## Step 1: Pick Your Template Script

Each script is a self-contained Modal training template. Pick the one that matches your model and workflow. All scripts are in `modal_py_templates.zip`.

### Wan 2.1 (Single Model — one training run)

| Script | Model | Use Case |
|---|---|---|
| `train_wan21_t2v_14b.py` | Wan 2.1 T2V 14B | Text-to-video, full quality |
| `train_wan21_t2v_1_3b.py` | Wan 2.1 T2V 1.3B | Text-to-video, lightweight/cheap prototyping |
| `train_wan21_i2v_480p.py` | Wan 2.1 I2V 14B 480p | Image-to-video, lower res |
| `train_wan21_i2v_720p.py` | Wan 2.1 I2V 14B 720p | Image-to-video, full quality |

### Wan 2.2 (Dual Experts — train high + low separately)

| Script | Model | Use Case |
|---|---|---|
| `train_wan22_t2v_vanilla.py` | Wan 2.2 T2V A14B | T2V, standard 30-50 step workflows |
| `train_wan22_t2v_lightning.py` | Wan 2.2 T2V A14B | T2V, optimized for Lightning speed LoRA |
| `train_wan22_i2v_vanilla.py` | Wan 2.2 I2V A14B | I2V, standard 30-50 step workflows |
| `train_wan22_i2v_lightning.py` | Wan 2.2 I2V A14B | I2V, optimized for Lightning speed LoRA |

**Vanilla vs Lightning:** Vanilla trains against the stock model weights — use if you don't plan to use a Lightning/speed-up LoRA at inference. Lightning merges a speed LoRA into the base weights before training, so your character LoRA is tuned specifically for the Lightning denoising path. At inference, load Lightning + your character LoRA together.

**Key rule:** The model you train on MUST match the model you inference with in ComfyUI. A LoRA trained on T2V won't work correctly on I2V, and vice versa.

---

## Step 2: Prepare Your Dataset

### Folder Structure

```
C:\Musubi_Modal\
├── train_wan21_i2v_720p.py      (or whichever template you chose)
├── wan21-dataset-config-modal.toml  (dataset config — MUST be in same dir as script)
└── datasets/
    └── annika/                   (your dataset folder name)
        ├── images/
        │   ├── 000.png
        │   ├── 000.txt           (caption for 000.png)
        │   └── ...
        └── videos/
            ├── 001.mp4
            ├── 001.txt           (caption for 001.mp4)
            └── ...
```

Every image and video **MUST** have a matching `.txt` caption file with the same base name.

### Update the TOML Config

Edit `wan21-dataset-config-modal.toml` to point to your dataset folder. Change the folder name in all paths:

```toml
[general]
caption_extension = ".txt"
enable_bucket = true
bucket_no_upscale = true

# --- Still images ---
[[datasets]]
image_directory = "/datasets/datasets/annika/images"
cache_directory = "/datasets/datasets/annika/images/cache"
resolution = [720, 1280]
batch_size = 1
num_repeats = 1

# --- Short video clips (high-res, 21 frames) ---
[[datasets]]
video_directory = "/datasets/datasets/annika/videos"
cache_directory = "/datasets/datasets/annika/videos/cache_1"
resolution = [1280, 720]
batch_size = 1
num_repeats = 1
frame_extraction = "head"
target_frames = [1, 21]

# --- Longer video clips (medium-res, 45 frames) ---
[[datasets]]
video_directory = "/datasets/datasets/annika/videos"
cache_directory = "/datasets/datasets/annika/videos/cache_2"
resolution = [960, 544]
batch_size = 1
num_repeats = 1
frame_extraction = "uniform"
target_frames = [45]
frame_sample = 2
```

### Video Prep

Convert all videos to 16fps before adding them to the dataset:

```bash
ffmpeg -i input.mp4 -r 16 -c:v libx264 -crf 18 output_16fps.mp4
```

Frame counts should follow the 4N+1 pattern (9, 13, 17, 21, 33, 41, 49, 65, 81) due to Wan's temporal architecture.

---

## Step 3: Configure the Training Script

Open your chosen template and edit the config block at the top:

### Wan 2.1 scripts

```python
OUTPUT_NAME         = "my-character-wan21-i2v-720p"
LEARNING_RATE       = "8e-5"
LR_SCHEDULER        = "polynomial"
LR_SCHEDULER_POWER  = "2"
MIN_LR_RATIO        = "0.01"
OPTIMIZER           = "adamw8bit"
NETWORK_DIM         = "24"
NETWORK_ALPHA       = "24"
MAX_EPOCHS          = "50"
SAVE_EVERY          = "5"
SEED                = "42"
DISCRETE_FLOW_SHIFT = "5.0"      # 5.0 for I2V, 3.0 for T2V
```

### Wan 2.2 scripts (vanilla)

These have per-expert configs for high and low noise:

```python
OUTPUT_NAME = "my-character-wan22-i2v-vanilla"

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
```

### Wan 2.2 scripts (Lightning)

Same per-expert config, plus a Lightning LoRA section:

```python
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
LIGHTNING_MERGE_STRENGTH = 1.0
```

---

## Step 4: Set Up Modal (One-Time)

If you haven't already:

1. Install Modal: `pip install modal`
2. Authenticate: `python -m modal setup`
3. Create a HuggingFace secret:
   ```bash
   python -m modal secret create my-huggingface-secret HF_TOKEN=hf_xxxxx
   ```

---

## Step 5: Upload Your Dataset

```bash
python -m modal run <your-script>.py::add_to_volume_datasets
```

This uploads `datasets/` and the dataset config TOML to Modal's persistent volume. Only needed once per dataset change.

To re-upload after changing your dataset, clear the old cache first:

```bash
python -m modal volume rm datasets -r /datasets/datasets/annika
python -m modal run <your-script>.py::add_to_volume_datasets
```

---

## Step 6: Launch Training

### Wan 2.1 (single run)

```bash
python -m modal run train_wan21_i2v_720p.py::run
```

### Wan 2.2 (two runs — one per expert)

```bash
# Train high-noise expert
python -m modal run train_wan22_i2v_lightning.py::run_high

# Train low-noise expert
python -m modal run train_wan22_i2v_lightning.py::run_low
```

Both produce separate LoRA files. Load both in ComfyUI at inference.

### Resume from a Checkpoint (Lightning scripts)

```bash
# Upload the checkpoint
python -m modal run train_wan22_i2v_lightning.py::upload_resume_checkpoint --checkpoint path/to/lora.safetensors

# Train — script auto-detects the checkpoint
python -m modal run train_wan22_i2v_lightning.py::run_high
```

---

## Step 7: Download Results

```bash
# List outputs
modal volume ls kohya-volume /outputs/

# List a specific run
modal volume ls kohya-volume /outputs/my-character-wan22-i2v-lightning/

# Download everything from a run
modal volume get kohya-volume /outputs/my-character-wan22-i2v-lightning/ ./local-output/
```

Place `.safetensors` files in `ComfyUI/models/loras/` for inference.

---

## Step 8: Test in ComfyUI

1. Place the trained `.safetensors` in `ComfyUI/models/loras/`
2. Use a LoRA Loader node connected between diffusion model and sampler
3. Test at strength 0.5, 0.75, and 1.0 to find the sweet spot
4. Make sure your inference model matches what you trained on (I2V LoRA needs I2V diffusion model)
5. For Wan 2.2: load both high-noise and low-noise LoRAs
6. For Lightning-trained LoRAs: also load the Lightning LoRA alongside your character LoRA

---

## Troubleshooting

### "ValueError: No training items found in the dataset"

Caches are missing or empty. Common causes:
- **Path mismatch** — the TOML points to a different folder name or capitalization than what's on the volume
- **Missing captions** — every image/video needs a `.txt` file with the same base name
- **Dataset not uploaded** — re-run the `add_to_volume_datasets` entrypoint

### Training seems stuck / GPU at 0%

For Lightning scripts, the LoRA merge runs on CPU before GPU training starts. This takes 5-10 minutes for a 14B model. Look for "Loading base DiT" in the logs.

### Disk/volume full

Clear old outputs: `modal volume rm kohya-volume -r /outputs/old-run-name/`

Clear old dataset caches: `modal volume rm datasets -r /datasets/datasets/annika`

---

## Quick Reference — All Commands

```bash
# Upload dataset
python -m modal run <script>.py::add_to_volume_datasets

# Train (Wan 2.1 — single run)
python -m modal run <script>.py::run

# Train (Wan 2.2 — both experts)
python -m modal run <script>.py::run_high
python -m modal run <script>.py::run_low

# Upload resume checkpoint (Lightning scripts)
python -m modal run <script>.py::upload_resume_checkpoint --checkpoint path/to/lora.safetensors

# Check resume checkpoint
modal volume ls kohya-volume /resume_checkpoints/

# Clear resume checkpoint
modal volume rm kohya-volume -r /resume_checkpoints/

# Download results
modal volume ls kohya-volume /outputs/
modal volume get kohya-volume /outputs/<run-name>/ ./local-output/

# Clear old data
modal volume rm datasets -r /datasets/datasets/<folder>
modal volume rm kohya-volume -r /outputs/<run-name>/
```
