<!-- Graduated from Notion: Quickstart: Training with Claude Code | Last sync: 2026-02-19 -->

# Training with Claude Code

Claude Code is Anthropic's command-line coding agent. Timothy and Minta developed this agentic training workflow together — Timothy architected the Claude Code integration approach and Minta did the bulk of the vibe coding to build out the reference documentation and script templates. Instead of manually editing config variables, running commands, and debugging errors yourself, you tell Claude Code what you want in plain English and it reads, writes, and executes everything for you.

This is particularly powerful for this pipeline because the training scripts have dozens of interdependent parameters where a single mismatch (wrong precision, wrong timestep boundary, wrong task flag) silently ruins a run. Claude Code validates all of it before launching.

## Prerequisites

**On your PC (both paths):**

1. **Node.js 18+** — Download from [nodejs.org](https://nodejs.org). Verify: `node --version`
2. **Claude Code** — Install globally: `npm install -g @anthropic-ai/claude-code`
3. **Anthropic API key** — From [console.anthropic.com](https://console.anthropic.com). Set it: `$env:ANTHROPIC_API_KEY = "sk-ant-..."`
4. **Your dataset ready** — `datasets/CharacterName/` with Images/ and Videos/ subfolders, every file captioned

**For Modal path additionally:** Modal CLI installed (`pip install modal`) and authenticated (`python -m modal setup`), HuggingFace secret created (`python -m modal secret create my-huggingface-secret HF_TOKEN=hf_xxx`)

**For RunPod path additionally:** RunPod account with credits, an A100-80GB pod deployed, SSH access configured

## The Reference Doc

The key to making Claude Code effective is giving it the right context. The `WAN_LORA_CLAUDE_CODE_REFERENCE.md` file contains every parameter, every platform procedure, every known bug, and the full validation checklist. When Claude Code reads this file, it knows exactly how to configure a training run correctly.

Place this file in your working directory alongside your training scripts.

## How Claude Code Helps

Instead of manually editing Python config blocks, you have conversations like:

```
you: Train annika on Wan 2.2 I2V with Lightning on Modal. Use dim 24 for
     high noise and dim 16 for low noise.

claude: [reads reference doc, updates OUTPUT_NAME, verifies fp16 precision,
        confirms I2V timestep boundary is 900 not 875, checks --i2v flag
        on latent caching, validates --preserve_distribution_shape is
        present, runs upload and dispatches training]
```

Or for debugging:

```
you: Training finished but outputs look hazy. Here's my config.

claude: [checks reference doc diagnostic table, identifies low-noise expert
        undertrained based on epoch count, suggests extending low-noise
        training with --network_weights resume from best checkpoint]
```

## Path A: Modal (Serverless)

Modal is the simpler path — everything runs as Python functions dispatched to cloud GPUs. No SSH, no tmux, no manual environment setup.

### Step 1: Open Claude Code in your project folder

```bash
cd C:\path\to\your\training\folder
claude
```

### Step 2: Tell Claude Code what to train

```
I want to train a Wan 2.2 T2V character LoRA for strawberryman on Modal.
Read WAN_LORA_CLAUDE_CODE_REFERENCE.md first.
My dataset is in datasets/strawberryman/.
Use the validated production config.
```

Claude Code will:
1. Read the reference doc to understand the full pipeline
2. Identify the correct script from the decision tree
3. Verify the dataset config TOML paths match your folder structure
4. Check that OUTPUT_NAME reflects your character
5. Run the validation checklist (fp16, discrete_flow_shift 3.0, timestep boundaries 875 for T2V, no --i2v since it's T2V, --preserve_distribution_shape present)
6. Upload the dataset
7. Dispatch training

### Step 3: Train both experts

For Wan 2.2, high-noise and low-noise experts are completely independent. **You can run both simultaneously** — same total GPU time, half the wall-clock time.

```bash
python -m modal run train_wan22_t2v.py::run_high
python -m modal run train_wan22_t2v.py::run_low
```

### Step 4: Download and evaluate

```
List what's on the kohya-volume and download the epoch 25 and 50
checkpoints for both experts.
```

### Step 5: Iterate

```
The low noise expert looks underfit at epoch 25 but overfit at epoch 50.
Can you resume from epoch 25 with a lower LR for 15 more epochs?
```

Claude Code reads the resume procedure from the reference doc, uploads the checkpoint, adjusts the LR, and dispatches.

## Path B: RunPod (Bare Metal)

### Option 1: Claude Code on your local PC (remote control)

Claude Code prepares files and generates commands locally. You paste the SSH/SCP commands manually — this is more reliable than automating SSH sessions.

```
I need to train Wan 2.2 T2V on RunPod with Lightning merge.
Read WAN_LORA_CLAUDE_CODE_REFERENCE.md.
Review train_runpod_t2v.py and verify it passes the validation checklist.
```

Claude Code will verify the script, check TOML paths, flag mismatches, and generate the exact SCP upload and training commands for your pod.

### Option 2: Claude Code directly on the pod (recommended)

Install Claude Code on the RunPod instance for full agentic control:

```bash
curl -fsSL https://nodejs.org/dist/v20.11.0/node-v20.11.0-linux-x64.tar.xz | tar -xJ -C /usr/local --strip-components=1
npm install -g @anthropic-ai/claude-code
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
cd /workspace
claude
```

Then:

```
Read WAN_LORA_CLAUDE_CODE_REFERENCE.md.
Train Wan 2.2 T2V high noise expert for strawberryman.
Dataset is in /workspace/datasets/strawberryman/.
Setup has already been run. Run caching manually first to verify, then train.
```

This is the most powerful setup because Claude Code can directly observe error output and react in real time.

## What Claude Code Validates

| Check | What it catches |
|---|---|
| Task flag vs model | `i2v-14B` used instead of `i2v-A14B` for Wan 2.2 (LoRA silently has no effect) |
| Precision | bf16 on Wan 2.2 (dtype mismatch crash) or fp8_scaled on Wan 2.1 (color corruption) |
| Timestep boundaries | 875 used for I2V instead of 900 (coverage gap) |
| I2V latent caching | Missing --i2v flag (KeyError at training time) |
| LR scheduler | --lr_scheduler_args used instead of --lr_scheduler_min_lr_ratio (LR silently decays to zero) |
| Alpha/LR relationship | Alpha changed without adjusting LR (most common failure) |
| TOML paths | Modal paths in a RunPod script or vice versa |
| Resume method | --resume used instead of --network_weights (LR spike that never recovers) |

## Tips

- **Always start with "Read WAN_LORA_CLAUDE_CODE_REFERENCE.md"** — this grounds Claude Code in Wan-specific requirements
- **Be specific** about what you want to change vs. keep
- **For debugging, paste the actual error output** — Claude Code matches it against the troubleshooting section
- **For new characters:** "Set up a new character called X. Create the dataset folder structure, update the TOML, update OUTPUT_NAME, and give me the captioning command."
- **Always tell Claude Code: "Use tmux for the training command"** when on a pod
- **Upload files manually before starting a Claude Code session** — let it focus on config validation, not file transfer
