# Replicate Auto-Captioner — Run Guide

Auto-caption images and videos for LoRA training using the Replicate API.
This script sends your media files to `google/gemini-2.5-flash` running on Replicate's infrastructure and saves structured captions as `.txt` files alongside each file.

**Key trade-off:** This is a PAID service. It costs Replicate credits, but it's significantly faster than the free Gemini captioner because there are no harsh rate limits.

---

## 1. Prerequisites

### Install dependencies

Just one package:

```powershell
python -m pip install requests
```

That's it. No Google SDK, no Replicate client library — just `requests`.

### Get a Replicate API token

1. Go to [https://replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)
2. Create a new token (starts with `r8_`)
3. Save it somewhere safe

### Pricing expectations

Replicate charges per prediction. Gemini Flash is one of the cheaper models on the platform. For a typical LoRA dataset (50-200 images, maybe a handful of videos), expect to spend somewhere in the range of a few cents to a couple dollars. Check [Replicate's pricing page](https://replicate.com/pricing) for current rates.

---

## 2. File Organization

### Expected folder structure

The script looks for two directories — one for images, one for videos. You can use both or just one.

**Before captioning:**

```
datasets\annika\
    images\
        img_001.png
        img_002.jpg
        img_003.webp
    videos\
        walk_cycle.mp4
        idle_anim.webm
```

**After captioning:**

```
datasets\annika\
    images\
        img_001.png
        img_001.txt      <-- caption for img_001.png
        img_002.jpg
        img_002.txt      <-- caption for img_002.jpg
        img_003.webp
        img_003.txt      <-- caption for img_003.webp
    videos\
        walk_cycle.mp4
        walk_cycle.txt   <-- caption for walk_cycle.mp4
        idle_anim.webm
        idle_anim.txt    <-- caption for idle_anim.webm
```

Each `.txt` file has the same name as its media file, just with a `.txt` extension. The caption is plain text containing your anchor word.

### Supported file types

- **Images:** `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`
- **Videos:** `.mp4`, `.webm`, `.mov`, `.avi`, `.mkv`

---

## 3. Configuration

Open `caption_replicate.py` and find the `CONFIG` section near the top (around line 33). Here's what each variable does:

### REPLICATE_API_TOKEN

**Do NOT hardcode your token in the script.** The script reads it from the environment variable `REPLICATE_API_TOKEN`. If you leave the placeholder `INSERT_TOKEN_HERE` in place, the script will refuse to run and tell you to set the env var.

Set it in PowerShell before running:

```powershell
$env:REPLICATE_API_TOKEN = "r8_your_token_here"
```

This keeps the token out of the script file itself, which matters when sharing or committing the script.

### IMAGES_DIR and VIDEOS_DIR

Paths to your dataset folders. Defaults:

```python
IMAGES_DIR = r"datasets\annika\images"
VIDEOS_DIR = r"datasets\annika\videos"
```

Change these to point to your actual dataset. If you only have images (no videos), you can leave `VIDEOS_DIR` pointing to a nonexistent path — the script will print a warning but keep going.

### CHARACTER_NAME

The anchor word — the name or phrase that identifies your character in every caption. Default: `"annika"`.

```python
CHARACTER_NAME = "annika"
```

Set this to whatever anchor word your LoRA uses. The script checks that this word appears in every caption and inserts it if the model omits it.

### REPLICATE_MODEL

Which model to run on Replicate. Default: `"google/gemini-2.5-flash"`.

You probably don't need to change this unless a newer model comes out or you want to experiment.

### SKIP_EXISTING

Default: `True`. If a `.txt` file already exists for a media file, skip it. Set to `False` to regenerate all captions from scratch.

---

## 4. Customizing the Caption Prompt

The script has three prompt variables that control what the model outputs. All are near the top of the file, right after the config section.

### SYSTEM_PROMPT (line ~61)

The main instruction set. Defines the captioning rules: anchor word usage, camera angles, what to include, what to exclude. This is shared between images and videos.

**Modify this if you want to change the overall captioning style.** For example:

- **Character LoRA (default):** Focuses on pose, expression, and motion. Explicitly excludes appearance descriptions (because the model already knows what the character looks like).
- **Style LoRA:** You'd want to include descriptions of art style, colors, textures. Remove rules 5-6 and add instructions to describe visual style.
- **Motion LoRA:** Emphasize temporal descriptions, speed, easing, transitions between poses.

### IMAGE_USER_PROMPT (line ~89)

Short instruction appended for image files. Tells the model to focus on pose, framing, and expression.

### VIDEO_USER_PROMPT (line ~87)

Short instruction appended for video files. Tells the model to focus on motion and actions over time.

These are intentionally brief — the heavy lifting is in the system prompt. You generally only need to touch these if you want to shift the focus (e.g., "describe the background in detail" for a scenery LoRA).

---

## 5. Running the Script

### Step by step

```powershell
# 1. Set your API token for this session
$env:REPLICATE_API_TOKEN = "r8_your_token_here"

# 2. Navigate to the project directory
cd C:\Musubi_Modal

# 3. Run the script
python caption_replicate.py
```

### What to expect

The script will:

1. Verify your API token against Replicate's account endpoint
2. Scan your image and video directories
3. Skip already-captioned files (if `SKIP_EXISTING = True`)
4. Print a summary: how many images, videos, and total to caption
5. Process each file one by one, showing progress like:

```
[1/47] [IMG] img_001.png
  Captioning... done.
  >> a portrait shot of annika with a relaxed posture, arms at their sides...

[2/47] [VID] walk_cycle.mp4
  Captioning... done.
  >> a wide shot of annika walking forward, arms swinging naturally...
```

6. Print a final summary with counts of captioned, failed, and skipped files

### Re-running

If the script gets interrupted (network issue, you close the terminal, etc.), just run it again. With `SKIP_EXISTING = True`, it picks up where it left off — already-captioned files are skipped.

---

## 6. Gemini (Free) vs Replicate (Paid)

Both use the same model family (Gemini Flash) and produce identical quality captions. The difference is infrastructure.

| | Gemini (free) | Replicate (paid) |
|---|---|---|
| **Cost** | Free | Costs Replicate credits |
| **Rate limits** | Aggressive — 10s+ delays, frequent 429 retries on large datasets | Minimal — 2s delay between requests, no rate limit issues |
| **Speed for 100 files** | Could take 30+ minutes with retries | ~5-10 minutes |
| **Dependencies** | `google-generativeai` SDK | Just `requests` |
| **Setup complexity** | Google API key + SDK install | Replicate token + `requests` |
| **Best for** | Small datasets, budget-conscious | Larger datasets, when you value your time |

**Recommendation:** Use Gemini for small test runs (under 20 files). Use Replicate when you have a real dataset and want it done fast.

---

## 7. Troubleshooting

### "ERROR: Replicate API token not set"

You didn't set the environment variable, or you accidentally left the placeholder in the script.

```powershell
# Make sure you set it in the SAME PowerShell session you run the script from
$env:REPLICATE_API_TOKEN = "r8_your_token_here"
```

Note: `$env:` variables only last for the current PowerShell session. If you close the window, you need to set it again.

### "ERROR: Invalid Replicate API token" (401)

Your token is wrong, expired, or revoked. Go to [https://replicate.com/account/api-tokens](https://replicate.com/account/api-tokens) and generate a new one.

### 422 errors

This means the model's input schema doesn't match what the script sent. The script handles this automatically by retrying with an alternate field name (`media` vs `image`). If you still see 422s after the fallback, the model's API may have changed — check the model page on Replicate for current input fields.

### Timeouts on large videos

Files are sent as base64 data URIs inline in the request body. Large videos (100MB+) will produce enormous payloads and may hit the 120-second request timeout. Options:

- Trim or compress your videos before captioning
- Reduce video resolution (the model doesn't need 4K to write a caption)
- For very large files, consider uploading to Replicate's file API first (not currently implemented in the script)

### Windows encoding errors

If you see Unicode-related errors in the terminal output, this was a known issue that's been fixed. The script uses `>>` instead of Unicode arrow characters in print statements. If you're running an older copy of the script, grab the latest version.

### "No images or videos found"

Double-check your `IMAGES_DIR` and `VIDEOS_DIR` paths. They're relative to wherever you run the script from. If you `cd` to a different directory, the relative paths will break. Either:

- Always run from `C:\Musubi_Modal`, or
- Change the paths to absolute paths like `r"C:\Musubi_Modal\datasets\annika\images"`

### Retries and backoff

The script retries each file up to 3 times on failure, with 15-second incremental backoff (15s, 30s, 45s). If a file fails all 3 attempts, it's logged as failed and the script moves on. You can re-run the script afterward to retry just the failed files (since successful ones get skipped via `SKIP_EXISTING`).
