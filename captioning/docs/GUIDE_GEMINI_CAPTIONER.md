# Gemini Auto-Captioner -- Run Guide

Auto-caption your images and videos for LoRA training using Google's Gemini API.
Script: `caption_gemini.py`

---

## 1. Prerequisites

### Install Python packages

```powershell
python -m pip install google-generativeai Pillow
```

That's it. Two packages:
- **google-generativeai** -- talks to the Gemini API
- **Pillow** -- loads images so Gemini can see them

> **Note:** You'll see a `FutureWarning` about `google.generativeai` being deprecated in favor of `google.genai`. This is cosmetic. The script works fine as-is. The package will need to be migrated eventually, but not today.

### Get a Gemini API key

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click "Create API key"
4. Copy the key somewhere safe

### Free tier limits

The free tier gives you **20 requests per minute**. The script handles this automatically:
- 10-second delay between requests
- Auto-retry on 429 (rate limit) errors with exponential backoff (45s, 90s, 135s...)
- You don't need to babysit it -- just let it run

For a dataset of ~100 images, expect roughly 20 minutes of runtime.

---

## 2. File Organization

The script expects your dataset in two folders: one for images, one for videos. You can use just one or both.

### Supported file types

| Type   | Extensions                          |
|--------|-------------------------------------|
| Images | `.png` `.jpg` `.jpeg` `.webp` `.bmp` |
| Videos | `.mp4` `.webm` `.mov` `.avi` `.mkv`  |

### How it works

The script creates a `.txt` file **next to each media file** with the same name. This is the standard format that LoRA training tools expect.

### Example folder structure

**Before captioning:**
```
datasets/
  annika/
    images/
      001.png
      002.jpg
      003.webp
    videos/
      walk_cycle.mp4
      wave.mp4
```

**After captioning:**
```
datasets/
  annika/
    images/
      001.png
      001.txt        <-- generated caption
      002.jpg
      002.txt        <-- generated caption
      003.webp
      003.txt        <-- generated caption
    videos/
      walk_cycle.mp4
      walk_cycle.txt  <-- generated caption
      wave.mp4
      wave.txt        <-- generated caption
```

Each `.txt` file contains a single caption like:
```
a closeup shot of annika walking down a street, arms swinging naturally with a slight bounce in their step
```

---

## 3. Configuration

Open `caption_gemini.py` and look for the `CONFIG` section near the top (around line 33). Here's each variable:

### GEMINI_API_KEY

**Recommended: set via environment variable** (so you don't accidentally commit your key):

```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

The script checks the `GEMINI_API_KEY` environment variable first, then falls back to whatever is hardcoded. If you're just running locally and don't care, you can paste it directly into the script:

```python
GEMINI_API_KEY = "your-key-here"
```

### IMAGES_DIR / VIDEOS_DIR

Paths to your dataset folders. Can be relative (to where you run the script) or absolute.

```python
IMAGES_DIR = r"datasets\annika\images"
VIDEOS_DIR = r"datasets\annika\videos"
```

If you only have images, just leave `VIDEOS_DIR` pointing to a non-existent path -- the script will warn but continue. Same in reverse.

### CHARACTER_NAME

Your anchor word — the name or phrase that identifies your character in every caption. Set it to whatever anchor word you'll use in training.

```python
CHARACTER_NAME = "annika"
```

The script checks that this word is present in every caption and inserts it if Gemini omits it.

### GEMINI_MODEL

Which Gemini model to use. Default is `gemini-2.0-flash` -- fast and good at vision tasks.

```python
GEMINI_MODEL = "gemini-2.0-flash"
```

You can swap this for `gemini-2.5-flash` or other models if available to you. Flash models are recommended for cost/speed.

### SKIP_EXISTING

When `True` (default), the script skips any file that already has a `.txt` next to it. This means you can **re-run the script safely** without re-captioning everything -- useful if it gets interrupted or you add new files.

```python
SKIP_EXISTING = True
```

Set to `False` if you want to regenerate all captions from scratch.

---

## 4. Customizing the Caption Prompt

The script has three prompt variables that control what Gemini writes. All are near the top of the script, starting around line 63.

### SYSTEM_PROMPT_BASE

This is the main instruction set that tells Gemini *how* to caption. It defines:
- The format (anchor word embedded naturally, camera view/angle described)
- What to include (poses, expressions, motion, interactions)
- What to exclude (appearance, clothing, background, art style)
- Example good and bad captions

This is set up for **character LoRA training** -- it deliberately avoids describing what the character looks like (since the model learns that from the images themselves).

### IMAGE_USER_PROMPT

Sent with each image. Tells Gemini to describe the pose, framing, and expression.

```python
IMAGE_USER_PROMPT = """Caption this image following the rules exactly. Describe the pose, framing, and any expression or gesture. Output ONLY the caption text, nothing else -- no quotes, no labels, no explanation."""
```

### VIDEO_USER_PROMPT

Sent with each video. Tells Gemini to focus on motion and actions over time.

```python
VIDEO_USER_PROMPT = """Caption this video following the rules exactly. Focus on the motion and actions happening over time. Output ONLY the caption text, nothing else -- no quotes, no labels, no explanation."""
```

### Adapting for different training goals

**For a character LoRA (default):**
Keep as-is. The prompt avoids describing appearance so the model learns the character visually from the training images.

**For a style LoRA:**
Modify `SYSTEM_PROMPT_BASE` to describe the art style instead. Remove the rule about not describing appearance, and add instructions to note stylistic elements (line weight, color palette, rendering technique).

**For a motion/animation LoRA:**
Expand `VIDEO_USER_PROMPT` to ask for more detail about timing, easing, acceleration, and frame-by-frame motion breakdown.

**General tips:**
- Always end user prompts with "Output ONLY the caption text, nothing else" -- Gemini loves to add preamble otherwise
- Keep captions short (1-3 sentences). Overly long captions hurt training more than they help.
- Test your prompt on a few files first before running on the full dataset

---

## 5. Running the Script

### Step 1: Set your API key

```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

### Step 2: Run the script

From the directory containing `caption_gemini.py`:

```powershell
python caption_gemini.py
```

### What you'll see

```
Skipping 12 already-captioned files
To caption: 45 images + 8 videos = 53 total
Character: annika
Model: gemini-2.0-flash
--------------------------------------------------

[1/53] [IMG] 001.png
  Captioning... done.
  >> a portrait shot of annika with a neutral expression, arms relaxed at their sides

[2/53] [VID] walk_cycle.mp4
  Uploading... done.
  >> a side view of annika walking forward, arms swinging naturally with hair bouncing...
```

The script will work through every file, then print a summary:

```
==================================================
Done! 53 captioned, 0 failed, 12 skipped
Image captions in: datasets\annika\images
Video captions in: datasets\annika\videos
```

### If you need to stop and resume

Just press `Ctrl+C` to stop. When you re-run, `SKIP_EXISTING = True` means it picks up where it left off (any file that already has a `.txt` gets skipped).

---

## 6. Troubleshooting

### 429 Rate Limit Errors

```
Rate limited. Waiting 45s... (attempt 1/5)
```

**This is expected on the free tier.** The script handles it automatically -- it waits and retries with increasing backoff (45s, 90s, 135s, 180s, 225s). You don't need to do anything. If it fails after 5 retries on a single file, it moves on and reports it as failed at the end.

### API Key Issues

```
ERROR: Gemini API key not set.
```

Make sure you either:
- Set `$env:GEMINI_API_KEY` in your current PowerShell session (it resets when you close the terminal), or
- Hardcode the key in the script

If the key is set but you get authentication errors, check that the key is valid at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey).

### Video Processing Failures

```
ERROR: Gemini rejected video: FAILED
```

Some things that can cause this:
- Video file is corrupted or uses an unusual codec
- Video is too long (try trimming to under 1 minute)
- Video file is too large (Gemini has upload limits)

Videos get uploaded to Gemini's Files API, processed server-side, then deleted after captioning. If upload fails, the error will usually tell you why.

### No Files Found

```
No images or videos found.
```

Check that:
- `IMAGES_DIR` / `VIDEOS_DIR` paths are correct
- The folders contain files with supported extensions
- Paths use raw strings (`r"path\to\folder"`) or forward slashes to avoid backslash escaping issues

### Windows Encoding Issues

If you see Unicode errors in the terminal output, the script already handles this -- it uses `>>` instead of Unicode arrow characters in print statements. If you still have issues, make sure your terminal supports UTF-8:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
```

### google.generativeai Deprecation Warning

```
FutureWarning: The google.generativeai SDK is deprecated. Use the google.genai SDK instead.
```

**Safe to ignore.** This is a cosmetic warning from the package. The script works fine. The package will eventually need to be migrated from `google.generativeai` to `google.genai`, but both work for now.
