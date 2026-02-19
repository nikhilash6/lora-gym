# Captioning Run Guide

Two auto-captioning scripts for preparing LoRA training datasets. Both produce identical caption formats — they only differ in the API backend.

| | Gemini (`caption_gemini.py`) | Replicate (`caption_replicate.py`) |
|---|---|---|
| **Cost** | Free (20 RPM on free tier) | Paid (Replicate credits) |
| **Speed** | ~10s/file + rate limit waits | ~2-5s/file, minimal waits |
| **Model** | gemini-2.0-flash | gemini-2.5-flash (via Replicate) |
| **Rate limits** | Aggressive on free tier (429s) | Generous |
| **Best for** | Small datasets (<50 files) | Large datasets, time-sensitive work |
| **Video handling** | Uploads to Gemini Files API | Sends as base64 data URI |

**Recommendation:** Use Replicate for bulk captioning. Use Gemini for small batches or when you don't want to spend credits.

---

## File Organization

Both scripts expect your dataset in this layout:

```
your_project_folder/
├── caption_gemini.py          (or caption_replicate.py)
├── datasets/
│   └── your_character/
│       ├── images/
│       │   ├── photo_001.png
│       │   ├── photo_002.jpg
│       │   └── ...
│       └── videos/
│           ├── clip_001.mp4
│           ├── clip_002.mp4
│           └── ...
```

After captioning, each file gets a `.txt` companion:

```
│       ├── images/
│       │   ├── photo_001.png
│       │   ├── photo_001.txt    <-- generated caption
│       │   ├── photo_002.jpg
│       │   ├── photo_002.txt    <-- generated caption
```

The training framework (musubi-tuner) looks for these `.txt` files automatically.

**Supported formats:**
- Images: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`
- Videos: `.mp4`, `.webm`, `.mov`, `.avi`, `.mkv`

---

## Before You Run: What to Configure

Open the script in any text editor. The config block is at the top, clearly marked.

### 1. API Key / Token

**Gemini** — Get a free key at https://aistudio.google.com/apikey

```powershell
# PowerShell (recommended — keeps key out of the script)
$env:GEMINI_API_KEY = "AIzaSy..."

# Or edit line ~38 in the script directly
GEMINI_API_KEY = "AIzaSy..."
```

**Replicate** — Get your token at https://replicate.com/account/api-tokens

```powershell
$env:REPLICATE_API_TOKEN = "r8_your_token_here"
```

### 2. Dataset Paths

Edit these two lines to point at your images and videos folders:

```python
IMAGES_DIR = r"datasets\your_character\images"
VIDEOS_DIR = r"datasets\your_character\videos"
```

Paths can be relative (to where you run the script) or absolute.

### 3. Character Anchor Word

This is the most important config. The anchor word is embedded in every caption and becomes the word or name you use at inference to activate your LoRA.

```python
CHARACTER_NAME = "your_anchor_word"
```

**Rules for choosing an anchor word:**
- Use a real name or short descriptive phrase that naturally fits into a sentence
- It should be the same word or phrase across ALL your training data
- Good: `annika`, `sarah chen`, `the red knight`
- Bad: `girl`, `character`, `person` (too generic — the model already associates these with everything)

Captions don't need to start with the anchor word — they should read like natural prompts:
```
a closeup shot of annika walking down a street
a wide shot of sarah chen sitting at a desk, looking at the camera
```

### 4. Captioner Instructions (the LLM prompt)

The `SYSTEM_PROMPT_BASE` (Gemini) or `SYSTEM_PROMPT` (Replicate) tells the vision model what to describe and what to ignore. **This is where you control what the LoRA learns.**

The default prompt is designed for **character identity training** — it deliberately omits appearance description because the model learns appearance from the pixels. The captions guide what *patterns* the model should associate with the anchor word.

See [CAPTIONING_PROMPT_GUIDE.md](CAPTIONING_PROMPT_GUIDE.md) for how to customize this for different training goals (style LoRAs, motion LoRAs, concept LoRAs, etc.).

---

## Running the Scripts

### Gemini

```powershell
# Install dependencies (first time only)
python -m pip install google-generativeai Pillow

# Set API key
$env:GEMINI_API_KEY = "your-key-here"

# Run from the folder containing your datasets/ directory
cd C:\Musubi_Modal
python caption_gemini.py
```

**Note:** The `google.generativeai` package is deprecated. If it stops working, switch to `google.genai` (may require updating the import and API calls).

### Replicate

```powershell
# Install dependencies (first time only)
python -m pip install requests

# Set API token
$env:REPLICATE_API_TOKEN = "r8_your_token_here"

# Run
cd C:\Musubi_Modal
python caption_replicate.py
```

### What Happens During a Run

1. Script scans `IMAGES_DIR` and `VIDEOS_DIR` for supported files
2. Skips any file that already has a `.txt` caption (`SKIP_EXISTING = True`)
3. For each file:
   - Sends it to the vision model with the appropriate prompt (image vs video)
   - Receives a text caption back
   - Cleans the caption (strips quotes, ensures anchor word is present)
   - Saves as `filename.txt` next to the original file
4. Waits between requests to respect rate limits (10s for Gemini, 2s for Replicate)

### Handling Errors

- **Rate limit (429):** Automatic retry with increasing backoff (45s, 90s, 135s...). If the free Gemini tier is exhausted for the day, switch to Replicate or wait until the quota resets.
- **File already captioned:** Skipped automatically. Set `SKIP_EXISTING = False` to re-caption everything.
- **Video processing failed:** Gemini sometimes rejects very long or corrupt videos. Check the video plays locally, then retry.

### Re-captioning

To re-caption specific files, delete their `.txt` files and run the script again. It will only process files without captions.

To re-caption everything: set `SKIP_EXISTING = False` in the script.

---

## Example Output

**Image caption:**
```
a portrait shot of annika with a gentle smile, gaze directed slightly upward,
hands relaxed at their sides.
```

**Video caption:**
```
a wide shot of annika stepping into frame and slowing to a stop, then turning
their head from side to side with an observant expression.
```

---

## Quick Checklist

- [ ] API key/token set (env variable or hardcoded)
- [ ] `IMAGES_DIR` and `VIDEOS_DIR` point to your dataset
- [ ] `CHARACTER_NAME` set to your anchor word
- [ ] Dependencies installed (`pip install google-generativeai Pillow` or `pip install requests`)
- [ ] No filenames with spaces (rename them first — spaces break Linux training tools)
- [ ] Run script, verify first few captions look correct
- [ ] Spot-check 5-10 `.txt` files to make sure the anchor word is there and descriptions make sense
