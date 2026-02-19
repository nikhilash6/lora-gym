# Captioning Prompt Guide

How to write (and customize) the LLM captioning prompt for different LoRA training goals.

---

## Core Principle: Captions Are a Pattern Guide

The model **sees your pixels** — it already knows what's in the image/video. Captions don't teach the model what something *looks like*; they tell the model **what patterns to associate with your anchor word**.

Think of it this way:
- **Pixels** = the raw data the model learns from
- **Captions** = the index that tells the model how to file that data

If you caption every image with `annika walking`, the model learns that `annika` = this character + walking. If you caption with just `annika` and describe the *action*, the model learns that `annika` = this character (in any action).

**You name what you want to control at inference. You leave out what you want baked in.**

---

## A Note on Anchor Words vs. the Old "Unique Token" Approach

The older Dreambooth method recommended inventing nonsense tokens like `sks`, `xxy`, or made-up compound words like `watercolorx` or `spinmove`. The idea was to use a string the model had never seen, so it wouldn't conflict with existing knowledge.

Modern models with T5 text encoders don't need this. T5 understands natural language at a much deeper level, and natural descriptive words work better than invented tokens. Use real words:

- A person's name like `annika` (a name is already a relatively unique anchor)
- A style descriptor like `illustration style`
- A motion descriptor like `fluid walk`
- An object name like `crystal sword`

The model generalizes across similar words. After training, `drawing`, `illustration`, and `sketch` may all activate the same learned style because T5 encodes them as semantically related. This is a feature, not a bug. Lean into it.

---

## The Anchor + Secondary Concepts Model

Every caption has two layers:

### 1. Anchor (required)

The **anchor word** — the single concept you're teaching. It does not need to be at the start of every caption. Place it naturally in a sentence, the way you'd write a prompt.

```
a closeup shot of annika walking down a street, smiling and looking over her shoulder
```

The anchor is what you'll type at inference to activate your LoRA. Everything the model sees across your dataset gets associated with this word.

### 2. Secondary Concepts (optional, strategic)

Anything else you **name** in the caption becomes a concept the model can associate independently. This is powerful but has consequences:

| You tag it | What happens |
|---|---|
| Character's red hat | Model learns "red hat" as a separate toggleable concept. You can prompt with/without it. |
| Character's eye color | Model learns eye color as a separate concept. Might not appear unless you prompt for it. |
| Nothing about appearance | Model bakes ALL appearance into the anchor word. `annika` = the whole package, every time. |

**Rule of thumb:** Only name secondary concepts if you want to reference them independently at inference. If a trait should *always* appear with the anchor word, don't caption it — let the pixels teach it.

---

## Deciding What to Tag

Ask yourself for each visual element:

> "Do I want to be able to turn this on/off with a text prompt?"

- **Yes** → Tag it in captions. The model learns it as a separate, promptable concept.
- **No, it should always be there** → Don't mention it. The model bakes it into the anchor word.
- **It varies across my dataset** → Tag it so the model knows which images go with which variant.

### Example: Character Who Always Wears the Same Outfit

Your dataset has 50 images of `annika` always wearing the same jacket and boots.

**Option A — Don't tag clothing (recommended for most character LoRAs):**
```
a full body shot of annika standing with arms at her sides, looking straight ahead
a medium shot of annika sitting and looking down with a thoughtful expression, three quarter view
```
Result: `annika` = character + jacket + boots, always. Simple and reliable.

**Option B — Tag clothing (if you want outfit swaps):**
```
a full body shot of annika in a leather jacket and combat boots, standing with arms at her sides
a medium shot of annika in a leather jacket and combat boots, sitting and looking to the side
```
Result: You can now prompt `annika wearing a sundress` and the model *might* swap outfits. But the jacket/boots association is weaker since it's a named secondary concept rather than baked in.

### Example: Character With Varying Outfits in Dataset

Your dataset has the character in 3 different outfits across different images.

**Tag the outfits** — otherwise the model gets confused about which outfit is "default":
```
a full body shot of annika in her school uniform, walking forward
a medium shot of annika in the red dress, posing and looking toward the camera
a close up of annika in casual clothes, sitting with a relaxed expression
```

### Example: Character With Consistent Appearance But Varied Actions

This is the most common case — and it's what the default prompt in both scripts is built for:
```
a closeup shot of annika walking down a street, smiling and looking over her shoulder
a medium shot of annika sitting and turning to look behind her, hand resting on the chair back
a full body shot of annika jumping, hair lifting on the upswing, arms raised
```

Appearance is never mentioned → baked into anchor word.
Actions are always described → model learns to separate action from identity.

---

## Customizing the Prompt for Different LoRA Types

The `SYSTEM_PROMPT` / `SYSTEM_PROMPT_BASE` in the captioning scripts controls what the LLM describes. Here's how to adapt it for different training goals.

### Character Identity LoRA (default)

**Goal:** Teach the model a specific character that can be placed in any scene/action.

**What to tag:** Anchor word used naturally in a prompt-style sentence + action + camera framing. Nothing about appearance.

**What to omit:** Physical appearance, clothing (if consistent), art style, background.

This is the default prompt in both scripts. No changes needed.

```
RULES:
1. Every caption uses "{CHARACTER_NAME}" naturally in a sentence, as you would write a prompt
2. Include camera view and angle
3. Describe movements, poses, expressions, gestures
4. Do NOT describe appearance, clothing, body type, or features
5. Do NOT describe background or setting
6. Keep to 1-3 sentences
```

**Example captions:**
```
a medium shot of annika leaning against a wall, arms crossed with a relaxed expression, three quarter view
a full body side view of annika running, hair trailing behind with arms pumping
a close up of annika looking directly at the camera with a slight smile, straight on
```

### Style LoRA

**Goal:** Teach the model a visual art style (watercolor, anime cel shading, pixel art, etc.)

**What to tag:** Anchor word + **what's in the scene** (subjects, objects, composition). You want the model to learn that the *style* is the constant, not any particular subject.

**What to omit:** Any description of the art style itself — that's what the anchor word covers. The pixels teach the style; captions teach scene content.

**Prompt changes:**
```
RULES:
1. Every caption uses "{STYLE_NAME}" naturally in a sentence (e.g., "...in illustration style")
2. Describe what is depicted: subjects, objects, scene composition
3. Include camera framing if applicable
4. Do NOT describe the art style, medium, texture, or rendering technique
5. Do NOT use words like "painting", "illustration", "rendered", "stylized"
6. Keep to 1-3 sentences, direct language
```

**Example captions:**
```
a woman sitting at a cafe table with a cup of coffee in illustration style, warm afternoon light through the window
a mountain landscape with a river in the foreground in illustration style, wide shot, clouds gathering above the peaks
a cat sleeping on a stack of books in illustration style, close up, soft shadows
```

Notice: the style anchor word is placed naturally at the end of the scene description. The style name itself (e.g., "watercolor") is never used as a descriptive word. The model sees the style in the pixels and learns: this anchor = this rendering style applied to whatever the caption describes.

**Why describe scene content for style LoRAs:** The model needs to know what the *subject* is so it can isolate the *style* as the common thread. If you don't describe scenes, the model might associate the anchor with specific subjects ("cats and mountains") rather than the rendering style.

**A note on generalization:** After training, the model may respond to related words like `drawing style` or `sketch style` similarly to your anchor, because T5 understands these as semantically close. This is expected and useful behavior.

### Motion / Action LoRA

**Goal:** Teach the model a specific movement pattern or action sequence (a dance, a gesture, a transition).

**What to tag:** Anchor word + **detailed motion description** (body parts, timing, direction, speed). The motion is what you're teaching, so describe it precisely.

**What to omit:** Character appearance, background, and anything unrelated to the movement itself.

**Prompt changes:**
```
RULES:
1. Every caption uses "{MOTION_NAME}" naturally in a sentence as a descriptor
2. Describe the motion in sequential order: what happens first, then next
3. Name specific body parts and their movements (arms, legs, head, torso, hips)
4. Include timing cues: "slowly", "quickly", "pauses then", "suddenly"
5. Include direction: "to the left", "forward", "rotating clockwise"
6. Do NOT describe character appearance
7. Do NOT describe the setting
8. Keep to 2-4 sentences — motion needs more description than static poses
```

**Example captions:**
```
a figure demonstrating a fluid walk with steady forward stride, smooth weight transfer from heel to toe, arms swinging in natural opposition to the legs
a figure showing a fluid walk from the side, even pace with relaxed shoulders, slight lean forward on each step, head level throughout the movement
```

### Concept / Object LoRA

**Goal:** Teach the model a specific object, prop, or environmental element.

Objects are treated the same way as characters. Use the real descriptive name of the object as the anchor — there is no need to invent a unique token.

**What to tag:** Anchor word used naturally in a sentence + **context around the object** (where it is, what's near it, how it's being used). The model needs to isolate the object from its surroundings.

**What to omit:** Detailed description of the object itself — let the pixels handle that.

**Prompt changes:**
```
RULES:
1. Every caption uses "{OBJECT_NAME}" naturally in a sentence, as you would write a prompt
2. Describe where the object is and how it relates to the scene
3. Describe any interaction (being held, sitting on a shelf, floating)
4. Include camera framing
5. Do NOT describe the object's appearance, material, or texture in detail
6. Keep to 1-2 sentences
```

**Example captions:**
```
a crystal sword held in two hands at chest height in a medium shot, blade pointing upward, slight glow reflecting on the holder's face
a crystal sword resting on a stone pedestal in a wide shot, straight on, centered in the frame
a crystal sword being swung in an arc from right to left in a full body view, from the side, motion blur along the blade's path
```

---

## Common Mistakes

### Over-describing (most common)

Bad — tells the model things the pixels already show:
```
annika, a young woman with long brown hair and blue eyes wearing a white t-shirt and jeans, standing in a park with green trees and a blue sky, smiling happily
```

Good — guides the model to associate the anchor with action, not appearance:
```
a full body shot of annika standing in a park, smiling with hands in her pockets
```

### Under-describing

Bad — gives the model nothing to work with:
```
annika
```

Good — gives enough context to learn action/identity separation:
```
a full body shot of annika standing, relaxed posture, looking straight ahead
```

### Inconsistent anchor word

Bad — model doesn't know these are the same concept:
```
annika walking...
Annika walking...
the character annika walking...
she is walking...
```

Good — exact same anchor word, every time:
```
a full body shot of annika walking forward...
a side view of annika running...
a medium shot of annika sitting...
```

### Describing what you want baked in

Bad if you want the character to always have red hair:
```
a full body shot of annika with red hair standing and looking to the right...
```
This makes "red hair" a toggleable secondary concept. At inference, prompting just `annika` might not consistently produce red hair.

Good — don't mention it, and the model bakes it in from the pixels:
```
a full body shot of annika standing, looking to the right
```

---

## Quick Reference

| LoRA Type | Anchor | Describe | Omit |
|---|---|---|---|
| Character | person's name (e.g., `annika`) | actions, poses, expressions, camera | appearance, clothing*, background |
| Style | style descriptor (e.g., `illustration style`) | scene subjects, composition, camera | art style, medium, technique |
| Motion | motion descriptor (e.g., `fluid walk`) | body movements, timing, direction | appearance, background |
| Object | object name (e.g., `crystal sword`) | context, placement, interaction | object appearance/detail |

*Tag clothing only if it varies in your dataset or you want outfit control at inference.

The anchor word does not need to be at the start of the caption. Place it where it reads naturally, as you would write a prompt. The model learns from the association between the word and the pixels, regardless of word order.

---

## Editing the Prompt in the Scripts

Both `caption_gemini.py` and `caption_replicate.py` have the prompt near the top of the file.

**Gemini:** Edit `SYSTEM_PROMPT_BASE` (around line 63)
**Replicate:** Edit `SYSTEM_PROMPT` (around line 61)

Both also have separate `IMAGE_USER_PROMPT` and `VIDEO_USER_PROMPT` variables. These are short instruction suffixes — the system prompt does the heavy lifting.

To customize for your LoRA type:
1. Replace the RULES section with the rules from the relevant section above
2. Update the EXAMPLES to match your format
3. Change `CHARACTER_NAME` to your anchor word
4. Run on a few test files and spot-check the captions before doing the full dataset
