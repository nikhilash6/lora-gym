# Captioning Methodology & Prompt Design Guide

## For LoRA Training (Wan 2.2 / musubi-tuner and beyond)

---

## Table of Contents

- [Why Caption At All?](#why-caption-at-all)
- [The Anchor System](#the-anchor-system)
- [Natural Language Captions](#natural-language-captions)
- [What to Include vs. Exclude](#what-to-include-vs-exclude)
- [Adapting for Different LoRA Types](#adapting-for-different-lora-types)
- [Secondary Concepts](#secondary-concepts)
- [Practical Tips](#practical-tips)

---

## Why Caption At All?

Here is the single most important thing to understand:

**The model learns from your images and videos regardless of what you write in captions. Captions do not teach the model what things look like — your visual data does that. Captions tell the model which patterns to associate with which words.**

Think of it this way:

- Your training images/videos are the **textbook** — they contain all the visual information.
- Your captions are the **index** — they tell the model where to file that information so you can retrieve it later with a prompt.

If you train a character LoRA with 50 images of "annika" and never mention clothing in any caption, the model still *sees* the clothing in every image. It still learns those visual patterns. You just haven't given it a specific word to associate with "that blue jacket" vs "that red dress." The clothing becomes part of the general concept of "annika" — which is often exactly what you want.

This is why over-captioning is a real problem. Every word you include is a filing instruction. If you describe everything in every caption, you fragment the model's attention across dozens of associations instead of reinforcing the ones you actually care about.

---

## The Anchor System

Anchors are the words in your captions that you want the model to strongly bind visual patterns to. You are deliberately choosing what gets a label and what stays implicit.

### Why We Say "Anchor Word," Not "Trigger Word"

The old concept of a "trigger word" comes from Dreambooth-era training, where you needed a unique, invented token (like `sks` or `ohwx`) because the text encoders of that era had limited understanding of natural language. You had to create a blank token with no prior associations.

Modern video models use T5 and other advanced text encoders that understand natural language relationships between words. This changes everything:

- **You don't need invented words.** Real names, real descriptive terms, and common words all work — often better than invented ones, because the model already understands the semantic neighborhood around them.
- **The model learns context.** When you train with "annika" as your anchor, the model doesn't just learn a token — it learns how that word relates to actions, settings, and descriptions around it. Similar words can even become partially interchangeable after training (e.g., "drawing" and "illustration" might both activate a style LoRA trained with only one of them).
- **Captions should read like prompts.** Your captions should match the natural language style you'd use when prompting the model at inference. The closer your training captions are to how you'll actually prompt, the better the LoRA responds.

### Primary Anchor

This is your **anchor word** — the main concept you are training. It appears in every caption, used naturally in the sentence.

| LoRA Type | Primary Anchor | Example |
|-----------|---------------|---------|
| Character | Character name | `annika` |
| Style | Style descriptor | `illustration style` |
| Motion | Motion descriptor | `fluid walk` |
| Object | Object name | `crystal sword` |

The primary anchor should be:
- **Descriptive** — a word or short phrase that makes sense in natural language
- **Consistent** — spelled exactly the same way in every single caption
- **Natural** — used in the caption the way you'd use it in a prompt, not forced to the front

For characters, a name works great. For styles, use the general version of the style you're training ("illustration style," "cinematic noir," "watercolor painting") — not an invented token. For objects, just name the object.

### Secondary Anchors

These are traits you want to be able to **reference and control independently** in your prompts at inference time. You only tag these when you have a specific reason to.

Ask yourself: *"Will I need to call for this specific trait by name in my prompts?"*

- **Yes** → Tag it (secondary anchor)
- **No** → Leave it out (learned implicitly)

Example: You trained a character who wears three different outfits across your dataset. If you want to be able to prompt for "annika in the red dress" vs "annika in the leather jacket," then clothing becomes a secondary anchor and you tag it. If they always wear the same thing, don't bother — it becomes part of the primary anchor's identity.

### What Stays Unanchored

Everything you do NOT mention is still learned — it just becomes fused into the primary anchor's overall concept. This is a feature, not a bug. Unanchored traits:

- Are learned implicitly from the visual data
- Become inseparable from the primary concept
- Cannot be independently controlled at inference
- Reduce caption complexity and keep training focused

---

## Natural Language Captions

This is the biggest shift from older captioning guides: **your captions should read like natural sentences, the same way you'd write a prompt.**

### Anatomy of a Good Caption

A caption brings together your primary anchor, secondary anchors, and the world they exist in — all in natural language:

```
a closeup shot of annika walking down a cobblestone street, smiling and looking over her shoulder
```

Breaking this down:
- **Primary anchor:** `annika` (the character you're training)
- **Secondary anchors:** `closeup shot` (camera framing), `walking` (action), `cobblestone street` (setting context), `smiling` (expression)
- **Natural language glue:** "a ... of ... down a ... and looking over her shoulder" — reads like a prompt

The anchor word does NOT need to be the first word. It needs to fit naturally into the sentence. All of these are fine:

```
annika standing at the edge of a rooftop in a wide shot, wind blowing through her hair
a medium shot of annika sitting cross-legged on the floor, looking down at something in her hands
closeup of annika laughing, head tilted back, eyes closed
```

### Why Natural Language Matters

When you write captions that read like prompts, you're training the model to respond to the same kind of language you'll use at inference. The model learns:
- That "annika" is a person who does things in scenes
- That "closeup shot" and "wide shot" control framing
- That actions and settings can vary independently of the character

If you write rigid, formulaic captions (`annika standing in a full body view, straight on`), the model learns that rigid pattern. If you write natural captions (`annika standing at the edge of a rooftop in a wide shot`), the model learns to be flexible with how it interprets your prompts.

### Match Your Model's Prompting Style

Write captions the way successful prompts work for the base model you're training on. For Wan 2.2, that means simple, direct descriptions of what's happening. Don't overthink it — if a caption reads like something you'd type into a prompt box and get a good result, it's a good caption.

---

## What to Include vs. Exclude

### Always Include

| Element | Why |
|---------|-----|
| **Primary anchor word** | This is the concept you're training. Without it, the model has no label for what you're teaching it. |
| **Action / pose / motion** | Describes what is happening, giving the model a way to differentiate between images. Without this, every caption is nearly identical and the model has less signal to work with. |
| **Camera view** (full body, medium shot, close up, etc.) | Teaches the model to respond to framing prompts. Also helps differentiate images that show different amounts of the subject. |
| **Scene context** | Brief description of the world the anchors exist in — walking down a street, sitting in a cafe, standing on a rooftop. Gives the model natural language context. |

### Include When Relevant

| Element | When to Include |
|---------|----------------|
| **Camera angle** (from above, from below, three quarter view, etc.) | When perspective is notable or you want angle control at inference. |
| **Facial expression / emotion** | When the expression is notable or varies across your dataset. Gives you emotional control at inference. |
| **Movement quality** (speed, rhythm, flow) | For video data. Describes temporal information that differentiates clips. |
| **Hair / accessory motion** | For video data when there is visible secondary motion (hair bouncing, scarf flowing, etc.). |
| **Environmental interaction** | When the character is actively interacting with something — sitting on a bench, leaning against a wall, picking up an object. |

### Exclude (Usually)

| Element | Why Exclude |
|---------|-------------|
| **Character appearance** (eye color, face shape, skin tone) | The model sees this in every image. Describing it in captions fragments the model's attention and can actually *weaken* the character concept by treating stable traits as variable. |
| **Clothing colors and details** | Same reasoning — unless you have multiple outfits and want independent control over them. If the character always wears the same thing, the outfit IS the character to the model. |
| **Body type / build** | Stable physical traits should be learned implicitly, not captioned. |
| **Art style / rendering quality** | For character LoRAs, this adds noise. The style is learned from your visual data. (Exception: style LoRAs, where this IS the point.) |
| **Lighting descriptions** | Unless you specifically want lighting as a controllable secondary concept. |

---

## Adapting for Different LoRA Types

The anchor system stays the same. What changes is *what you anchor on* and *what you describe around it*.

---

### Character LoRA

**Goal:** Teach the model a specific character that you can place in any scene with an anchor word.

**Primary anchor:** Character name
**Described:** Actions, poses, camera framing, expressions, scene context
**Omitted:** Appearance, clothing (unless multiple outfits), art style

#### System Prompt Structure

```
You are a video/image captioning assistant for AI training data.

Write natural language captions that include the character name "{anchor_word}"
used naturally in the sentence — it does not need to be the first word.

Include:
- Camera framing (full body, medium shot, close up, wide shot, etc.)
- Physical actions, poses, gestures
- Brief scene context (where they are, what's around them)
- Facial expressions and emotions when notable
- For video: movement quality, direction, speed, secondary motion (hair, clothing)

Do NOT include:
- Character appearance, body type, physical features
- Clothing colors or descriptions
- Art style or rendering quality
- Detailed lighting descriptions

Write 1-3 sentences in natural language, as if writing a prompt for the model.
Write as a single continuous caption with no line breaks, bullet points, or labels.
```

#### Example Captions

```
a closeup shot of annika walking down a street, smiling and looking over her shoulder

annika sitting on a park bench in a medium shot, leaning forward with her elbows on her knees, looking pensive

wide shot of annika standing at the edge of a rooftop, wind blowing through her hair, arms at her sides
```

---

### Style LoRA

**Goal:** Teach the model a visual style or aesthetic that you can apply to any content.

**Primary anchor:** Style descriptor (use the general version — "illustration style," not an invented token)
**Described:** Scene subjects and composition (what's in the image), framing
**Omitted:** Art style descriptors (the pixels teach the style), specific subject identities

The key shift: you describe *what's in the scene* so the model can isolate the *style* as the common thread. The style anchor word tells the model "this visual treatment" — the scene descriptions tell it "applied to this content."

After training, the model often generalizes — similar terms can activate the LoRA. If you trained with "illustration style," prompting "drawing style" might also work, because the text encoder understands the semantic relationship.

#### System Prompt Structure

```
You are an image/video captioning assistant for AI training data.

Write natural language captions that include the style phrase "{anchor_phrase}"
used naturally in the sentence.

Include:
- What is depicted: subjects, objects, scene composition
- Camera framing if applicable
- Mood or atmosphere conveyed by the scene content
- For video: pacing, camera movement, transitions

Do NOT include:
- Art style, medium, texture, or rendering technique descriptions
- Words like "painting," "illustration," "rendered," "stylized"
- Specific identity of subjects (describe generically: "a woman," "a figure")
- Detailed lighting or color palette descriptions

Write 1-3 sentences in natural language. Write as a single continuous caption.
```

#### Example Captions

```
a woman sitting at a cafe table with a cup of coffee in illustration style, warm afternoon light through the window, medium shot

illustration style wide shot of a mountain landscape with a river in the foreground, clouds gathering above the peaks

a cat sleeping on a stack of books in a closeup, illustration style, soft shadows falling across the scene
```

---

### Motion LoRA

**Goal:** Teach the model a specific movement pattern or action sequence.

**Primary anchor:** Motion descriptor (use a descriptive phrase, not an invented token)
**Described:** Temporal progression, speed, rhythm, body mechanics, secondary motion
**Omitted:** Character identity, appearance, static visual qualities

The key shift: you are captioning *how things move through time*, not how they look in a frame.

#### System Prompt Structure

```
You are a video captioning assistant for AI training data focused on motion.

Write natural language captions that include the motion phrase "{anchor_phrase}"
used naturally in the sentence.

Include:
- Type of motion and its quality
- Speed and rhythm (slow, brisk, stuttering, smooth, accelerating)
- Body mechanics (weight shift, follow-through, anticipation, momentum)
- Timing and progression (how the motion begins, peaks, resolves)
- Secondary motion (hair sway, cloth bounce)
- Direction and spatial trajectory

Do NOT include:
- Character identity or appearance
- Clothing descriptions or colors
- Background or environment details
- Static visual qualities (lighting, color palette)

Write 1-3 sentences in natural language. Write as a single continuous caption.
```

#### Example Captions

```
a figure demonstrating a fluid walk with steady forward stride, smooth weight transfer from heel to toe, arms swinging in relaxed opposition

fluid walk gradually decelerating from walking pace to a stop, momentum carrying the upper body forward before settling back, loose fabric continuing to sway after the body stills

a slow turning fluid walk, inside foot pivoting as the outside foot crosses over, hips leading the rotation with shoulders and head following through naturally
```

---

### Object LoRA

**Goal:** Teach the model a specific object that you can place in any scene. Treated the same way as characters.

**Primary anchor:** Object name
**Described:** Context, placement, interaction, scene
**Omitted:** Detailed description of the object itself (the pixels handle that)

#### Example Captions

```
a crystal sword resting on a stone pedestal in a wide shot, light catching along the blade

closeup of a crystal sword held in two hands at chest height, blade pointing upward, glow reflecting on the holder's face

a crystal sword being swung in an arc from right to left, motion blur along the blade's path, medium shot from the side
```

---

### Quick Reference: What Changes Per LoRA Type

| | Character | Style | Motion | Object |
|---|-----------|-------|--------|--------|
| **Anchor** | Character name | Style descriptor | Motion descriptor | Object name |
| **Core description** | Actions, poses, scene | Scene subjects, composition | Temporal dynamics | Context, placement |
| **Camera info** | Yes | Yes (as composition) | Only if camera moves | Yes |
| **Appearance** | Omitted | Omitted | Omitted | Omitted |
| **Background/setting** | Brief context | As scene content | Omitted | Brief context |
| **Timing/movement** | Described for video | Pacing only | Primary focus | If relevant |

---

## Secondary Concepts

Secondary concepts are anything beyond your primary anchor that you choose to explicitly name. This section helps you decide when naming something is worth it.

### The Decision Framework

For any trait in your dataset, ask these three questions:

**1. Does this trait vary across my dataset?**
- If NO (same in every image) → Usually leave unanchored. It fuses with the primary concept automatically.
- If YES → Move to question 2.

**2. Do I want independent control over this trait at inference?**
- If NO → Leave unanchored. The model will still learn the variants; they will just associate loosely with the primary anchor.
- If YES → Move to question 3.

**3. Do I have enough examples of each variant?**
- If NO (only 2 images of the red dress out of 50) → Probably not worth tagging. The model needs repeated examples to form a strong association.
- If YES (roughly balanced representation) → Tag it as a secondary anchor.

### Common Secondary Concept Decisions

| Trait | Tag It When... | Skip It When... |
|-------|---------------|-----------------|
| **Outfit / clothing** | Character has 3+ distinct outfits with multiple examples each, and you want to prompt for specific ones | Character wears the same thing in every image, or outfits vary but you do not need to control which appears |
| **Hair style** | Character has distinct hairstyles (ponytail vs. down vs. braided) across the dataset | Hair looks the same throughout |
| **Accessories** | A specific accessory appears/disappears and you want that as a toggle | Accessory is always present (becomes part of identity) |
| **Color variants** | You are training something that comes in explicit color options you want to control | Color is consistent or unimportant |
| **Expression set** | You need fine control over specific expressions and have multiple examples of each | Expressions vary naturally and you are fine with general prompt control |

### How to Tag Secondary Concepts

Keep it simple and consistent. Use the same phrasing every time, woven into natural language:

**Good — consistent, natural:**
```
a medium shot of annika in the red dress, sitting on a bench and looking to the side
annika in the leather jacket leaning against a wall in a full body shot, arms crossed
wide shot of annika in the red dress walking down a hallway, heels clicking
```

**Bad — inconsistent phrasing:**
```
annika wearing her red dress, sitting on a bench...
annika in the leather jacket outfit, leaning against a wall...
annika with the crimson dress on, walking forward...
```

Pick exact phrasing and reuse it. "in the red dress" — every time. Not "wearing the red dress," not "in her red outfit." Consistency is how the model forms clean associations.

---

## Practical Tips

### Common Mistakes

**Over-captioning (the most common mistake)**
You describe everything in the image — the character, their clothes, the background, the lighting, the art style, the mood. The model tries to associate ALL of that with your anchor word, fragmenting its attention. Result: a weak, muddled concept that does not respond cleanly to prompts.

**Fix:** If in doubt, leave it out. Start minimal, add secondary concepts only when you have a clear reason.

**Inconsistent anchor word**
You write "annika" in some captions, "Annika" in others, "the character" in a few. The model sees these as different tokens.

**Fix:** Copy-paste your anchor word. Never retype it.

**Describing stable traits as if they vary**
Every caption says "annika with blue eyes and blonde hair" — but the character always has blue eyes and blonde hair. You are wasting caption space on information that does not help the model differentiate between images.

**Fix:** Only describe traits that change across your dataset and that you want to control.

**Writing formulaic captions instead of natural language**
Every caption follows the exact same rigid structure: "annika standing in a full body view, straight on, arms at their sides." This teaches the model a pattern, not a concept.

**Fix:** Write captions the way you'd write prompts. Vary sentence structure. "a wide shot of annika standing at the edge of a cliff" and "annika laughing in a closeup, head tilted back" are both good.

**Inconsistent detail level**
Some captions are one short sentence, others are a detailed paragraph. This creates uneven training signal.

**Fix:** Aim for 1-3 sentences consistently across your entire dataset. Same level of detail throughout.

### Under-Captioning vs. Over-Captioning

**Under-captioning** means your captions are too sparse to differentiate between images in your dataset. If every caption is just "annika" with no action or framing information, the model has no text signal to distinguish one image from another.

**Over-captioning** means your captions contain so many details that the model cannot figure out what matters. The anchor word competes with dozens of other described concepts for the model's attention budget.

**The sweet spot:** Your primary anchor + what is happening + brief scene context + how it's framed. That is usually enough. Add secondary anchors only when you have a reason.

A useful gut check: **If you removed a phrase from your caption, would you lose the ability to prompt for something you actually need?** If the answer is no, that phrase probably should not be there.

### Consistency Across Your Dataset

This matters more than most people think. Some rules:

- **Same anchor word spelling and capitalization in every caption.** No exceptions.
- **Same vocabulary for recurring concepts.** If you call it "closeup" in one caption, do not call it "close-up shot" in another.
- **Same level of detail.** Do not write three words for one image and three sentences for another.
- **Same inclusion/exclusion rules for every caption.** If you decide not to describe clothing, do not describe it in ANY caption. One inconsistency teaches the model to expect that word sometimes, creating noise.
- **Natural language throughout.** Don't mix formulaic captions with natural ones.

### The Captioning Checklist

Before finalizing your captions, run through this:

- [ ] Every caption includes the same anchor word, spelled consistently
- [ ] Anchor word is used naturally in the sentence (not forced to the front)
- [ ] Actions/poses are described (what is the subject doing?)
- [ ] Camera framing is mentioned (full body, medium shot, closeup, wide shot)
- [ ] Brief scene context is included (where are they, what's around them)
- [ ] Nothing is described that should be learned implicitly
- [ ] 1-3 sentences, not a paragraph
- [ ] Reads like a prompt you'd actually type
- [ ] Consistent vocabulary and detail level across all captions
- [ ] Secondary concepts (if any) use the exact same phrasing every time
- [ ] For video: movement, speed, rhythm, and secondary motion are noted

---

## Final Thought

Captions are not descriptions. They are **control surfaces**. Every word you write is a knob you are installing on your LoRA — a way to influence what the model generates at inference time. Only install the knobs you actually plan to turn. And write them the way you'd turn them — in natural language.
