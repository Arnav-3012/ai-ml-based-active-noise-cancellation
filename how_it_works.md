# How Our "ANC" Actually Works — A Beginner's Guide

*A plain-language explanation of what our system does, how it's different from
noise-cancelling headphones, and how the model actually learns to clean up speech.*

---

## 1. Wait — isn't this just noise-cancelling headphones?

No, and this is the single most important thing to understand before anything else.

Think about how noise-cancelling headphones actually work. A tiny microphone on the
headphone listens to the sound *around* you — say, an airplane engine's steady hum.
The headphones then generate a **new sound wave that's the exact mirror image** of
that hum, and play it into your ear at the same time. When a sound wave and its
mirror image meet, they physically cancel each other out — like two ripples in
water that happen to be perfectly opposite, so they flatten into nothing. This is
called **phase cancellation**, and it happens live, in the air, *before* the sound
reaches your eardrum. Nothing is ever recorded — it's a purely physical, acoustic
trick.

Our project **does not do this**. Here's why it can't:

Imagine a soldier speaking into a radio microphone while gunfire and engine noise
are happening around them. The microphone doesn't capture "the voice" and "the
noise" separately — it captures **one single mixed sound wave**, all glued together
into a single recording, the instant it happens. By the time we have that audio
file, the mixing has already occurred. There's no way to "cancel it before it
arrives," because it already arrived, mixed.

So instead of fighting sound waves in real time, our system does something
completely different: it takes that **one already-mixed recording** and asks a
much harder question — *"given this messy signal, can I figure out what the voice
part alone would have sounded like, and reconstruct it?"*

This is called **speech enhancement**, or more precisely, **source separation**.
It's a computational, data-driven problem — closer to how you might mentally
"tune out" background chatter at a party and focus on one voice, except we're
teaching a computer to do that automatically, on a recording, after the fact.

**The one-sentence version:** noise-cancelling headphones fight sound waves in the
air before they reach you; our system is a detective, examining a recording that's
already been contaminated, and reconstructing what the clean voice probably sounded
like.

---

## 2. So what does "the model" actually mean?

When we say "the model," we mean a large mathematical function with millions of
adjustable internal numbers (called **parameters** — in our case, about 18.9
million of them). At the start, before any training, these numbers are essentially
meaningless — set either randomly or, in our case, inherited from a model that was
already trained on *general* noisy speech (more on that below).

**Training** is the process of nudging all 18.9 million numbers, little by little,
so that when you feed the model a noisy recording, its output gets closer and
closer to what the true clean speech actually sounded like.

The model we use is called **Denoiser (`dns48`)**, built by Facebook Research. It
has a specific shape, which matters for understanding how it "thinks":

- **An encoder** — a stack of layers that take the raw noisy waveform and gradually
  compress it into a more abstract, compact representation. Think of this like
  slowly summarizing a long paragraph into a few key bullet points, except the
  "bullet points" here are learned patterns useful for telling voice apart from
  noise, not human-readable text.
- **An LSTM bottleneck** — a special layer, sitting at the most compressed point,
  designed to remember context *over time*. Speech isn't just about one instant —
  understanding a word depends on the sounds just before and after it. This layer
  is what lets the model use a bit of surrounding context to make better decisions,
  not just judge one tiny slice of audio in total isolation.
- **A decoder** — the mirror image of the encoder, which takes that compressed,
  cleaned-up representation and expands it back out into a full audio waveform —
  hopefully one that sounds like clean speech.
- **Skip connections** — direct shortcuts that let fine, detailed information from
  early in the encoder "skip past" the heavily-compressed bottleneck and reach the
  decoder directly. Without these, a lot of fine acoustic detail would get lost in
  the compression, and the output would sound blurry or muffled even if the general
  shape was right.

One more important design choice: this model is **causal**. That means every
output sample only ever depends on *past* audio, never future audio. This isn't an
accident — it's exactly what makes the architecture usable in a real-time
communication device eventually: a soldier talking into a radio can't wait for
audio that hasn't been spoken yet.

---

## 3. How does the model actually *learn* to tell voice from noise?

This is the heart of it, and it's a beautifully simple idea once you see it.

### Step 1 — We build our own "answer key"

We can't record real gunfire mixed with real speech, thousands of times, in a
controlled way — that's impractical and dangerous. So instead, we **synthetically
create** training examples ourselves:

1. Take a clip of clean, isolated human speech (from a public speech dataset).
2. Take a clip of real noise — a gunshot, engine rumble, static, wind, etc.
3. **Mathematically add them together** at a controlled loudness ratio (we can
   choose exactly how loud the noise is relative to the speech — this ratio is
   called the **Signal-to-Noise Ratio, or SNR**, measured in decibels).
4. Now we have a *pair*: the noisy mixed clip (what the model will see as its
   input), and the original clean speech clip (which we secretly keep as the
   correct answer).

We did this thousands of times, using different speech, different noise types
(gunfire, engines, general background noise), and different SNR levels — from
very hard (noise louder than speech) to fairly easy (speech clearly dominant).

### Step 2 — The model guesses, we check, we correct

For each training pair:

1. We feed the model **only the noisy clip** (it never gets to see the clean
   answer directly).
2. The model processes it through its encoder → LSTM → decoder pipeline and
   produces its own attempt at "what I think the clean speech sounds like."
3. We compare the model's guess to the *actual* clean speech we kept aside, using
   a mathematical measure of "how different are these two waveforms" (we call this
   the **loss** — lower is better, meaning the guess and the true answer are
   closer together).
4. Using a technique called **backpropagation**, the training process calculates
   exactly which of the model's 18.9 million internal numbers contributed to the
   mistake, and nudges each one slightly in the direction that would have made the
   guess a little more accurate.
5. Repeat this millions of times, across many different noisy examples.

Over time, the model isn't memorizing specific clips — it's gradually learning
**general patterns** that distinguish "speech-like" structure from "noise-like"
structure: things like the rhythm and frequency patterns typical of a human voice,
versus the sudden broadband burst typical of a gunshot, versus the steady hum
typical of an engine. It learns this the same way you'd get better at picking a
friend's voice out of a noisy room the more times you practiced it — except the
model does this through math and repetition rather than conscious effort.

### Step 3 — We don't train from nothing

We didn't start the model from random, blank numbers. Denoiser's `dns48` comes
**pre-trained** on a large amount of general noisy speech already — it already
knows the *general shape* of "how to separate voice from noise" before we ever
touch it. What we do is called **fine-tuning**: we continue training that already-
capable model, specifically on *our* defence-relevant noise (gunfire, military
vehicles), so it specializes further in exactly the noise types our problem cares
about, without forgetting what it already knew broadly. This is much faster and
more effective than training a brand-new model from scratch.

---

## 4. What does the model actually *output* — new audio, or a filter?

This is a subtle but important point. The model does **not** invent new sound from
nothing, the way an AI image generator might invent a picture. Instead, internally,
it predicts something closer to a **mask** — essentially a set of "volume knobs,"
one for each small piece of its internal representation of the audio, each set
somewhere between 0 and 1.

- A knob near **1** means "this part is mostly speech — keep it."
- A knob near **0** means "this part is mostly noise — suppress it."

The model multiplies its internal representation of the input by this mask, then
reconstructs a waveform from what's left. This matters for two reasons:

1. **It bounds what can go wrong.** Because the model can only *keep or suppress*
   information that was already present in the input, it can't hallucinate speech
   content that was never there. When the model fails, it tends to fail by making
   speech sound muffled, distorted, or overly quiet — not by inventing fake words.
2. **It's fundamentally different from true ANC's wave-cancellation trick.** True
   ANC generates a brand-new anti-wave. Our model never generates a new wave from
   scratch — it filters and reconstructs from what's already there.

---

## 5. Why is this so much harder than it sounds?

A few real, honest challenges we ran into (and you should be ready to talk about,
since a good judge will probe exactly these):

- **When noise is louder than the speech**, some information about the voice is
  genuinely, permanently gone — buried so deeply under the noise that no amount of
  cleverness can perfectly recover it, the same way you can't un-shred a piece of
  paper. Our model gets impressively better at these hard cases the more it trains,
  but there's a real, physical ceiling to how good "impossible-to-fully-recover"
  audio can become.
- **Gunfire specifically is a hard case for a structural reason.** Classical
  techniques (the kind used in older systems) generally assume noise stays fairly
  constant over time — they learn "what the background hum sounds like" and
  subtract that same pattern continuously. A gunshot is the opposite: it's a
  sudden, extremely loud burst that's gone almost instantly. Constant-noise
  assumptions completely break on this kind of noise, which is exactly why older,
  classical methods (which we tested ourselves as a comparison baseline) do
  noticeably worse specifically on gunfire — and it's exactly the kind of case a
  data-driven, learned model like ours is built to handle better, since it learns
  the *actual* statistical shape of a gunshot from real examples, rather than
  assuming anything about how noise behaves.

---

## 6. Putting it all together — the whole journey of one audio clip

1. A soldier speaks. Gunfire happens nearby. The mic captures both, mixed into one
   waveform — this is our starting point.
2. That waveform enters the model's **encoder**, which compresses it into an
   internal, abstract representation, layer by layer.
3. At the bottleneck, the **LSTM** uses surrounding context (what came just before
   this moment) to help judge what's speech and what's noise.
4. The model produces a **mask** — its best guess, per tiny slice of the
   representation, of "how much of this is speech vs. noise."
5. That mask is applied, suppressing the noise-heavy parts and preserving the
   speech-heavy parts.
6. The **decoder**, aided by **skip connections** carrying fine detail from early
   in the process, reconstructs a full audio waveform from what's left.
7. The output: an estimate of what the soldier's voice alone would have sounded
   like — noticeably cleaner than the original mixed recording, though not always
   perfect, especially on the very hardest, loudest-noise cases.

That's the whole pipeline — not a physical trick performed on sound waves in the
air, but a learned, data-driven reconstruction performed on a digital recording,
built by showing a neural network millions of examples of "noisy in, clean out"
until it generalized the pattern.
