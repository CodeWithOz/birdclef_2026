# Submission 2 — How It Works

This document explains the principles behind the model trained in
`birdclef_plus_2026_multilabel_attempt.ipynb` and the inference pipeline in
`birdclef-2026-submission-2.ipynb`. It assumes familiarity with SGD, loss
functions, ReLU, and simple neural nets built from linear layers and ReLU.

---

## The problem

BirdCLEF+ 2026 asks: given a one-minute audio recording from the wild, estimate
the probability that each of 234 species (birds, frogs, insect sonotypes) is
audible in every 5-second window of that recording. The output is one row per
5-second window, with 234 probability columns.

The hard part is that 28 of the 234 species have **no isolated training
recordings** — they only appear in multi-species continuous soundscape
recordings. A model trained purely on isolated calls would always output zero
for those 28 species.

---

## Part 1 — Training the model

### Turning audio into an image

Neural networks that work on images are among the most mature and well-studied
tools in ML. Rather than building a new architecture for raw audio, we convert
each 5-second audio clip into a **mel spectrogram** image, then treat the
problem as image classification.

Here is what that conversion does:

1. **Short-Time Fourier Transform (STFT)**: The audio waveform is sliced into
   overlapping windows (~16 ms each). On each window we run an FFT to find
   which frequencies are present and how loud they are. The result is a 2D
   grid: time along the horizontal axis, frequency along the vertical axis,
   brightness = loudness at that frequency at that moment.

2. **Mel frequency scale**: The raw FFT uses linearly-spaced frequencies, but
   hearing (animal and human alike) is roughly logarithmic. We remap the
   frequency axis to the *mel scale*, which compresses high frequencies and
   expands low ones. This makes species-distinctive call patterns — which often
   occupy a narrow but characteristic pitch range — more visually prominent.

3. **Decibels**: Loudness is converted to a log scale (decibels) so quiet
   background sounds and loud calls coexist in the same image without the
   loud sounds completely washing out the quiet ones.

4. **Per-clip normalisation**: Each spectrogram is rescaled so its quietest
   value maps to 0 and its loudest value maps to 255. This removes the effect
   of microphone gain and overall recording volume, so the model focuses on
   *relative* patterns rather than absolute loudness.

The result is a greyscale image that looks roughly like a heat-map of which
pitches are active at each moment in time. A bird with a rising trill appears
as a diagonal stripe; a drum-like insect appears as a bright horizontal band.

### The model architecture — ResNet-50

We use **ResNet-50**, a 50-layer convolutional neural network originally
designed for image classification on ImageNet (1000 everyday object categories).

**Convolutional layers vs plain linear layers**

A plain linear layer connects every input pixel to every output neuron — for a
224×224 image that is ~50,000 inputs, which quickly becomes unmanageable.
Convolutional layers instead apply a small *filter* (e.g. 3×3 pixels) that
slides across the image. Each filter position produces one output value as a
weighted sum of the 9 pixels under it. Using many different filters in parallel
detects many different local patterns (edges, textures, colour gradients) all
at once, with far fewer parameters than a fully connected layer.

Stacking convolutional layers — with ReLU activations between them — builds up
progressively more abstract features: early layers detect edges, middle layers
detect shapes, deep layers detect complex textures or object parts.

**Residual connections**

Very deep networks are hard to train because gradients shrink as they
backpropagate through many layers (the vanishing gradient problem). ResNet
solves this by adding *skip connections*: the input to a block is added
directly to the block's output before the next ReLU. This gives the gradient a
shortcut path all the way back to early layers, making 50-layer networks
trainable in practice.

**Transfer learning**

ResNet-50 was originally trained to classify 1000 ImageNet categories. Its
weights encode general visual features — edges, textures, shapes — that are
useful for almost any image task, including spectrograms. We keep those weights
as a starting point (the "backbone") and replace only the final linear layer,
which originally output 1000 class scores, with a new linear layer that outputs
234 scores (one per species).

This is transfer learning: borrow a backbone that already knows how to "see",
and teach it a new final decision layer for our specific task. It converges
faster and to a better solution than training from random weights, especially
when training data is limited.

### Multi-label classification

Standard image classification assigns exactly one label to each image. Here,
a 5-second clip from a soundscape can contain calls from **multiple species
simultaneously**. This is multi-label classification.

The key change vs. standard classification is in the **output layer and loss
function**:

- **Standard (single-label)**: the output passes through *softmax*, which
  forces all 234 scores to sum to 1, treating them as a probability
  distribution over mutually exclusive classes.

- **Multi-label**: the output passes through *sigmoid* independently for each
  of the 234 scores, producing 234 independent probabilities in [0, 1]. Each
  class is treated as its own yes/no question.

The loss function is **Binary Cross-Entropy (BCE)** summed across all 234
classes. For each class *c* and each training example, BCE compares the
predicted probability *p_c* to the true label *y_c* (1 if present, 0 if
absent):

```
BCE = −[ y_c · log(p_c) + (1 − y_c) · log(1 − p_c) ]
```

Predicting high confidence for a class that is absent, or low confidence for
a class that is present, both increase this loss. SGD adjusts the weights to
reduce it.

### Handling class imbalance — `pos_weight`

Of the ~103,000 training images, a species like a common bird might appear in
10,000, while a rare insect sonotype might appear in only 20. A naive model
quickly learns to always predict "absent" for the rare species — this gets the
loss very low on the 99.98% of examples where the species is absent, while
accepting high loss on the 0.02% where it is present.

To counteract this, each class *c* is assigned a `pos_weight`:

```
pos_weight[c] = (total examples − positive examples) / positive examples
```

clipped between 1 and 50. In BCE, this weight multiplies the loss term for
positive examples only, making misclassifying a rare species proportionally
more costly. The model is thus forced to pay attention to rare species even
though they appear infrequently.

### The two data sources

**train_audio** — ~103,000 spectrograms of isolated recordings, one species
per spectrogram. These come from Kaggle datasets pre-converted to PNG images
(one image per ~5-second audio clip), capped at 500 images per species so
no single common species overwhelms the training data.

**train_soundscapes** — ~740 spectrograms from continuous multi-species
recordings, with multi-label annotations (e.g. `"species_a;species_b"`). This
is the only source of training signal for the 28 species that have no isolated
recordings.

Both sources are combined into a single dataframe and fed to the model
together. The train/validation split is done at the *recording* level (all
clips from the same recording go to either train or validation, never both),
preventing the model from "cheating" by seeing clips from the same recording
in both splits.

### Training schedule

Training proceeds in two stages:

1. **`fine_tune(2)` — frozen backbone, 2 epochs**: The ResNet-50 backbone
   weights are frozen (their gradients zeroed). Only the new 234-output head
   is trained. This quickly adapts the head to the new task without disrupting
   the general visual features the backbone already knows.

2. **`fit_one_cycle(5)` — full network, 5 epochs**: All weights are unfrozen
   and trained together for 5 epochs using the *1-cycle learning rate
   schedule*. The learning rate starts low, rises to a peak in the middle of
   training, then falls back to a very low value. This approach often reaches
   better solutions than a fixed learning rate by allowing large updates early
   (to escape local minima) and small precision updates at the end.

**Mixed precision (fp16)**: All tensor operations use 16-bit floats instead
of the default 32-bit. This halves memory usage and roughly doubles compute
speed on modern GPUs, with negligible effect on final accuracy.

---

## Part 2 — Inference on test soundscapes

### The test data

At submission time, 600 one-minute ogg audio files are provided. The
competition expects one output row per 5-second non-overlapping window per
file — 12 rows per file × 600 files = 7,200 rows — each with 234 species
probabilities.

### Efficient spectrogram computation

Rather than computing the mel spectrogram 12 times (once per 5-second clip),
we compute it **once for the entire 1-minute file** and then slice out clip-
width windows. The result is mathematically identical to computing each clip
separately (because the FFT is a local operation with no global state), but
about 12× faster — critical for staying within the 90-minute CPU time limit.

### Overlapping windows

A bird call that happens to start 4 seconds into a 5-second clip is poorly
represented: only 1 second of call sits in the clip, and the model has to
classify based on that fragment. With non-overlapping clips, roughly half of
all calls are cut short in this way.

We address this with a **sliding window at 50% overlap**:

- Instead of clips at 0–5s, 5–10s, 10–15s, …
- We also compute clips at 2.5–7.5s, 7.5–12.5s, 12.5–17.5s, …

This gives ~23 overlapping clips per 1-minute file instead of 12.

For each required submission row (every 5 seconds), we aggregate the
predictions from the 2–3 overlapping clips that cover that window by taking
the **maximum** predicted probability per class. Taking the max rather than
the mean reflects the assumption that the model only needs to *detect* a
species once in the window — a single well-aligned clip where the call is
centred gives a confident prediction, and that confidence should win over
clips where the call is clipped at the edge.

### From spectrogram window to probability

Each spectrogram window goes through:

1. **Resize to 224×224** — matches the input size ResNet-50 was trained on.

2. **Convert to RGB** — replicate the single greyscale channel three times.
   ResNet-50 expects a 3-channel image; this satisfies that without any
   information loss.

3. **ImageNet normalisation** — subtract the ImageNet mean and divide by the
   ImageNet standard deviation, channel by channel. This keeps the input
   distribution consistent with what the backbone saw during its original
   ImageNet training.

4. **ResNet-50 forward pass** — produces 234 raw scores (logits).

5. **Sigmoid** — converts each logit to a probability in (0, 1). A logit of 0
   → probability 0.5; a large positive logit → probability near 1; a large
   negative logit → probability near 0.

These probabilities are written directly to `submission.csv`.
