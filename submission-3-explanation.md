# Submission 3 — SpecAugment and Mixup

This document explains the two data-augmentation techniques added for submission 3:
**SpecAugment** (applied at the data-loading stage) and **Mixup** (applied just
before the loss is computed). It assumes familiarity with the spectrogram and
multi-label BCE setup described in `submission-2-explanation.md`.

---

## Why augmentation?

A model that memorises the training examples performs well on training data but
poorly on new recordings. Augmentation deliberately corrupts or blends training
examples during each epoch so the model never sees exactly the same input twice.
This forces it to learn general patterns (shapes, relative pitch contours, timing
structure) rather than pixel-level details that are specific to one recording.

Both techniques below are **training-only**: they are disabled during validation
and inference, so they do not distort the evaluation metrics or submission
predictions.

---

## SpecAugment

### What it is

SpecAugment (Park et al., 2019) was designed specifically for audio represented
as spectrograms. It randomly blacks out rectangular strips of the spectrogram,
independently along each axis:

- **Frequency masking**: choose a random start row *f₀* and a random height *f*
  (up to `freq_mask_param` pixels), and set all values in rows *f₀* to *f₀ + f*
  to the mean value of the spectrogram. This hides a band of pitches.
- **Time masking**: choose a random start column *t₀* and a random width *t*
  (up to `time_mask_param` pixels), and set all values in columns *t₀* to
  *t₀ + t* to the mean. This hides a slice of time.

The masked region is filled with the global mean (not zero) so the model does
not learn to treat zero as a special "augmented" signal.

### Why it helps for bird calls

A common failure mode for audio classifiers is over-reliance on the exact
frequency range of a call in the training clips. If a species always lands in,
say, the 2–4 kHz band in training recordings, the model may key on that band
rather than on the shape of the call. By randomly hiding frequency bands, the
model is forced to use the full spectro-temporal pattern — the trill shape, the
rhythm, the harmonic spacing — rather than just which absolute pitches are
present.

Time masking similarly prevents the model from relying on a call always being
at a fixed position within the 5-second window.

### How it is applied here

```
batch_tfms=aug_transforms(...) + [SpecAugment(freq_mask_param=30, time_mask_param=30)]
```

SpecAugment is a **batch transform**: it runs on GPU after the batch has been
assembled, so it adds essentially zero CPU overhead. The mask widths are sampled
uniformly from 0 to 30 pixels (out of 224) on each axis for each image in the
batch. Because the minimum is 0, some images get no masking at all — this is
intentional and standard in the original paper.

---

## Mixup

### What it is

Mixup (Zhang et al., 2018) creates synthetic training examples by blending two
real examples:

1. For each training batch, create a shuffled copy of the batch.
2. Sample a blending weight λ from a Beta(α, α) distribution (α = 0.4 here).
   At α = 0.4, λ tends to be close to 0 or 1 rather than 0.5, so most blended
   examples are dominated by one of the two originals.
3. The new input is `λ · x₁ + (1−λ) · x₂` — a pixel-wise weighted average of
   two spectrograms.
4. The loss is computed twice — once against label set *y₁* and once against
   *y₂* — and linearly interpolated:
   `loss = λ · BCE(pred, y₁) + (1−λ) · BCE(pred, y₂)`

This avoids the need for soft labels: rather than constructing a mixed target
(which would require special handling with `pos_weight`), both original label
sets are used separately and the resulting losses are blended.

### Why it helps

Mixup regularises the model by penalising sharp, overconfident predictions. If
the model sees a spectrogram that is 70% species A and 30% species B, it should
output roughly 0.7 for A and 0.3 for B — not a hard 1.0 for A and 0.0 for B.
This pushes the model towards smoother, better-calibrated probability estimates,
which directly improves metrics like Average Precision that care about ranking
and calibration, not just thresholded correctness.

Mixup also acts as an implicit data multiplier: every batch contains novel
blended examples that the model will never see in exactly the same form twice,
reducing overfitting even when the total number of training images is fixed.

### How it is applied here

```python
learn = vision_learner(
    dls, resnet50, loss_func=loss_func, metrics=metrics,
    n_out=len(ALL_SPECIES),
    cbs=[MixUp(0.4)],
).to_fp16()
```

`MixUp(0.4)` is a fastai **callback** — it hooks into the training loop just
before each forward pass. The `0.4` is the α parameter of the Beta distribution.
It is automatically disabled during validation so validation metrics reflect true
single-example predictions.

---

## Interaction between SpecAugment and Mixup

The two techniques are applied in sequence:

1. **SpecAugment** runs on the raw batch (individual spectrograms are randomly
   masked).
2. **Mixup** then blends pairs of those already-augmented spectrograms.

So the model sees spectrograms that are simultaneously masked *and* blended with
another masked spectrogram — a highly distorted training signal that forces the
model to rely only on the most robust, reproducible features of each species.
