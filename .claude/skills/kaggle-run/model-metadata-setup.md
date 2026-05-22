# Model Metadata Setup

When creating a **new model variation** on Kaggle (Step 5 of the skill, new variation
path), a `model-instance-metadata.json` file must exist in `./tmp/kaggle-output/`
alongside the downloaded model files before running `kaggle models variations create`.

Unlike `kernel-metadata.json`, this file does not need to be committed to the repo —
it is only needed transiently in the tmp output folder at upload time.

## Generating the metadata file

```bash
kaggle models instances init -p ./tmp/kaggle-output/
```

This creates a template. Edit it before running the create command.

## Format

```json
{
  "ownerSlug": "<owner>",
  "modelSlug": "bird-clef-classifier",
  "framework": "pytorch",
  "overview": "Brief description of what this variation does differently",
  "usage": "How to load and use this model variation"
}
```

## Key fields

- `ownerSlug`: the Kaggle username of the model owner
- `modelSlug`: always `bird-clef-classifier` — do not change this when adding a
  new variation; it refers to the parent model, not the variation
- `framework`: always `pytorch` for fastai models
- `overview`: describe what architectural change this variation represents

## Variation slug

The variation slug is derived from the metadata and becomes a permanent part of the
Kaggle model path:

```
/kaggle/input/models/<owner>/bird-clef-classifier/pytorch/<variation-slug>/<version>/
```

Choose a slug that is:
- Lowercase, hyphen-separated (e.g. `multilabel-234-v2`, `resnet-baseline`)
- Descriptive of the architecture, not the version number
- Confirmed with the user before creating, since it cannot be changed after creation

The confirmed slug should have already been established in Step 0 of the skill.
