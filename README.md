# BirdCLEF+ 2026 Workspace

This repository contains local experiments, preprocessing scripts, and notebook workflows for the [BirdCLEF+ 2026 Kaggle competition](https://www.kaggle.com/competitions/birdclef-2026).

The current workflow focuses on:
- generating mel-spectrogram PNG datasets from competition audio,
- packaging ranked species batches into zip archives for Kaggle dataset upload,
- training image models (fastai/ResNet baseline) on spectrograms.

## Competition links

- Competition home: [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)
- Data page: [Competition data](https://www.kaggle.com/competitions/birdclef-2026/data)
- Overview: [Problem statement and context](https://www.kaggle.com/competitions/birdclef-2026/overview)
- Rules: [Submission and usage rules](https://www.kaggle.com/competitions/birdclef-2026/rules)
- Leaderboard: [Public standings](https://www.kaggle.com/competitions/birdclef-2026/leaderboard)
- Code tab: [Community notebooks](https://www.kaggle.com/competitions/birdclef-2026/code)
- Discussion tab: [Q&A and competition updates](https://www.kaggle.com/competitions/birdclef-2026/discussion)

## Repo contents

- `scripts/generate_spectrogram_batches.py`  
  Generates spectrogram images from `train_audio`, batches species by frequency rank, zips each batch, and writes manifest files.
- `artifacts/spectrogram_batches/`  
  Output folder for generated batch zips and manifests.
- `birdclef_plus_2026_sound_classification_attempt_1.ipynb`  
  Main Kaggle-oriented training notebook (spectrogram image classifier experiments).
- `data/birdclef-2026/`  
  Local extracted competition files (`train.csv`, `train_audio/`, `train_soundscapes/`, etc.).

## Environment

This project uses `uv` with the repo virtual environment:

```bash
source .venv/bin/activate
uv sync
```

Run scripts with `uv run`, for example:

```bash
uv run python scripts/generate_spectrogram_batches.py
```

## Notes on labels and data splits

- `taxonomy.csv` and `sample_submission.csv` contain 234 classes.
- `train.csv` currently has labeled training clips for 206 classes.
- Additional classes can be represented in `train_soundscapes_labels.csv` via labeled time segments in `train_soundscapes/`.

