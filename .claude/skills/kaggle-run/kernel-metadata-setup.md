# Kernel Metadata Setup

Each notebook pushed to Kaggle via the CLI needs a `kernel-metadata.json` file
committed **in the same directory as the notebook** in the repo. This is a one-time
setup per notebook file — not per model variation. The metadata describes the
notebook (its Kaggle kernel ID, GPU settings, data sources), not the model version
it references. As new variations are created and new notebook files are added to the
repo, each new notebook gets its own metadata file. Existing metadata files are
never overwritten by this process.

The CLI reads the metadata from the repo directory directly when running
`kaggle kernels push -p <dir>` — no copying to a tmp folder is needed.

## Generating the metadata file

Run this in the directory containing the notebook:

```bash
kaggle kernels init -p <dir>
```

This creates a template `kernel-metadata.json`. Edit it with the correct values.

Alternatively, for an existing Kaggle notebook, pull its current metadata directly:

```bash
kaggle kernels pull <owner>/<kernel-slug> -m -p <dir>
```

The `-m` flag downloads the metadata file alongside the notebook. This is the
preferred approach for notebooks that already exist on Kaggle, as it guarantees
the slug and id are correct.

## Format

```json
{
  "id": "<owner>/<kernel-slug>",
  "title": "Human-readable title",
  "code_file": "<notebook-filename>.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": ["birdclef-2026"],
  "kernel_sources": [],
  "model_sources": []
}
```

## Key fields

- `id`: `<owner>/<kernel-slug>` where the slug matches the notebook's URL on Kaggle
- `enable_gpu`: `true` for the training notebook; the submission environment is
  controlled by Kaggle independently
- `competition_sources`: include `"birdclef-2026"` so competition data is available
  during the run
- `model_sources`: for the submission notebook, add the model here in the format
  `"<owner>/<model-slug>/<framework>/<variation-slug>/<version>"` — this must be
  updated each time a new version is uploaded (Step 6 of the skill handles this
  for the path string inside the notebook; the `model_sources` field here may also
  need updating if Kaggle requires it to mount the model)

## Finding the kernel slug

The slug is the last segment of the notebook's Kaggle URL:
`https://www.kaggle.com/code/<owner>/<slug>`

## Commit to the repo

`kernel-metadata.json` must be committed to the repo alongside the notebook.
`kaggle kernels push -p <dir>` reads both files from that directory at push time —
there is no separate tmp copy needed.
