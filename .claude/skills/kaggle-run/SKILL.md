---
name: kaggle-run
description: >
  Automates the BirdCLEF Kaggle competition run cycle after a code change has been
  implemented and committed. Use this skill whenever the user says something like
  "run it", "trigger a run", "submit the new model", "push to kaggle", "let's see
  how this does", or anything that implies kicking off a Kaggle training run and
  submission flow. Also use it when the user asks to create a new model variation
  or version on Kaggle. The skill covers: pushing the training notebook to Kaggle,
  polling for completion, downloading the model output, uploading it as a new model
  version or variation, updating the submission notebook to reference the new model,
  pushing the updated submission notebook, and preparing for manual submission in the UI.
compatibility:
  requires:
    - kaggle CLI — install with `pip install kaggle` (Python 3.11+ required);
      for uv-managed projects use `uv add kaggle`
    - Kaggle credentials configured at ~/.kaggle/kaggle.json
      (download from https://www.kaggle.com/settings/api → "Generate New Token")
    - kernel-metadata.json files committed alongside each notebook in the repo
      (see references/kernel-metadata-setup.md if these are missing)
---

# Kaggle Run Skill

This skill automates the BirdCLEF Kaggle run cycle. It assumes all code changes
have already been implemented, committed, and pushed to GitHub before this skill
is invoked.

---

## Step 0 — Confirm notebook files and run type

Before doing anything, ask the user **two questions** (can be asked together):

1. **Version or variation?**
   - **New version** = same architecture, incremental change → the same training and
     submission notebook files were updated in place and are already committed
   - **New variation** = different architecture/approach → new training and submission
     notebook files were created and are already committed

2. **Which notebook folder?**
   - Notebooks live under `notebooks/<variation-folder>/` in the repo, where each
     variation folder contains `training.ipynb`, `submission.ipynb`, and
     `kernel-metadata.json` files for each notebook. Ask the user to confirm which
     variation folder corresponds to this run (e.g. `notebooks/multilabel-234/`).
     For a new version this will be the same folder as last time; for a new variation
     it will be the newly created folder.
   - Also confirm the model variation slug the new output belongs to. For a new
     version, this is the existing slug (e.g. `multilabel_234`). For a new variation,
     the user will name the new slug now — confirm it before proceeding.

Do not proceed until both questions are answered.

---

## Step 1 — Verify prerequisites

Each variation has two subfolders under `notebooks/`: one for the training notebook
and one for the submission notebook (e.g. `notebooks/multilabel-234/` and
`notebooks/multilabel-234-submission/`). Check that the following exist in both
confirmed folders:

- The notebook `.ipynb` file (`training.ipynb` or `submission.ipynb`)
- A `kernel-metadata.json` file

If either metadata file is missing, stop and follow `references/kernel-metadata-setup.md`
before continuing.

---

## Step 2 — Push training notebook to Kaggle and trigger run

```bash
kaggle kernels push -p <path-to-training-notebook-directory>
```

The `-p` flag points to the directory containing the training notebook and its
`kernel-metadata.json`. Kaggle uses the metadata file to identify the target kernel
and uploads the notebook from that directory, triggering a full run.

Confirm the push succeeded before continuing.

---

## Step 3 — Poll until run completes

```bash
kaggle kernels status <kernel-id>
```

Where `<kernel-id>` is `{owner}/{kernel-slug}` from the training notebook's
`kernel-metadata.json`.

Poll every 2 minutes. Run the polling loop as a background task where possible so
the main agent thread is not blocked — the exact mechanism (shell backgrounding,
a separate process, etc.) depends on the runtime environment; use whatever is
appropriate. Report status updates to the user periodically.

Expected statuses:
- `running` / `queued` → keep polling
- `complete` → proceed to Step 4
- `error` / `cancelAcknowledged` → stop and report the failure to the user. Do not proceed.

---

## Step 4 — Download model output

```bash
kaggle kernels output <kernel-id> -p ./tmp/kaggle-output/
```

Ensure `./tmp/` is in the repo's `.gitignore` — downloaded model files are large
binary artifacts and must not be committed.

Run this as a background task where possible so the agent is not blocked during
the download. Use whatever backgrounding mechanism is appropriate for the runtime
environment. Notify the user when the download completes or fails.

Expected files: `model_multilabel_234.pkl` and `vocab.json` (or equivalent names
for a new variation).

Confirm both expected files are present before continuing.

---

## Step 5 — Upload model to Kaggle

Run the relevant command below as a background task where possible so the agent
is not blocked during the upload. Notify the user when the upload completes or fails.

### If new VERSION:
```bash
kaggle models variations versions create \
  <owner>/<model-slug>/PyTorch/<variation-slug> \
  -p ./tmp/kaggle-output/ \
  -n "<brief description of what changed>"
```

The version number is assigned automatically by Kaggle. Note the new version number
from the command output — it is needed in Step 6.

### If new VARIATION:
A `model-instance-metadata.json` must exist in `./tmp/kaggle-output/` before running
these commands. See `references/model-metadata-setup.md` for the format. The variation
slug was confirmed in Step 0.

```bash
kaggle models variations create <owner>/<model-slug> \
  -p ./tmp/kaggle-output/
```

Then create the first version of the new variation:

```bash
kaggle models variations versions create \
  <owner>/<model-slug>/PyTorch/<new-variation-slug> \
  -p ./tmp/kaggle-output/ \
  -n "Initial version"
```

Note the variation slug and version number from the output — both are needed in Step 6.

---

## Step 6 — Update submission notebook

Open the confirmed submission notebook (`.ipynb` file). Read the relevant cells and
locate all references to the model path and vocab path. Do not assume a fixed
structure — the reference may be:

- A top-level string assignment: `MODEL_PATH = '/kaggle/input/models/...'`
- A variable that feeds into a path elsewhere
- An f-string or path join

The Kaggle model path format is:
```
/kaggle/input/models/{owner}/{model-slug}/PyTorch/{variation-slug}/{version}/{filename}
```

Note: Kaggle stores the framework as `PyTorch` (capital P, capital T) regardless
of how it is specified at creation time. Use this casing when constructing or
verifying path strings.

Update `MODEL_PATH` and `VOCAB_PATH` to reflect the new version number (and new
variation slug if applicable). If this is a new variation, also update any markdown
cells or comments that reference the old variation by name.

Verify the change looks correct, then commit and push the updated submission notebook
to GitHub.

---

## Step 7 — Push submission notebook to Kaggle

```bash
kaggle kernels push -p <path-to-submission-notebook-directory>
```

Uploads the updated submission notebook to Kaggle so it is ready in the UI.
This does not trigger a submission run.

---

## Step 7b — Update model input in Kaggle UI (manual, required)

This step cannot be automated and must be done by the user before submitting.

In the Kaggle notebook UI:
1. Open the submission notebook
2. In the right sidebar under **Input**, find the current model source
3. Remove the old model version and add the new one (the version number confirmed in Step 5)
4. Confirm the sidebar now shows the correct version before proceeding

**This is required even though the notebook cell path and kernel-metadata.json have
already been updated.** For UI-triggered submission runs, Kaggle mounts whatever is
registered in the sidebar — not what the metadata file specifies. Skipping this step
will cause a FileNotFoundError at runtime.

---

## Step 8 — Hand off

Summarise what was completed:
- Training run completed ✓
- Model uploaded as `<owner>/<model-slug>/PyTorch/<variation-slug>/<version>` ✓
- Submission notebook updated and pushed ✓
- Remind the user to complete Step 7b (update model input in Kaggle sidebar) before submitting

Do not attempt to submit programmatically — code competition submission via the
public API is not supported and will return 403.

---

## Reference files

- `references/kernel-metadata-setup.md` — how to create kernel-metadata.json for a notebook
- `references/model-metadata-setup.md` — how to create model-instance-metadata.json for a new variation
