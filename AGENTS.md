## Development Environment

- Use the existing `.venv` and `uv` for all Python tasks. Never use `pip` or system `python`.
- The `kaggle` CLI is at `.venv/bin/kaggle` — not on the system PATH. Use the full path in any shell command, especially background shells.

## Workspace Facts

- BirdCLEF 2026 local data has 234 taxonomy species; `train.csv` has 206 unique `primary_label` species.
- V1 mel params (used by all best-performing models): `hop_length=512`, librosa defaults (`n_fft=2048`, `n_mels=128`, `fmin=0`).
- The 0.792 solo baseline is ConvNeXt-Small (v5), trained on species-001-010 + species-011-090 + soundscape-spectrograms with v1 mel params.
- Ensemble: ConvNeXt-Small (v5) + ECA-NFNet-L0 (v9), v1 mel params, temporal smoothing → 0.803.
- V3 (ConvNeXt-Small + ~238K pseudo-labeled soundscape clips, confidence ≥ 0.5): solo → 0.805.
- V3 + ECA-NFNet-L0 (v9) ensemble → 0.811.
- V4 (ECA-NFNet-L0 + pseudo-labels, same data as V3): training in progress. Goal: V3+V4 ensemble.

## Kaggle Workflow Rules

**Notebook paths:**
- Kernel source inputs mount at `/kaggle/input/notebooks/{owner}/{kernel-slug}/filename`
- Dataset inputs mount at `/kaggle/input/datasets/{owner}/{dataset-slug}/`
- Competition data mounts at `/kaggle/input/competitions/birdclef-2026/` (via kagglehub)
- Model inputs mount at `/kaggle/input/models/{owner}/{model-slug}/pytorch/{variation}/{version}/filename`

**Authentication:**
- Kaggle notebook environments are pre-authenticated as the notebook owner. Never add credential setup cells (`UserSecretsClient`, `KAGGLE_USERNAME`, `KAGGLE_KEY`). Confirmed by commit `39becdc`.

**Dataset zip uploads:**
- Kaggle **automatically extracts zip files** when you upload them to a dataset via `kaggle datasets create` or `kaggle datasets version`. The mounted dataset directory contains extracted files, not zips.
- Extraction structure is not guaranteed to be flat: files may appear at the dataset root OR inside subdirectories named after the zip (e.g. `batch_0001/file.png`).
- **Never `glob('*.zip')` in a mounted dataset directory.** Use `rglob('*.extension')` to find files regardless of directory depth. Build a `{filename: full_path}` lookup:
  ```python
  lookup = {p.name: str(p) for pl_dir in DIRS for p in pl_dir.rglob('*.png')}
  ```

**Git discipline:**
- Always `git pull` before committing. Kaggle auto-saves can create upstream commits that cause conflicts.

**Polling:**
- Use the `Monitor` tool with a single persistent background shell for Kaggle kernel status polling. Use `kstatus` (not `status`) as the variable name — `status` is read-only in zsh.

## Lessons Learned

_This section is updated whenever a new lesson is discovered. Any AI agent working on this repo should add entries here proactively — do not wait to be asked._

| Date | Lesson |
|------|--------|
| 2026-05 | Kaggle auto-extracts dataset zips; `glob('*.zip')` finds nothing. Use `rglob` and a filename→path lookup instead. |
| 2026-05 | `load_learner` does not auto-move models to GPU. After loading, always call `.model.to(device)` explicitly. |
| 2026-05 | V2 mel params (hop=320, fmin=50, fmax=14000, n_fft=1024) consistently underperform V1 params. Stick to V1 for new runs unless explicitly testing a new mel config. |
