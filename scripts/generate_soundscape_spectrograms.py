"""Generate mel-spectrogram PNGs from labeled 5s windows in train_soundscapes.

This is the multi-label counterpart to scripts/generate_spectrogram_batches.py.
It produces one PNG per unique (filename, start, end) row in
train_soundscapes_labels.csv, plus a soundscape_index.csv mapping each PNG to
its semicolon-joined label string. Output is zipped into a single archive
ready to upload as a Kaggle dataset for the multi-label trainer notebook.

Run locally so we never bake spectrograms inside Kaggle's /kaggle/working
disk/RAM ceiling:

    uv run python scripts/generate_soundscape_spectrograms.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


TARGET_SIZE = (224, 224)
SAMPLE_RATE = 32000
WINDOW_SECONDS = 5
ZIP_NAME = "soundscapes.zip"
INDEX_NAME = "soundscape_index.csv"


@dataclass
class SoundscapeManifest:
    unique_windows: int
    png_count: int
    skipped_existing_png_count: int
    error_count: int
    skipped_existing_zip: bool
    zip_path: str
    zip_size_bytes: int
    index_csv_path: str
    sample_rate: int
    window_seconds: int


def parse_hhmmss_to_seconds(value: str) -> int:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def normalize_to_uint8(s_db: np.ndarray) -> np.ndarray:
    min_val = float(s_db.min())
    max_val = float(s_db.max())
    if max_val == min_val:
        return np.zeros_like(s_db, dtype=np.uint8)
    return ((s_db - min_val) / (max_val - min_val) * 255).astype(np.uint8)


def create_zip_from_dir(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if not file_path.is_file():
                continue
            arcname = file_path.relative_to(source_dir)
            zf.write(file_path, arcname=str(arcname))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate mel-spectrogram PNGs from labeled 5s windows in "
            "train_soundscapes and zip them with a label index for Kaggle upload."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/birdclef-2026"),
        help="Directory containing train_soundscapes/ and train_soundscapes_labels.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/soundscape_spectrograms"),
        help="Root directory for work folder, zip, and manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if the zip already exists.",
    )
    return parser


def load_unique_windows(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    required = {"filename", "start", "end", "primary_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels CSV missing columns: {missing}")
    # Each (filename, start, end) is duplicated up to ~2x in the source CSV; the
    # primary_label string is identical across duplicates, so dropping them is safe.
    df = df.drop_duplicates(subset=["filename", "start", "end"]).reset_index(drop=True)
    df["start_sec"] = df["start"].map(parse_hhmmss_to_seconds)
    df["end_sec"] = df["end"].map(parse_hhmmss_to_seconds)
    return df


def main() -> None:
    args = build_parser().parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    labels_csv = data_root / "train_soundscapes_labels.csv"
    soundscapes_dir = data_root / "train_soundscapes"

    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")
    if not soundscapes_dir.exists():
        raise FileNotFoundError(f"Missing soundscapes directory: {soundscapes_dir}")

    work_root = output_root / "work"
    zip_root = output_root / "zips"
    manifest_csv = output_root / "manifest.csv"
    manifest_json = output_root / "manifest.json"
    error_log = output_root / "errors.log"
    zip_path = zip_root / ZIP_NAME

    output_root.mkdir(parents=True, exist_ok=True)
    zip_root.mkdir(parents=True, exist_ok=True)

    if zip_path.exists() and not args.force:
        with zipfile.ZipFile(zip_path) as zf:
            png_count = sum(1 for n in zf.namelist() if n.lower().endswith(".png"))
        manifest = SoundscapeManifest(
            unique_windows=png_count,
            png_count=png_count,
            skipped_existing_png_count=png_count,
            error_count=0,
            skipped_existing_zip=True,
            zip_path=str(zip_path),
            zip_size_bytes=zip_path.stat().st_size,
            index_csv_path="(in zip)",
            sample_rate=SAMPLE_RATE,
            window_seconds=WINDOW_SECONDS,
        )
        _write_manifest(manifest_csv, manifest_json, manifest)
        print(f"Zip exists, skipping: {zip_path}")
        return

    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    if error_log.exists():
        error_log.unlink()

    df = load_unique_windows(labels_csv)
    unique_windows = len(df)
    print(
        f"Found {unique_windows} unique 5s windows across "
        f"{df['filename'].nunique()} soundscape files."
    )

    index_rows: list[dict[str, str]] = []
    generated_count = 0
    skipped_existing_png_count = 0
    error_count = 0

    progress = tqdm(total=unique_windows, desc="Soundscape windows", unit="window")
    for filename, file_df in df.groupby("filename", sort=False):
        audio_path = soundscapes_dir / filename
        try:
            samples, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001 - one bad file should not halt
            error_count += len(file_df)
            with error_log.open("a", encoding="utf-8") as fp:
                fp.write(f"LOAD\t{audio_path}\t{exc}\n")
            progress.update(len(file_df))
            continue

        clip_length = WINDOW_SECONDS * sr
        stem = Path(filename).stem

        for row in file_df.itertuples(index=False):
            progress.update(1)
            start_sample = row.start_sec * sr
            end_sample = start_sample + clip_length
            if end_sample > len(samples):
                error_count += 1
                with error_log.open("a", encoding="utf-8") as fp:
                    fp.write(
                        f"OOB\t{audio_path}\tstart={row.start_sec}s "
                        f"file_len={len(samples) / sr:.2f}s\n"
                    )
                continue

            chunk = samples[start_sample:end_sample]
            png_name = f"{stem}__s{row.start_sec}.png"
            out_file = work_root / png_name

            try:
                if out_file.exists():
                    skipped_existing_png_count += 1
                else:
                    s = librosa.feature.melspectrogram(y=chunk, sr=sr)
                    s_db = librosa.power_to_db(s, ref=np.max)
                    s_norm = normalize_to_uint8(s_db)
                    Image.fromarray(s_norm).resize(TARGET_SIZE).save(out_file)
                    generated_count += 1
            except Exception as exc:  # noqa: BLE001
                error_count += 1
                with error_log.open("a", encoding="utf-8") as fp:
                    fp.write(f"SPECTROGRAM\t{audio_path}\t{row.start_sec}\t{exc}\n")
                continue

            index_rows.append(
                {
                    "image_path": png_name,
                    "labels": row.primary_label,
                }
            )
    progress.close()

    index_csv = work_root / INDEX_NAME
    with index_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_path", "labels"])
        writer.writeheader()
        writer.writerows(index_rows)

    create_zip_from_dir(work_root, zip_path)
    zip_size = zip_path.stat().st_size

    _run_sanity_checks(index_rows, data_root)

    manifest = SoundscapeManifest(
        unique_windows=unique_windows,
        png_count=generated_count + skipped_existing_png_count,
        skipped_existing_png_count=skipped_existing_png_count,
        error_count=error_count,
        skipped_existing_zip=False,
        zip_path=str(zip_path),
        zip_size_bytes=zip_size,
        index_csv_path=str(index_csv),
        sample_rate=SAMPLE_RATE,
        window_seconds=WINDOW_SECONDS,
    )
    _write_manifest(manifest_csv, manifest_json, manifest)

    print(
        f"Done. {generated_count} PNGs generated, "
        f"{skipped_existing_png_count} skipped as existing, "
        f"{error_count} errors."
    )
    print(f"Zip: {zip_path} ({zip_size / 1024 / 1024:.1f} MiB)")
    print(f"Index inside zip: {INDEX_NAME}")
    if error_log.exists():
        print(f"Errors logged to: {error_log}")


def _run_sanity_checks(index_rows: list[dict[str, str]], data_root: Path) -> None:
    if not index_rows:
        raise RuntimeError("No index rows produced; nothing to validate.")

    labels_seen: set[str] = set()
    for row in index_rows:
        labels_seen.update(row["labels"].split(";"))

    print(f"Sanity: {len(index_rows)} index rows, {len(labels_seen)} unique labels.")

    train_csv = data_root / "train.csv"
    taxonomy_csv = data_root / "taxonomy.csv"
    if not train_csv.exists() or not taxonomy_csv.exists():
        print("Sanity: skipping missing-28 check (train.csv or taxonomy.csv absent).")
        return

    train_species = set(pd.read_csv(train_csv)["primary_label"].unique())
    taxonomy_species = set(pd.read_csv(taxonomy_csv)["primary_label"].unique())
    missing28 = taxonomy_species - train_species
    missing_in_index = missing28 - labels_seen
    if missing_in_index:
        print(
            f"WARNING: {len(missing_in_index)} of the {len(missing28)} missing-28 "
            f"classes are NOT represented in the soundscape index: "
            f"{sorted(missing_in_index)}"
        )
    else:
        print(
            f"Sanity: all {len(missing28)} missing-from-train classes are present "
            f"in the soundscape index."
        )


def _write_manifest(
    manifest_csv: Path, manifest_json: Path, manifest: SoundscapeManifest
) -> None:
    with manifest_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(asdict(manifest).keys()))
        writer.writeheader()
        writer.writerow(asdict(manifest))
    with manifest_json.open("w", encoding="utf-8") as fp:
        json.dump(asdict(manifest), fp, indent=2)


if __name__ == "__main__":
    main()
