from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


TARGET_SIZE = (224, 224)
CLIP_DURATION_SECONDS = 5
CHUNK_OFFSETS_SECONDS = (0.0, 2.5)


@dataclass
class BatchManifestRow:
    batch_id: int
    rank_start: int
    rank_end: int
    species_count: int
    species: str
    image_count: int
    zip_path: str
    zip_size_bytes: int
    error_count: int
    skipped_existing_png_count: int
    skipped_existing_zip: bool


def get_chunks(
    samples: np.ndarray,
    sample_rate: int,
    clip_duration: int = CLIP_DURATION_SECONDS,
    offsets_seconds: tuple[float, ...] = CHUNK_OFFSETS_SECONDS,
) -> list[tuple[int, int, np.ndarray]]:
    """Chunk audio with explicit start offsets, keeping complete chunks only."""
    clip_length = clip_duration * sample_rate
    chunks: list[tuple[int, int, np.ndarray]] = []

    for offset_secs in offsets_seconds:
        start = int(round(offset_secs * sample_rate))
        while start + clip_length <= len(samples):
            chunk = samples[start : start + clip_length]
            start_sec = start // sample_rate
            offset_ms = int(round(offset_secs * 1000))
            chunks.append((offset_ms, start_sec, chunk))
            start += clip_length

    return chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and zip BirdCLEF spectrograms in ranked batches."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/birdclef-2026"),
        help="Directory containing train.csv and train_audio.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/spectrogram_batches"),
        help="Root directory for work folders, zips, and manifest.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of species per zip batch.",
    )
    return parser


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
            # species directories appear at zip root
            arcname = file_path.relative_to(source_dir)
            zf.write(file_path, arcname=str(arcname))


def count_png_in_zip(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path, mode="r") as zf:
        return sum(1 for name in zf.namelist() if name.lower().endswith(".png"))


def main() -> None:
    args = build_parser().parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    train_csv = data_root / "train.csv"
    train_audio = data_root / "train_audio"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train.csv: {train_csv}")
    if not train_audio.exists():
        raise FileNotFoundError(f"Missing train_audio directory: {train_audio}")

    work_root = output_root / "work"
    zip_root = output_root / "zips"
    manifest_csv = output_root / "manifest.csv"
    manifest_json = output_root / "manifest.json"
    error_log = output_root / "errors.log"

    output_root.mkdir(parents=True, exist_ok=True)
    zip_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)
    species_ranked = df["primary_label"].value_counts().index.tolist()
    num_species = len(species_ranked)

    rows: list[BatchManifestRow] = []

    if error_log.exists():
        error_log.unlink()

    for batch_start in range(0, num_species, args.batch_size):
        rank_start = batch_start + 1
        rank_end = min(batch_start + args.batch_size, num_species)
        batch_id = (batch_start // args.batch_size) + 1
        batch_species = species_ranked[batch_start : batch_start + args.batch_size]

        batch_slug = f"batch_{batch_id:03d}_rank_{rank_start:03d}-{rank_end:03d}"
        batch_work_dir = work_root / batch_slug
        batch_zip = zip_root / f"{batch_slug}.zip"

        if batch_zip.exists():
            rows.append(
                BatchManifestRow(
                    batch_id=batch_id,
                    rank_start=rank_start,
                    rank_end=rank_end,
                    species_count=len(batch_species),
                    species="|".join(batch_species),
                    image_count=count_png_in_zip(batch_zip),
                    zip_path=str(batch_zip),
                    zip_size_bytes=batch_zip.stat().st_size,
                    error_count=0,
                    skipped_existing_png_count=0,
                    skipped_existing_zip=True,
                )
            )
            print(f"[batch {batch_id}] Zip exists, skipping: {batch_zip}")
            continue

        if batch_work_dir.exists():
            shutil.rmtree(batch_work_dir)
        batch_work_dir.mkdir(parents=True, exist_ok=True)

        batch_df = df[df["primary_label"].isin(batch_species)]
        generated_count = 0
        skipped_existing_png_count = 0
        error_count = 0

        progress = tqdm(
            batch_df.itertuples(index=False),
            total=len(batch_df),
            desc=f"Batch {batch_id} ({rank_start}-{rank_end})",
            unit="file",
        )

        for row in progress:
            label = row.primary_label
            audio_path = train_audio / row.filename

            species_dir = batch_work_dir / label
            species_dir.mkdir(parents=True, exist_ok=True)

            try:
                samples, sample_rate = librosa.load(audio_path, sr=None)
                chunks = get_chunks(samples=samples, sample_rate=sample_rate)

                stem = Path(row.filename).stem
                for offset_ms, start_sec, chunk in chunks:
                    out_file = species_dir / f"{stem}_o{offset_ms}ms_s{start_sec}.png"
                    if out_file.exists():
                        skipped_existing_png_count += 1
                        continue

                    s = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                    s_db = librosa.power_to_db(s, ref=np.max)
                    s_norm = normalize_to_uint8(s_db)
                    image = Image.fromarray(s_norm).resize(TARGET_SIZE)
                    image.save(out_file)
                    generated_count += 1
            except Exception as exc:  # noqa: BLE001 - do not halt full pipeline
                error_count += 1
                with error_log.open("a", encoding="utf-8") as fp:
                    fp.write(f"{audio_path}\t{exc}\n")

        create_zip_from_dir(batch_work_dir, batch_zip)
        zip_size = batch_zip.stat().st_size
        image_count = generated_count + skipped_existing_png_count

        rows.append(
            BatchManifestRow(
                batch_id=batch_id,
                rank_start=rank_start,
                rank_end=rank_end,
                species_count=len(batch_species),
                species="|".join(batch_species),
                image_count=image_count,
                zip_path=str(batch_zip),
                zip_size_bytes=zip_size,
                error_count=error_count,
                skipped_existing_png_count=skipped_existing_png_count,
                skipped_existing_zip=False,
            )
        )

        shutil.rmtree(batch_work_dir, ignore_errors=True)
        print(
            f"[batch {batch_id}] zipped {image_count} PNGs across "
            f"{len(batch_species)} species -> {batch_zip.name}"
        )

    with manifest_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=list(asdict(rows[0]).keys()) if rows else list(BatchManifestRow.__annotations__.keys()),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    with manifest_json.open("w", encoding="utf-8") as fp:
        json.dump([asdict(row) for row in rows], fp, indent=2)

    print(f"Done. Created {len(rows)} batch archives in {zip_root}")
    print(f"Manifest CSV: {manifest_csv}")
    print(f"Manifest JSON: {manifest_json}")
    if error_log.exists():
        print(f"Errors logged to: {error_log}")


if __name__ == "__main__":
    main()
