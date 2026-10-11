"""Uniformly sample the official Criteo attribution file without redistributing it."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import RAW_REQUIRED_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def manifest_path(path: Path) -> str:
    """Return a portable project-relative path without exposing a home directory."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    input_path: Path,
    output_path: Path,
    sample_rate: float,
    seed: int,
    chunksize: int,
    source_url: str,
) -> dict[str, object]:
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    source_rows = sampled_rows = 0
    source_clicks = source_conversions = source_ctcvr = 0
    sampled_clicks = sampled_conversions = sampled_ctcvr = 0
    monotonic = True
    previous_timestamp: int | None = None

    reader = pd.read_csv(
        input_path,
        sep="\t",
        compression="infer",
        usecols=RAW_REQUIRED_COLUMNS,
        chunksize=chunksize,
    )
    # mtime=0 makes the compressed output byte-for-byte reproducible for the
    # same source, seed, pandas version, and parameters.
    with output_path.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed_output:
            output_handle = io.TextIOWrapper(compressed_output, encoding="utf-8", newline="")
            wrote_header = False
            for chunk in reader:
                missing = sorted(set(RAW_REQUIRED_COLUMNS) - set(chunk.columns))
                if missing:
                    raise ValueError(f"source is missing required columns: {missing}")
                timestamps = pd.to_numeric(chunk["timestamp"], errors="raise").to_numpy(dtype=np.int64)
                if previous_timestamp is not None and timestamps.size and timestamps[0] < previous_timestamp:
                    monotonic = False
                if timestamps.size and np.any(timestamps[1:] < timestamps[:-1]):
                    monotonic = False
                if timestamps.size:
                    previous_timestamp = int(timestamps[-1])

                click = chunk["click"].to_numpy(dtype=np.int8)
                conversion = chunk["conversion"].to_numpy(dtype=np.int8)
                ctcvr = (click == 1) & (conversion == 1)
                source_rows += len(chunk)
                source_clicks += int(click.sum())
                source_conversions += int(conversion.sum())
                source_ctcvr += int(ctcvr.sum())

                keep = rng.random(len(chunk)) < sample_rate
                sampled = chunk.loc[keep, RAW_REQUIRED_COLUMNS]
                sampled.to_csv(output_handle, index=False, header=not wrote_header)
                wrote_header = True
                sampled_rows += len(sampled)
                sampled_clicks += int(sampled["click"].sum())
                sampled_conversions += int(sampled["conversion"].sum())
                sampled_ctcvr += int(((sampled["click"] == 1) & (sampled["conversion"] == 1)).sum())
            output_handle.flush()
            output_handle.detach()

    if sampled_rows == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError("uniform sample produced zero rows; increase sample_rate")

    metadata = {
        "source_url": source_url,
        "source_path": manifest_path(input_path),
        "source_sha256": sha256_file(input_path),
        "source_rows": source_rows,
        "source_clicks": source_clicks,
        "source_conversions": source_conversions,
        "source_post_click_conversions": source_ctcvr,
        "sample_rate": sample_rate,
        "seed": seed,
        "sampled_rows": sampled_rows,
        "sampled_clicks": sampled_clicks,
        "sampled_conversions": sampled_conversions,
        "sampled_post_click_conversions": sampled_ctcvr,
        "source_was_timestamp_sorted": monotonic,
        "prepared_sha256": sha256_file(output_path),
        "method": "single-pass uniform Bernoulli sample; no label-based oversampling",
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--source-url",
        default="https://huggingface.co/datasets/criteo/criteo-attribution-dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare(
        input_path=args.input,
        output_path=args.output,
        sample_rate=args.sample_rate,
        seed=args.seed,
        chunksize=args.chunksize,
        source_url=args.source_url,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

