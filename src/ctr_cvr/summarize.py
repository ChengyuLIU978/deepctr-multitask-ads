"""Aggregate comparable experiment runs across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .train import clean_json, json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = [
        json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        for run_dir in args.run_dirs
    ]
    data_manifests = [
        json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
        for run_dir in args.run_dirs
    ]
    fingerprints = {
        item["source"].get("preparation_metadata", {}).get("prepared_sha256")
        for item in data_manifests
    }
    if len(fingerprints) != 1 or None in fingerprints:
        raise RuntimeError("runs do not share one auditable prepared-data fingerprint")
    split_boundaries = {
        (item["train_end_timestamp"], item["validation_end_timestamp"])
        for item in data_manifests
    }
    if len(split_boundaries) != 1:
        raise RuntimeError("runs do not share the same temporal split")

    models = sorted(set.intersection(*(set(item["results"]) for item in manifests)))
    aggregate: dict[str, object] = {}
    for model in models:
        metric_names = sorted(
            key
            for key, value in manifests[0]["results"][model]["test_metrics"].items()
            if isinstance(value, (int, float)) and key not in {"rows", "clicked_rows"}
        )
        metrics: dict[str, object] = {}
        for metric in metric_names:
            values = np.asarray(
                [item["results"][model]["test_metrics"][metric] for item in manifests],
                dtype=np.float64,
            )
            metrics[metric] = {
                "values": values.tolist(),
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        elapsed = np.asarray(
            [item["results"][model]["elapsed_seconds"] for item in manifests],
            dtype=np.float64,
        )
        aggregate[model] = {
            "metrics": metrics,
            "elapsed_seconds": {
                "values": elapsed.tolist(),
                "mean": float(elapsed.mean()),
                "sample_std": float(elapsed.std(ddof=1)) if len(elapsed) > 1 else 0.0,
            },
            "trainable_parameters": manifests[0]["results"][model]["trainable_parameters"],
        }

    result = clean_json(
        {
            "run_count": len(manifests),
            "seeds": [item["seed"] for item in manifests],
            "prepared_data_sha256": next(iter(fingerprints)),
            "temporal_split_boundaries": list(next(iter(split_boundaries))),
            "comparability_contract": {
                "same_data": True,
                "same_temporal_split": True,
                "same_feature_set": len({item["feature_set"] for item in manifests}) == 1,
                "same_capacity": len({item["capacity"] for item in manifests}) == 1,
                "same_maximum_epochs": len({item["maximum_epochs"] for item in manifests}) == 1,
            },
            "aggregate": aggregate,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
