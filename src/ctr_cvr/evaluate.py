"""Reload a saved best checkpoint and reproduce its test-set metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import FeaturePreprocessor, load_frame, temporal_split, validate_and_label
from .metrics import evaluate_ctr, evaluate_multitask
from .models import build_model
from .train import clean_json, json_dump, manifest_path, set_seed, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("deepfm", "esmm", "sharedbottom", "mmoe"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((args.run_dir / "data_manifest.json").read_text(encoding="utf-8"))
    seed = int(run_manifest["seed"])
    set_seed(seed, min(4, __import__("os").cpu_count() or 1))

    actual_data_sha256 = sha256_file(args.data)
    expected_data_sha256 = data_manifest["source"].get("verified_prepared_sha256")
    if expected_data_sha256 is None:
        expected_data_sha256 = data_manifest["source"].get("preparation_metadata", {}).get(
            "prepared_sha256"
        )
    if expected_data_sha256 and actual_data_sha256 != expected_data_sha256:
        raise RuntimeError(
            "Evaluation data SHA256 differs from the saved experiment: "
            f"expected {expected_data_sha256}, got {actual_data_sha256}"
        )

    frame = validate_and_label(load_frame(args.data))
    split = temporal_split(frame)
    if int(split.train_end_timestamp) != int(data_manifest["train_end_timestamp"]):
        raise RuntimeError("training boundary differs from the saved experiment")
    if int(split.validation_end_timestamp) != int(data_manifest["validation_end_timestamp"]):
        raise RuntimeError("validation boundary differs from the saved experiment")

    processor = FeaturePreprocessor.from_dict(data_manifest["preprocessor"])
    test_x = processor.transform(split.test)
    model = build_model(
        args.model,
        processor.feature_columns(),
        device="cuda:0" if args.device == "cuda" else "cpu",
        seed=seed,
        num_experts=int(run_manifest["num_experts"]),
        capacity=str(run_manifest["capacity"]),
    )
    checkpoint = args.run_dir / run_manifest["results"][args.model]["checkpoint"]["filename"]
    try:
        state = torch.load(checkpoint, map_location=model.device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=model.device)
    model.load_state_dict(state)
    predictions = model.predict(test_x, batch_size=args.batch_size)
    if args.model == "deepfm":
        metrics = evaluate_ctr(split.test["click"].to_numpy(), predictions)
    else:
        metrics = evaluate_multitask(
            split.test["click"].to_numpy(),
            split.test["ctcvr"].to_numpy(),
            predictions[:, 0],
            predictions[:, 1],
        )

    saved = run_manifest["results"][args.model]["test_metrics"]
    differences = {
        key: abs(float(metrics[key]) - float(saved[key]))
        for key in metrics
        if metrics[key] is not None and saved.get(key) is not None and isinstance(metrics[key], (float, np.floating))
    }
    maximum_difference = max(differences.values(), default=0.0)
    result = clean_json(
        {
            "model": args.model,
            "checkpoint": manifest_path(checkpoint),
            "verified_data_sha256": actual_data_sha256,
            "metrics": metrics,
            "saved_metric_max_absolute_difference": maximum_difference,
            "matches_saved_metrics": maximum_difference < 1e-10,
        }
    )
    json_dump(args.run_dir / f"{args.model}_reload_evaluation.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
