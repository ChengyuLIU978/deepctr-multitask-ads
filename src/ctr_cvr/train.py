"""Train DeepFM, ESMM, and MMoE with one auditable data/metric contract."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from .data import (
    FeaturePreprocessor,
    generate_demo_data,
    labels,
    load_frame,
    partition_summary,
    temporal_split,
    validate_and_label,
)
from .metrics import evaluate_ctr, evaluate_multitask
from .models import SUPPORTED_CAPACITIES, SUPPORTED_MODELS, build_model, parameter_count


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def manifest_path(path: Path) -> str:
    """Return a portable project-relative path without exposing a home directory."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def parse_models(value: str) -> list[str]:
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(models) - set(SUPPORTED_MODELS))
    if invalid or not models:
        raise argparse.ArgumentTypeError(f"models must be a comma list from {SUPPORTED_MODELS}; invalid={invalid}")
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-mode", choices=("demo", "real"), default="demo")
    parser.add_argument("--data", type=Path, help="prepared Criteo CSV/TSV; required for --data-mode real")
    parser.add_argument("--demo-rows", type=int, default=12_000)
    parser.add_argument("--models", type=parse_models, default=list(SUPPORTED_MODELS))
    parser.add_argument("--feature-set", choices=("all", "sparse_only"), default="all")
    parser.add_argument("--epochs", type=int, default=10, help="maximum epochs; early stopping may finish sooner")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=3)
    parser.add_argument("--capacity", choices=SUPPORTED_CAPACITIES, default="base")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def choose_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return "cuda:0"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_manifest() -> dict[str, object]:
    import deepctr_torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "deepctr_torch": getattr(deepctr_torch, "__version__", "unknown"),
    }


def main() -> None:
    args = parse_args()
    if args.data_mode == "real" and args.data is None:
        raise SystemExit("--data is required when --data-mode real")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        raise SystemExit("epochs/batch-size must be positive and patience must be non-negative")
    if args.num_experts < 2:
        raise SystemExit("--num-experts must be at least 2")

    set_seed(args.seed, args.threads)
    device = choose_device(args.device)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (PROJECT_ROOT / "artifacts" / f"{run_id}_{args.data_mode}_{args.feature_set}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_mode == "demo":
        raw = generate_demo_data(args.demo_rows, args.seed)
        source = {"kind": "synthetic_demo", "rows_requested": args.demo_rows}
    else:
        prepared_sha256 = sha256_file(args.data)
        raw = load_frame(args.data)
        source = {
            "kind": "criteo_real",
            "path": manifest_path(args.data),
            "verified_prepared_sha256": prepared_sha256,
        }
        preparation_metadata_path = args.data.with_suffix(args.data.suffix + ".metadata.json")
        if preparation_metadata_path.exists():
            preparation_metadata = json.loads(
                preparation_metadata_path.read_text(encoding="utf-8")
            )
            expected_sha256 = preparation_metadata.get("prepared_sha256")
            if expected_sha256 and expected_sha256 != prepared_sha256:
                raise RuntimeError(
                    "Prepared data SHA256 differs from its provenance metadata: "
                    f"expected {expected_sha256}, got {prepared_sha256}"
                )
            source["preparation_metadata"] = preparation_metadata
        else:
            source["preparation_metadata_warning"] = (
                "No sibling .metadata.json was found; provenance is incomplete."
            )

    frame = validate_and_label(raw)
    split = temporal_split(frame)
    preprocessor = FeaturePreprocessor(
        feature_set=args.feature_set,
        embedding_dim=args.embedding_dim,
    ).fit(split.train)
    train_x = preprocessor.transform(split.train)
    validation_x = preprocessor.transform(split.validation)
    test_x = preprocessor.transform(split.test)
    feature_columns = preprocessor.feature_columns()

    data_manifest = {
        "source": source,
        "label_contract": {
            "ctr": "click",
            "ctcvr": "click AND conversion",
            "cvr_evaluation_population": "rows with click == 1",
        },
        "split_method": "chronological 70/15/15; identical timestamps stay together",
        "train_end_timestamp": split.train_end_timestamp,
        "validation_end_timestamp": split.validation_end_timestamp,
        "partitions": {
            "train": partition_summary(split.train),
            "validation": partition_summary(split.validation),
            "test": partition_summary(split.test),
        },
        "preprocessor": preprocessor.to_dict(),
    }
    json_dump(output_dir / "data_manifest.json", clean_json(data_manifest))
    json_dump(output_dir / "environment.json", clean_json(environment_manifest()))

    results: dict[str, object] = {}
    for model_name in args.models:
        from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint

        model_seed = args.seed
        set_seed(model_seed, args.threads)
        model = build_model(
            model_name,
            feature_columns,
            device=device,
            seed=model_seed,
            num_experts=args.num_experts,
            capacity=args.capacity,
        )
        trainable_parameters = parameter_count(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        if model_name == "deepfm":
            model.compile(optimizer, loss="binary_crossentropy", metrics=["binary_crossentropy"])
            train_y = labels(split.train, ["click"])
            validation_y = labels(split.validation, ["click"])
        else:
            model.compile(
                optimizer,
                loss=["binary_crossentropy", "binary_crossentropy"],
                metrics=["binary_crossentropy"],
            )
            train_y = labels(split.train, ["click", "ctcvr"])
            validation_y = labels(split.validation, ["click", "ctcvr"])

        checkpoint_path = output_dir / f"{model_name}_best.weights.pt"
        monitor = "val_binary_crossentropy"
        callbacks = [
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=monitor,
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode="min",
            ),
            EarlyStopping(
                monitor=monitor,
                min_delta=args.min_delta,
                patience=args.patience,
                verbose=1,
                mode="min",
                restore_best_weights=True,
            ),
        ]
        started = time.perf_counter()
        history = model.fit(
            train_x,
            train_y,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=2,
            validation_data=(validation_x, validation_y),
            shuffle=True,
            callbacks=callbacks,
        )
        if not checkpoint_path.exists():
            raise RuntimeError(f"best checkpoint was not created: {checkpoint_path}")
        try:
            best_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            best_state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(best_state)
        predictions = model.predict(test_x, batch_size=args.batch_size)
        elapsed_seconds = time.perf_counter() - started
        if model_name == "deepfm":
            metrics = evaluate_ctr(split.test["click"].to_numpy(), predictions)
        else:
            metrics = evaluate_multitask(
                split.test["click"].to_numpy(),
                split.test["ctcvr"].to_numpy(),
                predictions[:, 0],
                predictions[:, 1],
            )
        validation_history = history.history.get(monitor, [])
        best_epoch = int(np.argmin(validation_history) + 1) if validation_history else None
        result = {
            "model": model_name,
            "library": "DeepCTR-Torch 0.3.0",
            "trainable_parameters": trainable_parameters,
            "maximum_epochs": args.epochs,
            "epochs_ran": len(history.history.get("loss", [])),
            "best_epoch": best_epoch,
            "early_stopping": {
                "monitor": monitor,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "restored_best_weights": True,
            },
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "capacity": args.capacity,
            "num_experts": args.num_experts if model_name == "mmoe" else None,
            "elapsed_seconds": elapsed_seconds,
            "history": history.history,
            "checkpoint": {
                "filename": checkpoint_path.name,
                "sha256": sha256_file(checkpoint_path),
                "git_policy": "local artifact; ignored by git",
            },
            "test_metrics": metrics,
        }
        results[model_name] = clean_json(result)
        json_dump(output_dir / f"{model_name}_result.json", results[model_name])
        del model, optimizer, predictions, best_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "data_mode": args.data_mode,
        "feature_set": args.feature_set,
        "models": args.models,
        "seed": args.seed,
        "device": device,
        "capacity": args.capacity,
        "num_experts": args.num_experts,
        "maximum_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "early_stopping_min_delta": args.min_delta,
        "output_dir": manifest_path(output_dir),
        "results": results,
        "truthfulness_note": (
            "Synthetic schema smoke test; not a real-data benchmark."
            if args.data_mode == "demo"
            else "Real Criteo sample experiment; scope and sample method are in data_manifest.json."
        ),
    }
    run_manifest = clean_json(run_manifest)
    json_dump(output_dir / "run_manifest.json", run_manifest)
    registry = PROJECT_ROOT / "experiments" / "registry.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run_manifest, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(run_manifest, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

