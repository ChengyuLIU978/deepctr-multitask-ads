"""Build one auditable summary for the negative-transfer study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train import json_dump


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS = (
    "ctr_auc",
    "ctr_logloss",
    "ctcvr_auc",
    "ctcvr_logloss",
    "cvr_auc_on_clicked",
    "cvr_logloss_on_clicked",
)
SOURCES = {
    "deepfm": ("final_real_10pct_summary.json", "deepfm"),
    "esmm_equal": ("final_real_10pct_summary.json", "esmm"),
    "sharedbottom_equal": ("negative_sharedbottom_equal_summary.json", "sharedbottom"),
    "mmoe_equal": ("negative_mmoe_equal_summary.json", "mmoe"),
    "mmoe_ctr_heavy": ("negative_mmoe_ctr_heavy_summary.json", "mmoe"),
    "mmoe_ctcvr_heavy": ("negative_mmoe_ctcvr_heavy_summary.json", "mmoe"),
    "mmoe_experts2": ("negative_mmoe_experts2_summary.json", "mmoe"),
    "mmoe_experts5": ("negative_mmoe_experts5_summary.json", "mmoe"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "negative_transfer_summary.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_configuration(summary: dict, model: str, source_file: str) -> dict:
    aggregate = summary["aggregate"][model]
    metrics = {
        name: {
            "mean": aggregate["metrics"][name]["mean"],
            "sample_std": aggregate["metrics"][name]["sample_std"],
        }
        for name in METRICS
        if name in aggregate["metrics"]
    }
    return {
        "source": f"artifacts/{source_file}",
        "model": model,
        "run_count": summary["run_count"],
        "seeds": summary["seeds"],
        "prepared_data_sha256": summary["prepared_data_sha256"],
        "experiment_configuration": summary.get("experiment_configuration"),
        "metrics": metrics,
        "elapsed_seconds": aggregate["elapsed_seconds"],
        "trainable_parameters": aggregate["trainable_parameters"],
        "best_epochs": aggregate.get("best_epochs"),
        "epochs_ran": aggregate.get("epochs_ran"),
    }


def metric_deltas(configurations: dict, candidate: str, baseline: str) -> dict[str, float]:
    candidate_metrics = configurations[candidate]["metrics"]
    baseline_metrics = configurations[baseline]["metrics"]
    return {
        metric: candidate_metrics[metric]["mean"] - baseline_metrics[metric]["mean"]
        for metric in sorted(set(candidate_metrics) & set(baseline_metrics))
    }


def main() -> None:
    args = parse_args()
    configurations = {}
    fingerprints = set()
    for name, (source_file, model) in SOURCES.items():
        summary = read_json(args.artifacts_dir / source_file)
        configuration = extract_configuration(summary, model, source_file)
        configurations[name] = configuration
        fingerprints.add(configuration["prepared_data_sha256"])
    if len(fingerprints) != 1:
        raise RuntimeError("negative-transfer inputs do not share one data fingerprint")

    diagnostic = read_json(
        args.artifacts_dir / "diagnostic_real_10pct_lr3e4_seed2026" / "run_manifest.json"
    )
    equal_seed = read_json(
        args.artifacts_dir / "negative_mmoe_equal_seed2026" / "run_manifest.json"
    )
    training_diagnostic = {
        "default_learning_rate": {
            "learning_rate": equal_seed["learning_rate"],
            "maximum_epochs": equal_seed["maximum_epochs"],
            "patience": equal_seed["early_stopping_patience"],
            "mmoe_best_epoch": equal_seed["results"]["mmoe"]["best_epoch"],
            "mmoe_epochs_ran": equal_seed["results"]["mmoe"]["epochs_ran"],
            "mmoe_test_metrics": equal_seed["results"]["mmoe"]["test_metrics"],
        },
        "lower_learning_rate": {
            "learning_rate": diagnostic["learning_rate"],
            "maximum_epochs": diagnostic["maximum_epochs"],
            "patience": diagnostic["early_stopping_patience"],
            "best_epochs": {
                model: result["best_epoch"] for model, result in diagnostic["results"].items()
            },
            "epochs_ran": {
                model: result["epochs_ran"] for model, result in diagnostic["results"].items()
            },
            "test_metrics": {
                model: result["test_metrics"] for model, result in diagnostic["results"].items()
            },
        },
    }

    result = {
        "study": "multitask negative transfer on the 10% real Criteo sample",
        "prepared_data_sha256": next(iter(fingerprints)),
        "metric_delta_definition": "candidate mean minus baseline mean; no significance claim",
        "configurations": configurations,
        "deltas": {
            "sharedbottom_vs_esmm": metric_deltas(configurations, "sharedbottom_equal", "esmm_equal"),
            "mmoe_equal_vs_esmm": metric_deltas(configurations, "mmoe_equal", "esmm_equal"),
            "mmoe_ctr_heavy_vs_equal": metric_deltas(
                configurations, "mmoe_ctr_heavy", "mmoe_equal"
            ),
            "mmoe_ctcvr_heavy_vs_equal": metric_deltas(
                configurations, "mmoe_ctcvr_heavy", "mmoe_equal"
            ),
            "mmoe_experts2_vs_experts3": metric_deltas(
                configurations, "mmoe_experts2", "mmoe_equal"
            ),
            "mmoe_experts5_vs_experts3": metric_deltas(
                configurations, "mmoe_experts5", "mmoe_equal"
            ),
        },
        "training_diagnostic": training_diagnostic,
        "guardrail": (
            "Three seeds describe observed variability but do not establish statistical significance "
            "or online business impact."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
