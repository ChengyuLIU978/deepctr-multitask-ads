import pytest

from ctr_cvr.analyze_negative_transfer import metric_deltas
from ctr_cvr.summarize import configuration_value


def test_metric_deltas_use_candidate_minus_baseline():
    configurations = {
        "candidate": {"metrics": {"auc": {"mean": 0.8}, "loss": {"mean": 0.2}}},
        "baseline": {"metrics": {"auc": {"mean": 0.7}, "loss": {"mean": 0.3}}},
    }

    deltas = metric_deltas(configurations, "candidate", "baseline")

    assert deltas["auc"] == pytest.approx(0.1)
    assert deltas["loss"] == pytest.approx(-0.1)


def test_configuration_value_recovers_legacy_run_level_batch_size():
    manifest = {
        "results": {
            "esmm": {"batch_size": 2048},
            "mmoe": {"batch_size": 2048},
        }
    }

    assert configuration_value(manifest, "batch_size") == 2048
