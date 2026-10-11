"""Metrics for CTR, joint click-conversion, and post-click CVR."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def _flat(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = _flat(y_true)
    y_score = _flat(y_score)
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def binary_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = _flat(y_true)
    y_score = np.clip(_flat(y_score), 1e-7, 1 - 1e-7)
    return float(log_loss(y_true, y_score, labels=[0, 1]))


def evaluate_ctr(y_ctr: np.ndarray, p_ctr: np.ndarray) -> dict[str, float | int | None]:
    y_ctr = _flat(y_ctr)
    p_ctr = np.clip(_flat(p_ctr), 0.0, 1.0)
    return {
        "rows": int(y_ctr.size),
        "ctr_auc": safe_auc(y_ctr, p_ctr),
        "ctr_logloss": binary_logloss(y_ctr, p_ctr),
        "actual_ctr": float(y_ctr.mean()),
        "predicted_ctr": float(p_ctr.mean()),
    }


def evaluate_multitask(
    y_ctr: np.ndarray,
    y_ctcvr: np.ndarray,
    p_ctr: np.ndarray,
    p_ctcvr: np.ndarray,
) -> dict[str, float | int | None]:
    """Evaluate unconditional tasks and CVR on the clicked population.

    For both ESMM and MMoE, conditional CVR is derived as
    P(click & conversion) / P(click). The ratio is clipped only for metric
    stability; the raw probability-order violation rate is reported so MMoE's
    independent heads cannot silently hide inconsistent probabilities.
    """

    y_ctr = _flat(y_ctr)
    y_ctcvr = _flat(y_ctcvr)
    p_ctr = np.clip(_flat(p_ctr), 1e-7, 1 - 1e-7)
    raw_p_ctcvr = _flat(p_ctcvr)
    p_ctcvr = np.clip(raw_p_ctcvr, 1e-7, 1 - 1e-7)
    metrics = evaluate_ctr(y_ctr, p_ctr)
    metrics.update(
        {
            "ctcvr_auc": safe_auc(y_ctcvr, p_ctcvr),
            "ctcvr_logloss": binary_logloss(y_ctcvr, p_ctcvr),
            "actual_ctcvr": float(y_ctcvr.mean()),
            "predicted_ctcvr": float(p_ctcvr.mean()),
            "probability_order_violation_rate": float(np.mean(raw_p_ctcvr > p_ctr)),
        }
    )

    clicked = y_ctr == 1
    metrics["clicked_rows"] = int(clicked.sum())
    if clicked.any():
        y_cvr = y_ctcvr[clicked]
        p_cvr = np.clip(p_ctcvr[clicked] / p_ctr[clicked], 1e-7, 1 - 1e-7)
        metrics.update(
            {
                "cvr_auc_on_clicked": safe_auc(y_cvr, p_cvr),
                "cvr_logloss_on_clicked": binary_logloss(y_cvr, p_cvr),
                "actual_cvr_on_clicked": float(y_cvr.mean()),
                "predicted_cvr_on_clicked": float(p_cvr.mean()),
            }
        )
    else:
        metrics.update(
            {
                "cvr_auc_on_clicked": None,
                "cvr_logloss_on_clicked": None,
                "actual_cvr_on_clicked": None,
                "predicted_cvr_on_clicked": None,
            }
        )
    return metrics

