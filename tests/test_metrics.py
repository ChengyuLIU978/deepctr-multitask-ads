import numpy as np

from ctr_cvr.metrics import evaluate_ctr, evaluate_multitask


def test_ctr_metrics_are_finite():
    result = evaluate_ctr(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert result["ctr_auc"] == 1.0
    assert result["ctr_logloss"] > 0


def test_multitask_reports_probability_violations_and_cvr():
    y_ctr = np.array([0, 1, 1, 0, 1, 1])
    y_ctcvr = np.array([0, 0, 1, 0, 0, 1])
    p_ctr = np.array([0.1, 0.7, 0.8, 0.2, 0.6, 0.9])
    p_ctcvr = np.array([0.2, 0.1, 0.5, 0.1, 0.2, 0.7])
    result = evaluate_multitask(y_ctr, y_ctcvr, p_ctr, p_ctcvr)
    assert result["clicked_rows"] == 4
    assert result["probability_order_violation_rate"] > 0
    assert result["cvr_auc_on_clicked"] is not None

