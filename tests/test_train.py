import argparse

import numpy as np
import pandas as pd
import pytest

from ctr_cvr.train import MultitaskValidationDiagnostics, parse_task_loss_weights


def test_task_loss_weights_keep_constant_total_scale():
    ctr, ctcvr = parse_task_loss_weights("3,1")
    assert ctr == pytest.approx(1.5)
    assert ctcvr == pytest.approx(0.5)
    assert ctr + ctcvr == pytest.approx(2.0)


@pytest.mark.parametrize("value", ["1", "1,0", "a,b", "-1,2"])
def test_task_loss_weights_reject_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_task_loss_weights(value)


def test_validation_diagnostics_restore_training_mode():
    class FakeModel:
        training = True

        def predict(self, features, batch_size):
            del features, batch_size
            self.training = False
            return np.array([[0.2, 0.1], [0.8, 0.4], [0.7, 0.5], [0.1, 0.05]])

        def train(self):
            self.training = True

    frame = pd.DataFrame({"click": [0, 1, 1, 0], "ctcvr": [0, 0, 1, 0]})
    callback = MultitaskValidationDiagnostics({}, frame, batch_size=4)
    model = FakeModel()
    callback.set_model(model)
    logs = {}
    callback.on_epoch_end(0, logs)
    assert model.training is True
    assert "val_balanced_logloss" in logs
