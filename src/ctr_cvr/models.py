"""Thin, explicit model factory around the official DeepCTR-Torch library."""

from __future__ import annotations

from typing import Sequence


SUPPORTED_MODELS = ("deepfm", "esmm", "sharedbottom", "mmoe")
SUPPORTED_CAPACITIES = ("small", "base", "large")

CAPACITY_UNITS = {
    "small": {
        "deepfm": (32, 16),
        "esmm": (32, 16),
        "expert": (32, 16),
        "gate": (8,),
        "tower": (8,),
    },
    "base": {
        "deepfm": (64, 32),
        "esmm": (64, 32),
        "expert": (64, 32),
        "gate": (16,),
        "tower": (16,),
    },
    "large": {
        "deepfm": (128, 64),
        "esmm": (128, 64),
        "expert": (128, 64),
        "gate": (32,),
        "tower": (32,),
    },
}


def build_model(
    name: str,
    feature_columns: Sequence[object],
    device: str,
    seed: int,
    num_experts: int = 3,
    capacity: str = "base",
):
    """Build one of the four deliberately small interview-project models."""

    name = name.lower()
    if capacity not in SUPPORTED_CAPACITIES:
        raise ValueError(f"unsupported capacity {capacity!r}; choose from {SUPPORTED_CAPACITIES}")
    if num_experts < 2:
        raise ValueError("MMoE requires at least two experts")
    units = CAPACITY_UNITS[capacity]
    if name == "deepfm":
        from deepctr_torch.models import DeepFM

        return DeepFM(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            dnn_hidden_units=units["deepfm"],
            dnn_dropout=0.1,
            task="binary",
            seed=seed,
            device=device,
        )
    if name == "esmm":
        from deepctr_torch.models.multitask import ESMM

        return ESMM(
            dnn_feature_columns=feature_columns,
            tower_dnn_hidden_units=units["esmm"],
            dnn_dropout=0.1,
            task_types=("binary", "binary"),
            task_names=("ctr", "ctcvr"),
            seed=seed,
            device=device,
        )
    if name == "sharedbottom":
        from deepctr_torch.models.multitask import SharedBottom

        return SharedBottom(
            dnn_feature_columns=feature_columns,
            bottom_dnn_hidden_units=units["expert"],
            tower_dnn_hidden_units=units["tower"],
            dnn_dropout=0.1,
            task_types=("binary", "binary"),
            task_names=("ctr", "ctcvr"),
            seed=seed,
            device=device,
        )
    if name == "mmoe":
        from deepctr_torch.models.multitask import MMOE

        return MMOE(
            dnn_feature_columns=feature_columns,
            num_experts=num_experts,
            expert_dnn_hidden_units=units["expert"],
            gate_dnn_hidden_units=units["gate"],
            tower_dnn_hidden_units=units["tower"],
            dnn_dropout=0.1,
            task_types=("binary", "binary"),
            task_names=("ctr", "ctcvr"),
            seed=seed,
            device=device,
        )
    raise ValueError(f"unsupported model {name!r}; choose from {SUPPORTED_MODELS}")


def parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))

