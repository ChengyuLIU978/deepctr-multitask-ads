from ctr_cvr.data import FeaturePreprocessor, generate_demo_data, temporal_split, validate_and_label
from ctr_cvr.models import SUPPORTED_MODELS, build_model


def test_all_models_build_on_cpu():
    frame = validate_and_label(generate_demo_data(1_000, seed=3))
    split = temporal_split(frame)
    columns = FeaturePreprocessor(feature_set="sparse_only").fit(split.train).feature_columns()
    for name in SUPPORTED_MODELS:
        model = build_model(name, columns, device="cpu", seed=3)
        assert model is not None


def test_mmoe_expert_count_changes_parameter_count():
    frame = validate_and_label(generate_demo_data(1_000, seed=4))
    split = temporal_split(frame)
    columns = FeaturePreprocessor(feature_set="sparse_only").fit(split.train).feature_columns()
    two = build_model("mmoe", columns, device="cpu", seed=4, num_experts=2)
    five = build_model("mmoe", columns, device="cpu", seed=4, num_experts=5)
    assert sum(p.numel() for p in five.parameters()) > sum(p.numel() for p in two.parameters())

