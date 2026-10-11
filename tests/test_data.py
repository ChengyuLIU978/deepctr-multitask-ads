import numpy as np

from ctr_cvr.data import (
    FeaturePreprocessor,
    generate_demo_data,
    temporal_split,
    validate_and_label,
)


def test_label_contract_and_temporal_split():
    frame = validate_and_label(generate_demo_data(2_000, seed=7))
    assert ((frame["ctcvr"] == 1) <= (frame["click"] == 1)).all()
    split = temporal_split(frame)
    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()
    assert len(split.train) + len(split.validation) + len(split.test) == len(frame)


def test_preprocessor_uses_train_only_statistics():
    frame = validate_and_label(generate_demo_data(2_000, seed=11))
    split = temporal_split(frame)
    processor = FeaturePreprocessor().fit(split.train)
    original_mean = processor.dense_means["cost_log"]
    future = split.test.copy()
    future["cost"] = 1_000_000.0
    transformed = processor.transform(future)
    assert processor.dense_means["cost_log"] == original_mean
    assert transformed["cost_log"].mean() > 1.0
    assert all(np.issubdtype(values.dtype, np.number) for values in transformed.values())


def test_preprocessor_manifest_round_trip():
    frame = validate_and_label(generate_demo_data(2_000, seed=17))
    split = temporal_split(frame)
    original = FeaturePreprocessor().fit(split.train)
    restored = FeaturePreprocessor.from_dict(original.to_dict())
    original_features = original.transform(split.test)
    restored_features = restored.transform(split.test)
    for name in original_features:
        np.testing.assert_array_equal(original_features[name], restored_features[name])

