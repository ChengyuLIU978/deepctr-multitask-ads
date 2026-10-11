"""Data contracts, leakage-safe feature processing, and temporal splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SPARSE_FEATURES = ["uid", "campaign", *[f"cat{i}" for i in range(1, 10)]]
DENSE_FEATURES = ["cost_log", "hour_sin", "hour_cos", "elapsed_days"]
RAW_REQUIRED_COLUMNS = ["timestamp", "click", "conversion", "cost", *SPARSE_FEATURES]

# These columns are deliberately never model inputs: they are known only after
# the impression/click, or directly encode an attribution outcome.
LEAKAGE_COLUMNS = {
    "conversion_timestamp",
    "conversion_id",
    "attribution",
    "click_pos",
    "click_nb",
    "cpo",
    "time_since_last_click",
}

DEFAULT_BUCKETS = {
    "uid": 100_000,
    "campaign": 50_000,
    **{f"cat{i}": 20_000 for i in range(1, 10)},
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_demo_data(n_rows: int = 12_000, seed: int = 2026) -> pd.DataFrame:
    """Generate a small *synthetic* dataset only for executable smoke tests.

    The schema mirrors the legal Criteo source, but the rows are not Criteo
    observations and must never be reported as a real-data experiment.
    """

    if n_rows < 1_000:
        raise ValueError("n_rows must be at least 1000 so every split is meaningful")
    rng = np.random.default_rng(seed)
    timestamp = np.arange(n_rows, dtype=np.int64) * 60
    campaign = rng.integers(1, 80, size=n_rows)
    uid = rng.integers(1, 2_500, size=n_rows)
    cost = rng.lognormal(mean=-7.5, sigma=0.7, size=n_rows)

    data: dict[str, np.ndarray] = {
        "timestamp": timestamp,
        "uid": uid,
        "campaign": campaign,
        "cost": cost,
    }
    for i in range(1, 10):
        data[f"cat{i}"] = rng.integers(1, 30 + i * 7, size=n_rows)

    hour = (timestamp // 3600) % 24
    ctr_logit = (
        -2.35
        + 0.45 * np.isin(campaign, [3, 7, 18, 42])
        + 0.35 * np.isin(data["cat3"], [2, 5, 11])
        + 0.25 * np.cos(2 * np.pi * hour / 24)
        + rng.normal(0, 0.25, n_rows)
    )
    click = rng.binomial(1, _sigmoid(ctr_logit)).astype(np.int8)

    cvr_logit = (
        -2.0
        + 0.75 * np.isin(campaign, [7, 18])
        + 0.55 * np.isin(data["cat5"], [4, 12, 20])
        + 0.20 * np.sin(2 * np.pi * hour / 24)
        + rng.normal(0, 0.20, n_rows)
    )
    post_click_conversion = rng.binomial(1, _sigmoid(cvr_logit)).astype(np.int8)
    conversion = (click * post_click_conversion).astype(np.int8)
    data["click"] = click
    data["conversion"] = conversion
    return pd.DataFrame(data)


def load_frame(path: str | Path) -> pd.DataFrame:
    """Load a prepared CSV/TSV file; compressed files are detected by pandas."""

    path = Path(path)
    suffixes = {suffix.lower() for suffix in path.suffixes}
    sep = "\t" if ".tsv" in suffixes else ","
    return pd.read_csv(path, sep=sep, compression="infer")


def validate_and_label(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce the data contract and construct consistent CTR/CTCVR labels.

    `conversion` can include a non-click/view-through event. ESMM requires the
    conversion event to be a subset of clicks, so the project label is:
    y_ctcvr = 1 iff click == 1 AND conversion == 1.
    """

    missing = sorted(set(RAW_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame.empty:
        raise ValueError("input dataset is empty")

    result = frame.copy()
    for label in ("click", "conversion"):
        values = pd.to_numeric(result[label], errors="raise").astype(np.int8)
        invalid = set(values.unique()) - {0, 1}
        if invalid:
            raise ValueError(f"{label} must be binary, got {sorted(invalid)}")
        result[label] = values

    result["timestamp"] = pd.to_numeric(result["timestamp"], errors="raise").astype(np.int64)
    result["cost"] = pd.to_numeric(result["cost"], errors="coerce").fillna(0.0)
    if (result["cost"] < 0).any():
        raise ValueError("cost must be non-negative")

    result["ctcvr"] = ((result["click"] == 1) & (result["conversion"] == 1)).astype(np.int8)
    result = result.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return result


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end_timestamp: int
    validation_end_timestamp: int


def temporal_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """Split chronologically and keep equal timestamps in the same partition."""

    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n_rows = len(ordered)
    if n_rows < 30:
        raise ValueError("at least 30 rows are required for a temporal split")

    timestamps = ordered["timestamp"].to_numpy()
    train_probe = max(0, int(n_rows * train_fraction) - 1)
    validation_probe = max(train_probe + 1, int(n_rows * (train_fraction + validation_fraction)) - 1)
    train_end = int(np.searchsorted(timestamps, timestamps[train_probe], side="right"))
    validation_end = int(np.searchsorted(timestamps, timestamps[validation_probe], side="right"))
    if not (0 < train_end < validation_end < n_rows):
        raise ValueError("timestamp distribution cannot form three non-empty partitions")

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()
    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
        train_end_timestamp=int(train["timestamp"].max()),
        validation_end_timestamp=int(validation["timestamp"].max()),
    )


class FeaturePreprocessor:
    """Fixed hashing for sparse fields plus train-only scaling for dense fields."""

    def __init__(
        self,
        feature_set: str = "all",
        bucket_sizes: dict[str, int] | None = None,
        embedding_dim: int = 8,
    ) -> None:
        if feature_set not in {"all", "sparse_only"}:
            raise ValueError("feature_set must be 'all' or 'sparse_only'")
        self.feature_set = feature_set
        self.bucket_sizes = dict(bucket_sizes or DEFAULT_BUCKETS)
        self.embedding_dim = embedding_dim
        self.dense_means: dict[str, float] = {}
        self.dense_stds: dict[str, float] = {}
        self.origin_timestamp: int | None = None
        self._fitted = False

    @property
    def sparse_features(self) -> list[str]:
        return list(SPARSE_FEATURES)

    @property
    def dense_features(self) -> list[str]:
        return list(DENSE_FEATURES) if self.feature_set == "all" else []

    def _dense_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.origin_timestamp is None:
            raise RuntimeError("preprocessor has not been fitted")
        timestamp = frame["timestamp"].to_numpy(dtype=np.float64)
        seconds_in_day = np.mod(timestamp, 86_400.0)
        radians = 2.0 * np.pi * seconds_in_day / 86_400.0
        cost = frame["cost"].to_numpy(dtype=np.float64)
        return pd.DataFrame(
            {
                "cost_log": np.log1p(np.clip(cost, 0, None) * 1_000_000.0),
                "hour_sin": np.sin(radians),
                "hour_cos": np.cos(radians),
                "elapsed_days": (timestamp - float(self.origin_timestamp)) / 86_400.0,
            },
            index=frame.index,
        )

    def fit(self, train: pd.DataFrame) -> "FeaturePreprocessor":
        """Fit only statistics that are allowed to see the training period."""

        if train.empty:
            raise ValueError("cannot fit on an empty training partition")
        self.origin_timestamp = int(train["timestamp"].min())
        dense = self._dense_frame(train)
        for name in DENSE_FEATURES:
            mean = float(dense[name].mean())
            std = float(dense[name].std(ddof=0))
            self.dense_means[name] = mean
            self.dense_stds[name] = std if std > 1e-8 else 1.0
        self._fitted = True
        return self

    @staticmethod
    def _hash_series(series: pd.Series, bucket_size: int) -> np.ndarray:
        if bucket_size < 2:
            raise ValueError("bucket_size must be at least 2")
        output = np.zeros(len(series), dtype=np.int64)
        valid = series.notna().to_numpy()
        if not valid.any():
            return output
        valid_values = series.loc[valid]
        numeric = pd.to_numeric(valid_values, errors="coerce")
        if numeric.notna().all():
            raw = numeric.to_numpy(dtype=np.int64, copy=False).astype(np.uint64, copy=False)
        else:
            raw = pd.util.hash_pandas_object(valid_values.astype(str), index=False).to_numpy(dtype=np.uint64)
        output[valid] = (raw % np.uint64(bucket_size)).astype(np.int64) + 1
        return output

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("call fit(train) before transform")
        features: dict[str, np.ndarray] = {}
        for name in self.sparse_features:
            features[name] = self._hash_series(frame[name], self.bucket_sizes[name])
        if self.dense_features:
            dense = self._dense_frame(frame)
            for name in self.dense_features:
                values = (dense[name].to_numpy() - self.dense_means[name]) / self.dense_stds[name]
                features[name] = values.astype(np.float32)
        return features

    def feature_columns(self) -> list[object]:
        """Create DeepCTR-Torch feature-column objects lazily."""

        from deepctr_torch.inputs import DenseFeat, SparseFeat

        columns: list[object] = [
            SparseFeat(
                name,
                vocabulary_size=self.bucket_sizes[name] + 1,
                embedding_dim=self.embedding_dim,
            )
            for name in self.sparse_features
        ]
        columns.extend(DenseFeat(name, 1) for name in self.dense_features)
        return columns

    def to_dict(self) -> dict[str, object]:
        if not self._fitted:
            raise RuntimeError("preprocessor has not been fitted")
        return {
            "feature_set": self.feature_set,
            "bucket_sizes": self.bucket_sizes,
            "embedding_dim": self.embedding_dim,
            "origin_timestamp": self.origin_timestamp,
            "dense_means": self.dense_means,
            "dense_stds": self.dense_stds,
            "excluded_leakage_columns": sorted(LEAKAGE_COLUMNS),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FeaturePreprocessor":
        """Restore a fitted preprocessor from an experiment manifest."""

        processor = cls(
            feature_set=str(payload["feature_set"]),
            bucket_sizes={str(k): int(v) for k, v in dict(payload["bucket_sizes"]).items()},
            embedding_dim=int(payload["embedding_dim"]),
        )
        processor.origin_timestamp = int(payload["origin_timestamp"])
        processor.dense_means = {
            str(k): float(v) for k, v in dict(payload["dense_means"]).items()
        }
        processor.dense_stds = {
            str(k): float(v) for k, v in dict(payload["dense_stds"]).items()
        }
        processor._fitted = True
        return processor


def labels(frame: pd.DataFrame, names: Iterable[str]) -> np.ndarray:
    selected = list(names)
    values = frame[selected].to_numpy(dtype=np.float32)
    return values[:, 0] if len(selected) == 1 else values


def partition_summary(frame: pd.DataFrame) -> dict[str, int | float]:
    clicked = frame["click"] == 1
    return {
        "rows": int(len(frame)),
        "start_timestamp": int(frame["timestamp"].min()),
        "end_timestamp": int(frame["timestamp"].max()),
        "clicks": int(frame["click"].sum()),
        "raw_conversions": int(frame["conversion"].sum()),
        "post_click_conversions": int(frame["ctcvr"].sum()),
        "view_through_conversions_excluded": int(((frame["conversion"] == 1) & ~clicked).sum()),
        "ctr": float(frame["click"].mean()),
        "ctcvr": float(frame["ctcvr"].mean()),
        "cvr_on_clicked": float(frame.loc[clicked, "ctcvr"].mean()) if clicked.any() else float("nan"),
    }

