"""Standardisation in src/dataset.py: no silent floor on a column's spread."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.dataset import DegenerateColumnError, LOBDataModule, check_spread


def _write(tmp_path: Path, columns: dict[str, np.ndarray], n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    frame = {"ts_event": np.arange(n, dtype=np.int64)}
    frame.update(columns)
    frame["target_log_return"] = rng.normal(0.0, 1e-4, n)
    path = tmp_path / "features.parquet"
    pl.DataFrame(frame).write_parquet(path)
    return {
        "data": {
            "processed_file": str(path),
            "batch_size": 8,
            "seq_length": 5,
            "prediction_horizon": 3,
            "train_split": 0.7,
            "val_split": 0.15,
            "winsor_quantile": 0.001,
        }
    }


def test_small_units_are_standardised_not_floored(tmp_path):
    # Spreads of order 1e-11, the scale of the price columns when prices were 1e9 too small.
    # The former absolute floor of 1e-8 divided these by 1e-8 and left them at sd ~1e-3.
    n = 4000
    rng = np.random.default_rng(1)
    tiny = rng.normal(0.0, 1e-11, n)
    plain = rng.normal(5.0, 2.0, n)
    dm = LOBDataModule(_write(tmp_path, {"tiny": tiny, "plain": plain}, n))
    a, b = dm.idx_train
    train = dm.features[a:b]
    assert 5e-12 < dm.normalisation.std[0] < 2e-11
    np.testing.assert_allclose(train.std(axis=0), 1.0, rtol=1e-3)
    np.testing.assert_allclose(train.mean(axis=0), 0.0, atol=1e-3)


def test_same_column_in_any_unit_gives_the_same_inputs(tmp_path):
    n = 4000
    rng = np.random.default_rng(2)
    dollars = rng.normal(0.0, 0.01, n)
    out = []
    for k, scale in enumerate((1.0, 1e-9)):
        d = tmp_path / str(k)
        d.mkdir()
        out.append(LOBDataModule(_write(d, {"spread": dollars * scale}, n)).features)
    np.testing.assert_allclose(out[0], out[1], atol=1e-5)


def test_constant_column_is_refused(tmp_path):
    n = 4000
    rng = np.random.default_rng(3)
    cfg = _write(tmp_path, {"ok": rng.normal(size=n), "flat": np.full(n, 0.01)}, n)
    with pytest.raises(DegenerateColumnError, match="flat"):
        LOBDataModule(cfg)


def test_spread_at_rounding_level_is_refused(tmp_path):
    # Varies by about 1e-7 of its magnitude in float64, which float32 cannot represent.
    n = 4000
    rng = np.random.default_rng(4)
    cfg = _write(tmp_path, {"level": 400.0 + rng.normal(0.0, 4e-5, n)}, n)
    with pytest.raises(DegenerateColumnError, match="level"):
        LOBDataModule(cfg)


def test_check_spread_threshold_is_relative():
    names = ["a", "b"]
    check_spread(np.array([1e-12, 1.0]), np.array([1e-11, 10.0]), names)
    with pytest.raises(DegenerateColumnError, match=r"\bb\b"):
        check_spread(np.array([1e-12, 1e-5]), np.array([1e-11, 10.0]), names)
    with pytest.raises(DegenerateColumnError):
        check_spread(np.array([0.0]), np.array([0.0]), ["zero"])
    with pytest.raises(DegenerateColumnError):
        check_spread(np.array([np.nan]), np.array([1.0]), ["nan"])
