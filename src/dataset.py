"""Chronological LOB windows, train-fitted normalisation, purged splits.

Winsorisation is enforced at inference too; see README.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

# Dropped before the model sees them: bookkeeping columns, the label, and any absolute
# price level. micro_price is engineered but excluded for that last reason, which is what
# leaves 43 inputs rather than 44.
FORBIDDEN_COLUMNS = {
    "ts_event",
    "ts_in_delta",
    "timestamp_sec",
    "micro_price",
    "mid_price",
    "target_log_return",
    "log_return",
}


@dataclass
class Normalisation:
    """The affine map from raw features to model inputs, serialised next to the weights."""

    feature_names: list[str]
    winsor_lo: list[float]
    winsor_hi: list[float]
    mean: list[float]
    std: list[float]
    target_mean: float
    target_std: float
    winsor_quantile: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        lo = np.asarray(self.winsor_lo, dtype=np.float32)
        hi = np.asarray(self.winsor_hi, dtype=np.float32)
        mu = np.asarray(self.mean, dtype=np.float32)
        sd = np.asarray(self.std, dtype=np.float32)
        return (np.clip(x, lo, hi) - mu) / sd

    def input_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Normalised [x_min, x_max]. Holds for any input because apply() clips first."""
        # float32 throughout, matching apply(). Evaluating in float64 can land the edge one
        # ulp inside the float32 result, which would make the exported box unsound.
        lo = self.apply(np.asarray(self.winsor_lo, dtype=np.float32))
        hi = self.apply(np.asarray(self.winsor_hi, dtype=np.float32))
        return lo, hi

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Normalisation":
        return Normalisation(**d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: Path) -> "Normalisation":
        return Normalisation.from_dict(json.loads(Path(path).read_text()))


class LOBWindows(Dataset):
    """Sliding windows over one contiguous, chronological slice of the series."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        start: int,
        stop: int,
        seq_len: int,
        horizon: int,
        purge: bool,
    ):
        self.features = features
        self.targets = targets
        self.seq_len = seq_len

        # the window [s, s + seq_len - 1] is labelled horizon ticks past its end, so a purged
        # split has to stop horizon short of the boundary or the label crosses it
        last_end = stop - horizon if purge else stop
        self.start = start
        self.n = max(0, (last_end - start) - seq_len + 1)
        if self.n == 0:
            raise ValueError(
                f"empty split: start={start} stop={stop} seq_len={seq_len} horizon={horizon}"
            )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.start + idx
        window = self.features[s : s + self.seq_len]
        label = self.targets[s + self.seq_len - 1]
        return torch.from_numpy(np.ascontiguousarray(window)), torch.tensor(
            label, dtype=torch.float32
        )


class LOBDataModule:
    """Loads the parquet once, fits normalisation on train only, and cuts purged splits."""

    def __init__(self, config: dict):
        data_cfg = config["data"]
        self.seq_len = int(data_cfg["seq_length"])
        self.horizon = int(data_cfg.get("prediction_horizon", 100))
        self.batch_size = int(data_cfg["batch_size"])
        winsor_q = float(data_cfg.get("winsor_quantile", 0.001))

        frac_train = float(data_cfg.get("train_split", 0.70))
        frac_val = float(data_cfg.get("val_split", 0.15))

        df = pl.read_parquet(data_cfg["processed_file"])
        feature_names = [c for c in df.columns if c not in FORBIDDEN_COLUMNS]
        raw = df.select(feature_names).to_numpy().astype(np.float32)
        targets = (
            df.select("target_log_return").to_numpy().flatten().astype(np.float32)
        )

        n = len(raw)
        self.n_rows = n
        self.idx_train = (0, int(n * frac_train))
        self.idx_val = (self.idx_train[1], int(n * (frac_train + frac_val)))
        self.idx_test = (self.idx_val[1], n)

        train_raw = raw[self.idx_train[0] : self.idx_train[1]]

        # limits and statistics from train only; applying them to val/test is fine, fitting
        # them there would be leakage
        lo = np.quantile(train_raw, winsor_q, axis=0).astype(np.float32)
        hi = np.quantile(train_raw, 1.0 - winsor_q, axis=0).astype(np.float32)

        # a near-constant column collapses to lo == hi, which np.clip and the input box both
        # need to be a real interval: widen to min/max, then nudge by one ulp. A fixed 1e-6
        # would vanish in float32 once |lo| exceeds about 32.
        degenerate = hi <= lo
        if degenerate.any():
            lo[degenerate] = train_raw[:, degenerate].min(axis=0)
            hi[degenerate] = train_raw[:, degenerate].max(axis=0)
            still = hi <= lo
            hi[still] = np.nextafter(lo[still], np.float32(np.inf))

        clipped_train = np.clip(train_raw, lo, hi)
        mean = clipped_train.mean(axis=0).astype(np.float32)
        std = np.maximum(clipped_train.std(axis=0), 1e-8).astype(np.float32)

        train_targets = targets[self.idx_train[0] : self.idx_train[1]]
        self.normalisation = Normalisation(
            feature_names=list(feature_names),
            winsor_lo=lo.tolist(),
            winsor_hi=hi.tolist(),
            mean=mean.tolist(),
            std=std.tolist(),
            target_mean=float(train_targets.mean()),
            target_std=float(max(train_targets.std(), 1e-8)),
            winsor_quantile=winsor_q,
        )

        # One clip path only: apply() is what a deployment must run, so the exported
        # input_box() and the training features cannot drift apart.
        self.features = self.normalisation.apply(raw)
        self.targets = (
            (targets - self.normalisation.target_mean) / self.normalisation.target_std
        ).astype(np.float32)

    @property
    def input_dim(self) -> int:
        return self.features.shape[1]

    def dataset(self, split: str) -> LOBWindows:
        start, stop = {
            "train": self.idx_train,
            "val": self.idx_val,
            "test": self.idx_test,
        }[split]
        return LOBWindows(
            self.features,
            self.targets,
            start,
            stop,
            self.seq_len,
            self.horizon,
            # test needs no purge: it ends at the data boundary and 01_build_features.py
            # has already dropped the unlabelled tail rows.
            purge=(split != "test"),
        )

    def loader(
        self,
        split: str,
        shuffle: Optional[bool] = None,
        batch_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> DataLoader:
        if shuffle is None:
            shuffle = split == "train"
        return DataLoader(
            self.dataset(split),
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )

    def split_report(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "horizon": self.horizon,
            "purge_ticks": self.horizon,
            "splits": {
                "train": {"range": list(self.idx_train), "windows": len(self.dataset("train"))},
                "val": {"range": list(self.idx_val), "windows": len(self.dataset("val"))},
                "test": {"range": list(self.idx_test), "windows": len(self.dataset("test"))},
            },
        }


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    dm = LOBDataModule(config)
    return dm.loader("train"), dm.loader("val"), dm.loader("test")
