"""Evaluate every trained architecture on the held-out test split.

Metric and reporting choices are in docs/09-statistics.md.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import; headless runs have no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from src.dataset import LOBDataModule
from src.models.lstm import LSTMBaseline
from src.models.mamba import LOBMamba
from src.models.transformer import LOBTransformer
from src.utils.config import load_config
from src.utils.repro import provenance, set_seed
from src.utils.stats import diebold_mariano, effective_sample_size, summarise_predictions

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = Path("models/results")
CHECKPOINT_DIR = Path("models/checkpoints")

BUILDERS = {
    "lstm": LSTMBaseline,
    "mamba": LOBMamba,
    "transformer": LOBTransformer,
}

# (display name, checkpoint tag, builder key)
CANDIDATES = [
    ("Mamba (bounded delta)", "mamba", "mamba"),
    ("Mamba (softplus delta)", "mamba_softplus", "mamba"),
    ("Causal Transformer", "transformer", "transformer"),
    ("LSTM", "lstm", "lstm"),
]


def load_checkpoint(tag: str, builder_key: str, input_dim: int, fallback_config: dict):
    path = CHECKPOINT_DIR / f"best_{tag}.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", fallback_config)
    arch = ckpt.get("architecture", {})
    model = BUILDERS[builder_key](
        int(arch.get("input_dim", input_dim)), config
    )
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except (KeyError, RuntimeError) as exc:
        print(f"  skipping {tag}: incompatible checkpoint ({exc.__class__.__name__})")
        return None
    model.eval()
    return {"model": model, "checkpoint": ckpt, "path": path}


@torch.no_grad()
def predict(model, loader, max_batches: int, device) -> tuple[np.ndarray, np.ndarray]:
    preds, targets = [], []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        preds.append(model(x.to(device)).float().cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def fit_ridge(dm: LOBDataModule, n_batches: int, seed: int) -> Ridge:
    """Flattened-window linear baseline. Seeded so the sample is reproducible."""
    generator = torch.Generator().manual_seed(seed)
    loader = dm.loader("train", shuffle=True, generator=generator)
    xs, ys = [], []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        xs.append(x.reshape(x.shape[0], -1).numpy())
        ys.append(y.numpy())
    model = Ridge(alpha=1.0)
    model.fit(np.concatenate(xs), np.concatenate(ys))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on the test split.")
    parser.add_argument("--test-batches", type=int, default=200)
    parser.add_argument("--ridge-batches", type=int, default=40)
    args = parser.parse_args()

    config = load_config()
    seed = int(config.get("seed", 1337))
    set_seed(seed)
    horizon = int(config["data"].get("prediction_horizon", 100))
    device = torch.device("cpu")

    dm = LOBDataModule(config)
    test_loader = dm.loader("test", shuffle=False)
    target_std = dm.normalisation.target_std
    target_mean = dm.normalisation.target_mean

    print("fitting Ridge baseline ...")
    ridge = fit_ridge(dm, args.ridge_batches, seed)

    print(f"running inference on {args.test_batches} test batches ...")
    predictions: dict[str, np.ndarray] = {}
    targets: np.ndarray | None = None
    metadata: dict[str, dict] = {}

    for display, tag, key in CANDIDATES:
        loaded = load_checkpoint(tag, key, dm.input_dim, config)
        if loaded is None:
            print(f"  {display}: no checkpoint, skipped")
            continue
        p, t = predict(loaded["model"], test_loader, args.test_batches, device)
        predictions[display] = p
        targets = t if targets is None else targets
        train_meta = loaded["checkpoint"].get("training", {})
        metadata[display] = {
            "checkpoint": str(loaded["path"]),
            "steps": train_meta.get("step"),
            "n_params": train_meta.get("n_params"),
            "selection_metric": train_meta.get("selection_metric"),
            "best_val_rank_ic": train_meta.get("best_val_rank_ic"),
            "val_loss_at_best": train_meta.get("val_loss_at_best"),
            "architecture": loaded["checkpoint"].get("architecture", {}),
        }
        print(f"  {display}: {len(p)} predictions")

    if targets is None:
        raise SystemExit("no usable checkpoints found; train something first")

    xs = []
    for i, (x, _) in enumerate(test_loader):
        if i >= args.test_batches:
            break
        xs.append(x.reshape(x.shape[0], -1).numpy())
    predictions["Ridge"] = ridge.predict(np.concatenate(xs))
    metadata["Ridge"] = {"n_params": int(ridge.coef_.size)}

    print("\ncomputing overlap-aware metrics ...")
    results = {
        name: summarise_predictions(p, targets, horizon=horizon, seed=seed)
        for name, p in predictions.items()
    }

    for name, p in predictions.items():
        # Undo the full affine label map, not just the scale: targets are (r - mean)/std.
        pnl = np.sign(p) * (targets * target_std + target_mean)
        # forward windows overlap by horizon-1 ticks, so the SE denominator is n_eff, not len(p)
        n_eff = effective_sample_size(len(p), horizon)
        results[name]["mean_pnl_per_decision"] = float(pnl.mean())
        results[name]["information_ratio"] = float(pnl.mean() / (pnl.std() + 1e-12))
        results[name]["information_ratio_se"] = float(1.0 / np.sqrt(n_eff))
        cum = pnl.cumsum()
        results[name]["max_drawdown_logret"] = float(
            (np.maximum.accumulate(cum) - cum).max()
        )

    order = sorted(results, key=lambda k: -results[k]["rank_ic"])
    print(
        f"\n{'model':26s} {'RankIC':>8s} {'95% CI (block boot)':>24s} "
        f"{'naive SE':>9s} {'block SE':>9s} {'hit':>7s}"
    )
    for name in order:
        r = results[name]
        print(
            f"{name:26s} {r['rank_ic']:+8.4f} "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]{'':>6s} "
            f"{r['se_naive_iid']:9.4f} {r['se_block_bootstrap']:9.4f} "
            f"{r['hit_rate']:6.2%}"
        )
    print(
        f"\nn = {len(targets):,} overlapping observations "
        f"-> {effective_sample_size(len(targets), horizon):.0f} independent forward windows"
    )

    names = list(order)
    dm_matrix = {a: {} for a in names}
    for a in names:
        for b in names:
            if a == b:
                dm_matrix[a][b] = {"stat": 0.0, "p_value": 1.0}
            else:
                dm_matrix[a][b] = diebold_mariano(
                    targets, predictions[a], predictions[b], horizon=horizon
                )

    print("\nDiebold-Mariano p-values (Bartlett + HLN, h = %d):" % horizon)
    print(f"{'':26s}" + "".join(f"{n[:12]:>14s}" for n in names))
    for a in names:
        row = "".join(f"{dm_matrix[a][b]['p_value']:14.3f}" for b in names)
        print(f"{a:26s}{row}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rolling_vol = (
        pd.Series(targets).rolling(window=horizon, min_periods=1).std().bfill().values
    )
    # Columns stay in normalised target units; multiply by target_std and add target_mean
    # for log returns. The JSON metrics beside this are already de-normalised.
    frame = {"target": targets, "volatility": rolling_vol, **predictions}
    pd.DataFrame(frame).to_parquet(RESULTS_DIR / "out_of_sample_predictions.parquet")

    (RESULTS_DIR / "test_metrics.json").write_text(
        json.dumps(
            {
                "provenance": provenance(seed),
                "split": "test",
                "n_observations": int(len(targets)),
                "n_effective": effective_sample_size(len(targets), horizon),
                "horizon": horizon,
                "target_std_logret": target_std,
                "models": metadata,
                "metrics": results,
                "diebold_mariano": dm_matrix,
            },
            indent=2,
        )
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [1.3, 1]}
    )
    fig.suptitle(
        "Held-out test split: LOB return forecasting", fontsize=16, fontweight="bold"
    )

    for name in order:
        pnl = (np.sign(predictions[name]) * (targets * target_std + target_mean)).cumsum()
        ax1.plot(pnl, linewidth=2.0, label=f"{name} (rank IC {results[name]['rank_ic']:+.4f})")
    ax1.axhline(0, color="#adb5bd", linewidth=0.8)
    ax1.set_title("Cumulative sign-following PnL, log-return units", fontsize=12)
    ax1.set_xlabel("test observation")
    ax1.set_ylabel("cumulative log return")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.axis("off")
    header = ["Model", "Rank IC", "95% CI (block)", "naive SE", "block SE", "Hit", "MSE"]
    rows = [
        [
            name,
            f"{results[name]['rank_ic']:+.4f}",
            f"[{results[name]['ci_lo']:+.4f}, {results[name]['ci_hi']:+.4f}]",
            f"{results[name]['se_naive_iid']:.4f}",
            f"{results[name]['se_block_bootstrap']:.4f}",
            f"{results[name]['hit_rate']:.2%}",
            f"{results[name]['mse']:.4f}",
        ]
        for name in order
    ]
    table = ax2.table(cellText=[header, *rows], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#dee2e6")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f1f3f5")
    ax2.set_title(
        f"n = {len(targets):,} overlapping observations "
        f"({effective_sample_size(len(targets), horizon):.0f} independent windows). "
        "Overlapping windows: read the block SE, not the naive one.",
        fontsize=10,
        pad=18,
    )

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=200, bbox_inches="tight")
    print(f"\nwrote {RESULTS_DIR}/test_metrics.json, predictions parquet, model_comparison.png")


if __name__ == "__main__":
    main()
