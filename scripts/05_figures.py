"""Publication figures: forecast comparisons and SSM certification plots."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from src.models.mamba import LOBMamba
from src.models.transformer import LOBTransformer
from src.utils.config import load_config
from src.utils.stats import diebold_mariano
from src.verification.certify import certify_block, certify_model

OUT_DIR = Path("docs/figures")
RESULTS = Path("models/results/out_of_sample_predictions.parquet")
CHECKPOINT_DIR = Path("models/checkpoints")

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    }
)


def load_mamba(config: dict, dt_parametrisation: str = "bounded") -> LOBMamba:
    """Load the trained model for that parametrisation, falling back to a fresh one.

    The bounds depend on the weights, so a fresh fallback will not reproduce the paper's numbers.
    """
    cfg = json.loads(json.dumps(config))
    cfg["model"]["mamba"]["dt_parametrisation"] = dt_parametrisation
    tag = "mamba" if dt_parametrisation == "bounded" else "mamba_softplus"
    path = CHECKPOINT_DIR / f"best_{tag}.pt"

    if path.exists():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        arch = ckpt.get("architecture", {})
        model = LOBMamba(int(arch.get("input_dim", 43)), cfg)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            return model.eval()
        except (KeyError, RuntimeError):
            print(f"  {path.name} incompatible; using a fresh model for this figure")
    else:
        print(f"  {path.name} not found; using a fresh model for this figure")
    return LOBMamba(43, cfg).eval()


def fig_bound_vs_floor(config: dict) -> None:
    """The invariant does not use the timescale floor; the geometric bound pays for it."""
    from src.verification.certify import certify_block
    from src.verification.invariant import certify_block_state

    floors = [1e-5, 1e-4, 1e-3, 1e-2]
    invariant, geometric = [], []
    for floor in floors:
        cfg = json.loads(json.dumps(config))
        cfg["model"]["mamba"]["dt_min"] = floor
        torch.manual_seed(int(config.get("seed", 1337)))
        block = LOBMamba(43, cfg).eval().layers[0]
        cert = certify_block(block, seq_len=100)
        invariant.append(certify_block_state(block, cert["boxes"]["u"], cert["boxes"]["B"])["radius"])
        geometric.append(cert["summary"]["h_abs_max_geometric"])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.loglog(floors, geometric, marker="o", linewidth=2.2, color="#c92a2a",
              label="geometric bound")
    ax.loglog(floors, invariant, marker="s", linewidth=2.2, color="#2b8a3e",
              label="invariant")
    ax.set_xlabel(r"timescale floor $\delta_{\min}$")
    ax.set_ylabel(r"certified $\sup|h|$")
    ax.set_title("The geometric bound pays for the floor and the invariant does not\n"
                 r"gap $=(1-e^{-\lambda d_{hi}})/(1-e^{-\lambda d_{lo}})$", fontsize=12)
    ax.legend(fontsize=9)
    fig.savefig(OUT_DIR / "bound_vs_floor.pdf")
    fig.savefig(OUT_DIR / "bound_vs_floor.png", dpi=200)
    plt.close(fig)


def fig_bound_vs_horizon(config: dict) -> None:
    """One induction step covers every length; the unrolled bound is recomputed and climbs."""
    from src.verification.certify import certify_block
    from src.verification.invariant import certify_block_state
    from src.verification import intervals as iv
    from src.verification.intervals import Interval

    model = load_mamba(config)
    block = model.layers[0]
    base = certify_block(block, seq_len=8)
    radius = certify_block_state(block, base["boxes"]["u"], base["boxes"]["B"])["radius"]
    drive = base["boxes"]["B_bar"] * Interval(
        base["boxes"]["u"].lo.unsqueeze(-1), base["boxes"]["u"].hi.unsqueeze(-1)
    )
    widened = float(iv.state_bound_widened(base["boxes"]["A_bar"], drive)[0].abs_max().max())

    lengths = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    unrolled = [certify_block(block, seq_len=L)["summary"]["h_abs_max_horizon"] for L in lengths]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.loglog(lengths, unrolled, marker="o", markersize=4, linewidth=2, color="#c92a2a",
              label="unrolled, recomputed per length")
    ax.axhline(widened, color="#c92a2a", linestyle="--", linewidth=1.2,
               label="its fixed point")
    ax.axhline(radius, color="#2b8a3e", linewidth=2.2, label="invariant, one induction step")
    ax.set_xlabel("sequence length $L$")
    ax.set_ylabel(r"certified $\sup|h|$")
    ax.set_title("The invariant does not depend on sequence length", fontsize=12)
    ax.legend(fontsize=9)
    fig.savefig(OUT_DIR / "bound_vs_horizon.pdf")
    fig.savefig(OUT_DIR / "bound_vs_horizon.png", dpi=200)
    plt.close(fig)


def fig_certified_vs_realised(certificate_path: Path) -> None:
    """Looseness of the certificate, broken down by quantity."""
    if not certificate_path.exists():
        print(f"  skipping looseness figure: {certificate_path} not found")
        return
    art = json.loads(certificate_path.read_text())
    rows = art["looseness"]
    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    quantities = list(dict.fromkeys(df["quantity"]))
    width = 0.38
    x = np.arange(len(quantities))
    for li, layer in enumerate(sorted(df["layer"].unique())):
        sub = df[df["layer"] == layer].set_index("quantity").reindex(quantities)
        ax.bar(x + li * width, sub["looseness"], width, label=f"layer {layer}")
    ax.set_yscale("log")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(quantities)
    ax.axhline(1.0, color="#212529", linewidth=1.0)
    ax.set_ylabel("certified radius / realised radius")
    ax.set_title(
        "Looseness of the interval certificate by quantity\n"
        "(1.0 = tight; the state recursion is where interval arithmetic pays)", fontsize=12,
    )
    ax.legend(fontsize=9)
    fig.savefig(OUT_DIR / "certified_vs_realised.pdf")
    fig.savefig(OUT_DIR / "certified_vs_realised.png", dpi=200)
    plt.close(fig)


def fig_output_range(config: dict) -> None:
    """Empirical output ranges over six input scales, against the certified range."""
    model = load_mamba(config)
    cert = certify_model(model, seq_len=100)
    lo, hi = cert["output_range"]["lo"], cert["output_range"]["hi"]

    scales = [1e-3, 1e-1, 1.0, 1e1, 1e3, 1e6]
    ranges = []
    with torch.no_grad():
        for s in scales:
            out = model(torch.randn(64, 100, model.input_dim) * s)
            ranges.append((out.min().item(), out.max().item()))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhspan(lo, hi, color="#d3f9d8", label=f"certified range [{lo:.2f}, {hi:.2f}]")
    ax.axhline(lo, color="#2b8a3e", linewidth=1.6)
    ax.axhline(hi, color="#2b8a3e", linewidth=1.6)
    for i, (a, b) in enumerate(ranges):
        ax.plot([i, i], [a, b], linewidth=6, color="#1971c2", solid_capstyle="round")
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f"$10^{{{int(np.log10(s))}}}$" for s in scales])
    ax.set_xlabel("input scale (standard deviations of white noise)")
    ax.set_ylabel("model output")
    ax.set_title(
        "The final LayerNorm bounds the output for every input, however extreme", fontsize=12
    )
    ax.legend(fontsize=9, loc="upper right")
    fig.savefig(OUT_DIR / "certified_output_range.pdf")
    fig.savefig(OUT_DIR / "certified_output_range.png", dpi=200)
    plt.close(fig)


def fig_dm_heatmap(df: pd.DataFrame, horizon: int) -> None:
    models = [c for c in df.columns if c not in ("target", "timestamp", "volatility")]
    n = len(models)
    p = np.ones((n, n))
    y = df["target"].to_numpy()
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i != j:
                p[i, j] = diebold_mariano(y, df[a].to_numpy(), df[b].to_numpy(), horizon)["p_value"]

    fig, ax = plt.subplots(figsize=(1.6 * n + 2, 1.3 * n + 1.5))
    im = ax.imshow(p, cmap="coolwarm_r", vmin=0, vmax=0.1)
    ax.set_xticks(range(n), models, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n), models, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{p[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(
        f"Diebold-Mariano p-values\nBartlett-weighted, HLN-corrected, h = {horizon}", fontsize=11
    )
    fig.colorbar(im, ax=ax, label="p-value")
    fig.savefig(OUT_DIR / "dm_test_heatmap.pdf")
    fig.savefig(OUT_DIR / "dm_test_heatmap.png", dpi=200)
    plt.close(fig)


def fig_regime_ic(df: pd.DataFrame) -> None:
    if "volatility" not in df.columns:
        return
    d = df.copy()
    d["Regime"] = pd.qcut(d["volatility"], q=4, labels=["Quiet", "Normal", "Volatile", "Extreme"])
    models = [c for c in d.columns if c not in ("target", "timestamp", "volatility", "Regime")]

    regimes = ["Quiet", "Normal", "Volatile", "Extreme"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(models)
    x = np.arange(len(regimes))
    for mi, m in enumerate(models):
        vals = [
            spearmanr(d[d["Regime"] == r]["target"], d[d["Regime"] == r][m]).statistic
            for r in regimes
        ]
        ax.bar(x + mi * width, vals, width, label=m)
    ax.set_xticks(x + 0.4 - width / 2, regimes)
    ax.axhline(0, color="#212529", linewidth=0.9)
    ax.set_ylabel("Spearman rank IC")
    ax.set_title("Rank IC conditioned on realised volatility regime", fontsize=12)
    ax.legend(fontsize=8)
    fig.savefig(OUT_DIR / "regime_conditioned_ic.pdf")
    fig.savefig(OUT_DIR / "regime_conditioned_ic.png", dpi=200)
    plt.close(fig)


def fig_latency(config: dict, lengths=(64, 128, 256, 512, 1024)) -> None:
    """Wall-clock forward-pass latency; the scan is an unfused Python loop, so the constants
    are implementation-specific and not a fair O(L) vs O(L^2) comparison."""
    mamba = load_mamba(config)
    input_dim = mamba.input_dim
    models = {
        "Mamba (unrolled Python scan)": mamba,
        "Causal Transformer": LOBTransformer(input_dim, config=config).eval(),
    }
    results = {k: [] for k in models}
    for L in lengths:
        x = torch.randn(1, L, input_dim)
        for name, model in models.items():
            with torch.no_grad():
                for _ in range(3):
                    model(x)
                t0 = time.perf_counter()
                for _ in range(10):
                    model(x)
                results[name].append((time.perf_counter() - t0) / 10 * 1000)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, lat in results.items():
        ax.plot(lengths, lat, marker="o", linewidth=2, label=name)
    ref = np.array(lengths, dtype=float)
    ax.plot(lengths, results["Causal Transformer"][0] * (ref / ref[0]) ** 2,
            linestyle=":", color="#adb5bd", label=r"$O(L^2)$ reference")
    ax.plot(lengths, results["Mamba (unrolled Python scan)"][0] * (ref / ref[0]),
            linestyle="--", color="#adb5bd", label=r"$O(L)$ reference")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context length $L$")
    ax.set_ylabel("forward-pass latency (ms)")
    ax.set_title(
        "Forward-pass latency, CPU\n"
        "The scan is an unfused Python loop; constants are not comparable "
        "to a CUDA kernel", fontsize=11,
    )
    ax.legend(fontsize=8)
    fig.savefig(OUT_DIR / "latency_complexity.pdf")
    fig.savefig(OUT_DIR / "latency_complexity.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--certificate", default="models/bounds/mamba_certificate.json")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    config = load_config()
    horizon = int(config["data"].get("prediction_horizon", 100))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("certification figures ...")
    fig_bound_vs_floor(config)
    fig_bound_vs_horizon(config)
    fig_output_range(config)
    fig_certified_vs_realised(Path(args.certificate))

    if RESULTS.exists():
        print("statistical figures ...")
        df = pd.read_parquet(RESULTS)
        fig_dm_heatmap(df, horizon)
        fig_regime_ic(df)
    else:
        print(f"  {RESULTS} not found; run 03_evaluate_models.py first")

    if not args.skip_latency:
        print("latency figure ...")
        fig_latency(config)

    print(f"\nwrote figures to {OUT_DIR}/")
    for p in sorted(OUT_DIR.glob("*.pdf")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
