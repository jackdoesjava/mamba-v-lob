"""Publication figures: forecast comparisons and SSM certification plots.

No figure carries a title. What each one shows is said in its caption in the paper, so
the image does not repeat the sentence printed underneath it.
"""

from __future__ import annotations

import argparse
import json
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
from src.verification import intervals as iv
from src.verification.certify import certify_block, certify_model
from src.verification.intervals import Interval
from src.verification.invariant import abstraction_gap, certify_block_state

OUT_DIR = Path("docs/figures")
RESULTS = Path("models/results/out_of_sample_predictions.parquet")
SWEEP = Path("docs/certification/sweep.json")
CHECKPOINT_DIR = Path("models/checkpoints")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
        "pdf.fonttype": 42,
    }
)

# The text block of an 11pt article with 1in margins is 6.5in wide.
WIDE = (6.5, 4.0)
HALF = (6.5, 3.1)

# Blue against orange rather than red against green: the pair survives deuteranopia.
OURS = "#1f77b4"
OTHER = "#ff7f0e"
INK = "#212529"
FAINT = "#adb5bd"


def save(fig, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=200)
    plt.close(fig)


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

    fig, ax = plt.subplots(figsize=WIDE)
    ax.loglog(floors, geometric, marker="o", linewidth=2, color=OTHER, label="geometric bound")
    ax.loglog(floors, invariant, marker="s", linewidth=2, color=OURS, label="invariant")
    ax.set_xlabel(r"timescale floor $\delta_{\min}$")
    ax.set_ylabel(r"certified $\sup|h|$")
    ax.legend()
    save(fig, "bound_vs_floor")


def fig_bound_vs_horizon(config: dict) -> None:
    """One induction step covers every length; the unrolled bound is recomputed and climbs."""
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

    fig, ax = plt.subplots(figsize=WIDE)
    ax.loglog(lengths, unrolled, marker="o", markersize=4, linewidth=2, color=OTHER,
              label="unrolled, recomputed per length")
    ax.axhline(widened, color=OTHER, linestyle="--", linewidth=1.2, label="its fixed point")
    ax.axhline(radius, color=OURS, linewidth=2.2, label="invariant, one induction step")
    ax.set_xlabel("sequence length $L$")
    ax.set_ylabel(r"certified $\sup|h|$")
    ax.legend()
    save(fig, "bound_vs_horizon")


@torch.no_grad()
def fig_gap_prediction(config: dict) -> None:
    """The closed-form gap against the ratio the two bounds actually produce, per element.

    Left: the ratio against the pole, with the closed form drawn over the propagated
    timescale box. Right: predicted against measured, which should sit on the diagonal.
    One point per (channel, state) in every layer of the trained model.
    """
    model = load_mamba(config)
    seq_len = int(config["verification"]["seq_len"])
    markers = ["o", "s", "^", "D"]

    fig, (left, right) = plt.subplots(1, 2, figsize=HALF)
    worst, lam_lo, lam_hi = 0.0, float("inf"), 0.0
    boxes = []
    for li, block in enumerate(model.layers):
        cert = certify_block(block, seq_len=seq_len)
        state = certify_block_state(block, cert["boxes"]["u"], cert["boxes"]["B"])
        A = -torch.exp(block.A_log).detach()
        measured = cert["boxes"]["h_geometric"].abs_max() / state["invariant"].abs_max().clamp(min=1e-30)
        predicted = abstraction_gap(A, cert["boxes"]["delta"])
        worst = max(worst, float(((measured - predicted).abs() / predicted).max()))
        boxes.append(cert["boxes"]["delta"])

        lam, measured, predicted = (
            t.detach().flatten().numpy() for t in (A.abs(), measured, predicted)
        )
        lam_lo, lam_hi = min(lam_lo, lam.min()), max(lam_hi, lam.max())
        style = dict(s=14, marker=markers[li % len(markers)], facecolors="none",
                     edgecolors=OTHER, linewidths=0.8)
        left.scatter(lam, measured, label=f"layer {li}", **style)
        right.scatter(predicted, measured, label=f"layer {li}", **style)

    # The timescale box is propagated per channel and not every channel reaches the floor,
    # so the closed form is a band over the boxes rather than one curve. Its upper edge is
    # the widest box, which is where the channels that saturate the declared range sit.
    grid = torch.logspace(np.log10(lam_lo) - 0.1, np.log10(lam_hi) + 0.1, 200)
    d_lo = torch.cat([b.lo.reshape(-1) for b in boxes]).unsqueeze(-1)
    d_hi = torch.cat([b.hi.reshape(-1) for b in boxes]).unsqueeze(-1)
    curves = abstraction_gap(-grid.unsqueeze(0), Interval(d_lo, d_hi)).detach()
    upper, lower = curves.max(dim=0).values.numpy(), curves.min(dim=0).values.numpy()
    left.fill_between(grid.numpy(), lower, upper, color=OURS, alpha=0.15, linewidth=0,
                      label=r"closed form, all $\delta$ boxes")
    left.plot(grid.numpy(), upper, color=OURS, linewidth=2, label=r"closed form, widest box")
    left.set_xscale("log")
    left.set_xlabel(r"pole $\lambda = |A|$")
    left.set_ylabel("geometric bound / invariant")
    left.legend(loc="lower left")

    lims = [min(left.get_ylim()[0], 0.0), left.get_ylim()[1]]
    right.plot(lims, lims, color=INK, linewidth=0.8, label="$y = x$")
    right.set_xlim(lims)
    right.set_ylim(lims)
    right.set_xlabel("predicted")
    right.set_ylabel("measured")
    right.text(0.04, 0.96, f"max relative error {worst:.1e}", transform=right.transAxes,
               va="top", fontsize=9)
    right.legend(loc="lower right")
    print(f"  gap prediction: max relative error {worst:.2e}, propagated timescale boxes "
          f"within [{float(d_lo.min()):.4g}, {float(d_hi.max()):.4g}], "
          f"{int((d_lo <= d_lo.min() * 1.001).sum())} of {len(d_lo)} channels reach the floor")
    save(fig, "gap_prediction")


def fig_sweep_seeds(sweep_path: Path) -> None:
    """Across initialisation seeds the invariant moves with the weights and the ratio does not.

    Both series are shown as a percentage of their own median so they share a scale. The
    geometric bound is left out because it would sit exactly on top of the invariant: the
    ratio is what the closed form fixes, so the two bounds move together.
    """
    if not sweep_path.exists():
        print(f"  skipping seed figure: {sweep_path} not found")
        return
    rows = json.loads(sweep_path.read_text())["seeds"]
    seeds = [r["seed"] for r in rows]
    invariant = np.array([r["invariant"] for r in rows])
    ratio = np.array([r["ratio"] for r in rows])

    def spread(v: np.ndarray) -> np.ndarray:
        return 100.0 * (v / np.median(v) - 1.0)

    fig, ax = plt.subplots(figsize=WIDE)
    ax.axhline(0.0, color=INK, linewidth=0.8)
    ax.plot(seeds, spread(invariant), marker="o", linewidth=1.8, color=OTHER,
            label="invariant radius")
    ax.plot(seeds, spread(ratio), marker="s", linewidth=2.2, color=OURS,
            label="geometric bound / invariant")
    ax.set_xticks(seeds)
    ax.set_xlabel("initialisation seed")
    ax.set_ylabel("deviation from median (%)")
    ax.legend()
    save(fig, "sweep_seeds")


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
    fig, ax = plt.subplots(figsize=WIDE)
    quantities = list(dict.fromkeys(df["quantity"]))
    width = 0.38
    x = np.arange(len(quantities))
    for li, layer in enumerate(sorted(df["layer"].unique())):
        sub = df[df["layer"] == layer].set_index("quantity").reindex(quantities)
        ax.bar(x + li * width, sub["looseness"], width, label=f"layer {layer}")
    ax.set_yscale("log")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(quantities)
    ax.axhline(1.0, color=INK, linewidth=1.0)
    ax.set_ylabel("certified radius / realised radius")
    ax.legend()
    save(fig, "certified_vs_realised")


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

    fig, ax = plt.subplots(figsize=WIDE)
    ax.axhspan(lo, hi, color=OURS, alpha=0.12, label=f"certified range [{lo:.2f}, {hi:.2f}]")
    ax.axhline(lo, color=OURS, linewidth=1.4)
    ax.axhline(hi, color=OURS, linewidth=1.4)
    for i, (a, b) in enumerate(ranges):
        # markers at both ends, so a range too narrow to draw as a segment still shows
        ax.plot([i, i], [a, b], linewidth=6, color=OTHER, solid_capstyle="round",
                marker="o", markersize=4, label="realised range" if i == 0 else None)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f"$10^{{{int(np.log10(s))}}}$" for s in scales])
    ax.set_xlabel("input scale (standard deviations of white noise)")
    ax.set_ylabel("model output")
    ax.legend(loc="upper right")
    save(fig, "certified_output_range")


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
    fig.colorbar(im, ax=ax, label="p-value")
    save(fig, "dm_test_heatmap")


def fig_regime_ic(df: pd.DataFrame) -> None:
    if "volatility" not in df.columns:
        return
    d = df.copy()
    d["Regime"] = pd.qcut(d["volatility"], q=4, labels=["Quiet", "Normal", "Volatile", "Extreme"])
    models = [c for c in d.columns if c not in ("target", "timestamp", "volatility", "Regime")]

    regimes = ["Quiet", "Normal", "Volatile", "Extreme"]
    fig, ax = plt.subplots(figsize=WIDE)
    width = 0.8 / len(models)
    x = np.arange(len(regimes))
    for mi, m in enumerate(models):
        vals = [
            spearmanr(d[d["Regime"] == r]["target"], d[d["Regime"] == r][m]).statistic
            for r in regimes
        ]
        ax.bar(x + mi * width, vals, width, label=m)
    ax.set_xticks(x + 0.4 - width / 2, regimes)
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_ylabel("Spearman rank IC")
    ax.legend()
    save(fig, "regime_conditioned_ic")


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

    fig, ax = plt.subplots(figsize=WIDE)
    for name, lat in results.items():
        ax.plot(lengths, lat, marker="o", linewidth=2, label=name)
    ref = np.array(lengths, dtype=float)
    ax.plot(lengths, results["Causal Transformer"][0] * (ref / ref[0]) ** 2,
            linestyle=":", color=FAINT, label=r"$O(L^2)$ reference")
    ax.plot(lengths, results["Mamba (unrolled Python scan)"][0] * (ref / ref[0]),
            linestyle="--", color=FAINT, label=r"$O(L)$ reference")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context length $L$")
    ax.set_ylabel("forward-pass latency (ms)")
    ax.legend()
    save(fig, "latency_complexity")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--certificate", default="models/bounds/mamba_certificate.json")
    parser.add_argument("--sweep", default=str(SWEEP))
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    config = load_config()
    horizon = int(config["data"].get("prediction_horizon", 100))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("certification figures ...")
    fig_bound_vs_floor(config)
    fig_bound_vs_horizon(config)
    fig_gap_prediction(config)
    fig_sweep_seeds(Path(args.sweep))
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
