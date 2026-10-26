"""Incremental certificates for both trained arms, against the additive-timescale baseline.

  log-timescale  |a - a~| <= (e^rho - 1)(1 - a) with rho = L_log_delta delta: one constant
                 for every sequence length, and no floor on delta. Reported with the cell
                 theorem's time-warp term 2 M (e^rho - 1) and with the common-start one
                 2 M tanh(rho/2)
  additive       |a - a~| <= lam exp(-lam dt_min) |dt - dt~|: linear in L without a floor,
                 and proportional to 1/(1 - exp(-lam dt_min)) with the certified one

Both are data-free, taken with respect to a sup-norm perturbation of each block's normalised
input, and use the same certified boxes, so the comparison isolates the timescale rule. The
script also records what the bounds are set against: how far the data-free bound sits above
sampled perturbations, the timescale floor a gradient attack reaches, the realised distance
|h_{t-1} - c_t| that the local corollary uses in place of 2 M, LayerNorm's Lipschitz constants,
and the local norm-propagation domain run with each recurrence rule on held-out windows.

Writes docs/certification/incremental.{json,md} and docs/figures/incremental_vs_length.{pdf,png}.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.models.mamba import LOBMamba, SelectiveSSMBlock
from src.utils.config import load_config
from src.utils.repro import provenance, set_seed
from src.verification.certify import certify_block
from src.verification.incremental import (
    RULES,
    VARIANTS,
    BlockIncrement,
    contribution,
    incremental_certificate,
    layernorm_lipschitz,
    layernorm_realised,
    local_block_sensitivity,
    local_sensitivity_rule,
    realised_gap,
    weight_free_factor,
)
from src.verification.lipschitz import local_sensitivity

OUT_DIR = Path("docs/certification")
FIG_DIR = Path("docs/figures")
LENGTHS = [10, 100, 1000, 5000]
DELTAS = [1e-6, 1e-4, 1e-2]
LOCAL_EPS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
BLOCK_EPS = [1e-6, 1e-4, 1e-2]
GAP_WINDOWS, GAP_LENGTH = 8, 1000
ARMS = {"bounded": "best_mamba.pt", "softplus": "best_mamba_softplus.pt"}
RULE_NAMES = {"additive": "additive", "log_delta": "log-timescale, running maximum",
              "log_delta_recursive": "log-timescale, recursive",
              "log_delta_endpoint": "log-timescale, endpoint"}

# the colours and type of scripts/05_figures.py, so the figure sits with the paper's others
OURS = "#1f77b4"
OTHER = "#ff7f0e"
INK = "#212529"
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
HALF = (6.5, 3.1)


def load_model(checkpoint: Path, config: dict, dt_parametrisation: str) -> LOBMamba:
    cfg = copy.deepcopy(config)
    cfg["model"]["mamba"]["dt_parametrisation"] = dt_parametrisation
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = LOBMamba(int(ckpt["architecture"].get("input_dim", 43)), cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        return model.eval()
    print(f"  {checkpoint.name} not found; using a fresh model")
    return LOBMamba(43, cfg).eval()


def stats(t: torch.Tensor) -> dict:
    t = t.flatten()
    return {"min": float(t.min()), "median": float(t.median()), "max": float(t.max())}


def int_stats(t: torch.Tensor) -> dict:
    """stats of a count, as integers where finite."""
    return {k: int(v) if math.isfinite(v) else v for k, v in stats(t).items()}


def crossing_length(inc: BlockIncrement, variant: str = "cell") -> float:
    """Length at which the additive bound without a floor passes ours at the block output.

    Per output coordinate k that bound is a_k + b_k L with b_k >= 0, so its maximum over k first
    reaches ours, the maximum over k of the log-timescale slope, at min_k (ours - a_k) / b_k.
    """
    ours = inc.output_slope(variant).max()
    a = inc.additive_output_slope(0, floor=False)
    b = inc.additive_output_slope(1, floor=False) - a
    return float(((ours - a) / b).min())


def additive_slopes_at_floor(inc: BlockIncrement, floor: torch.Tensor) -> BlockIncrement:
    """The same certificate with the floor replaced, for the lower bound at the attacked floor."""
    return BlockIncrement(**{**inc.__dict__, "floor": floor.to(inc.floor.dtype)})


def shares(inc: BlockIncrement, variant: str) -> dict:
    """Where the block's output slope comes from, at its largest output coordinate."""
    parts = inc.output_slope_parts(variant)
    total = inc.output_slope(variant)
    k = int(total.argmax())
    return {"coordinate": k, "output_slope": float(total[k]),
            **{name: float(p[k] / total[k]) for name, p in parts.items()}}


def attack_floor(block: SelectiveSSMBlock, seed: int, steps: int = 150) -> torch.Tensor:
    """Smallest delta per channel that gradient descent on the block's raw input reaches.

    A value reached is attained, so inf delta is at most this, and so is any floor a certificate
    could use. The floored additive constant decreases in the floor, so evaluated here it is a
    lower bound on what any valid floor would give. Evaluated in float32, unrestricted input.
    """
    blk = copy.deepcopy(block).eval()
    for p in blk.parameters():
        p.requires_grad_(False)
    best = torch.full((blk.d_inner,), math.inf)
    for lr, scale in ((0.5, 10.0), (0.1, 1.0), (2.0, 100.0)):
        torch.manual_seed(seed)
        x = (torch.randn(16, 32, blk.d_model) * scale).requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            u, _ = blk.ssm_branch_input(x)
            delta, _, _ = blk.ssm_parameters(u)
            best = torch.minimum(best, delta.detach().amin(dim=(0, 1)))
            # per restart and channel, push the smallest timescale over time down
            delta.log().amin(dim=1).sum().backward()
            opt.step()
    return best


@torch.no_grad()
def sampled_looseness(block: SelectiveSSMBlock, inc: BlockIncrement, seed: int) -> dict:
    """Largest realised/certified ratio per link over sampled LayerNorm-output pairs.

    Random inputs and random perturbations of the raw block input, normalised, in float64.
    Since realised <= true <= certified, the certificate is at most 1/ratio times the true
    sensitivity; sampling does not search for the worst case, so it may be much less.
    """
    bare = copy.deepcopy(block).double().eval()
    bare.norm = nn.Identity()
    ln = copy.deepcopy(block.norm).double()
    lam = torch.exp(bare.A_log)

    def internals(xn):
        out, tr = bare(xn, trace=True)
        return {
            "u": tr.u,
            "log_delta": torch.log(tr.delta),
            "B": tr.B,
            "C": tr.C,
            "c": tr.B.unsqueeze(2) * tr.u.unsqueeze(-1) / lam,
            "h": tr.h,
            "y": tr.y,
            "out": out - xn,
        }

    lip = {"u": inc.L_u, "log_delta": inc.L_log_delta, "B": inc.L_B, "C": inc.L_C}
    worst: dict[str, float] = {}
    torch.manual_seed(seed)
    for x_scale, z_scale in itertools.product((0.3, 3.0, 30.0), (1e-5, 1e-3, 1e-1)):
        x = torch.randn(4, 48, block.d_model, dtype=torch.float64) * x_scale
        x_pert = x + (torch.rand_like(x) * 2 - 1) * z_scale * x_scale
        xn, xn_pert = ln(x), ln(x_pert)
        delta = (xn - xn_pert).abs().amax(dim=(1, 2))
        a, b = internals(xn), internals(xn_pert)
        for i, d in enumerate(delta.tolist()):
            bounds = {k: (k, v * d) for k, v in lip.items()}
            bounds.update({
                "c": ("c", inc.forcing_slope() * d),
                "h": ("h", inc.state_bound(d)),
                "h_common_start": ("h", inc.state_bound(d, variant="common_start")),
                "y": ("y", inc.y_bound(d)),
                "out": ("out", inc.output_bound(d)),
                "out_common_start": ("out", inc.output_bound(d, variant="common_start")),
            })
            for key, (field, bound) in bounds.items():
                r = float(((a[field][i] - b[field][i]).abs().amax(0) / bound).max())
                worst[key] = max(worst.get(key, 0.0), r)
    return worst


def certify_arm(model: LOBMamba, seq_len: int, seed: int) -> tuple[list[dict], list[BlockIncrement]]:
    rows, incs = [], []
    for li, block in enumerate(model.layers):
        cert = certify_block(block, seq_len=seq_len)
        inc = incremental_certificate(block, cert=cert)
        incs.append(inc)

        ours_output = inc.output_slope()
        warp_ratio = inc.warp_ratio()
        x_cert = inc.lam * inc.floor.unsqueeze(1)
        n_star = inc.crossover_horizon()

        attacked = attack_floor(block, seed)
        # the certified floor is a lower bound on delta, so the attack must not get below it
        below = int((attacked.double() < inc.floor * (1 - 1e-6)).sum())
        at_attack = additive_slopes_at_floor(inc, attacked)
        warp_ratio_attack = inc.warp_ratio(attacked.double())
        x_attack = inc.lam * attacked.double().unsqueeze(1)
        below_one = warp_ratio_attack < 1

        rows.append(
            {
                "layer": li,
                "lam": [float(inc.lam.min()), float(inc.lam.max())],
                "M_max": float(inc.M.max()),
                "two_M": {"max": 2.0 * float(inc.M.max()), "median": 2.0 * float(inc.M.median())},
                "certified_floor": stats(inc.floor),
                "certified_ceiling": stats(inc.ceiling),
                "attacked_floor": stats(attacked.double()),
                "attacked_below_certified_floor": below,
                "lipschitz_max": {
                    k: float(getattr(inc, f"L_{k}").max())
                    for k in ("u", "gate", "B", "C", "z", "log_delta", "delta")
                },
                "ours": {
                    "forcing_slope_max": float(inc.forcing_slope().max()),
                    "state_slope_max": float(inc.state_slope().max()),
                    "y_slope_max": float(inc.y_slope().max()),
                    "output_slope_max": float(ours_output.max()),
                    "output_slope_median": float(ours_output.median()),
                    "output_slope_shares": shares(inc, "cell"),
                },
                "ours_common_start": {
                    "state_slope_max": float(inc.state_slope("common_start").max()),
                    "y_slope_max": float(inc.y_slope("common_start").max()),
                    "output_slope_max": float(inc.output_slope("common_start").max()),
                    "output_slope_median": float(inc.output_slope("common_start").median()),
                    "output_slope_shares": shares(inc, "common_start"),
                },
                "additive_no_floor": {
                    str(L): {
                        "state_slope_max": float(inc.additive_state_slope(L, floor=False).max()),
                        "output_slope_max": float(inc.additive_output_slope(L, floor=False).max()),
                    }
                    for L in LENGTHS
                },
                "additive_floored": {
                    **{
                        str(L): {
                            "state_slope_max": float(inc.additive_state_slope(L, floor=True).max()),
                            "output_slope_max": float(inc.additive_output_slope(L, floor=True).max()),
                        }
                        for L in LENGTHS
                    },
                    "limit": {
                        "state_slope_max": float(inc.additive_state_slope(None, floor=True).max()),
                        "output_slope_max": float(inc.additive_output_slope(None, floor=True).max()),
                    },
                },
                "block_output_ratio_floored_over_ours": float(
                    inc.additive_output_slope(None, floor=True).max() / ours_output.max()),
                "warp_ratio_floored_over_ours": stats(warp_ratio),
                "lam_floor_product": {"min": float(x_cert.min()), "max": float(x_cert.max())},
                "first_factor": stats(inc.first_factor()),
                "weight_free_factor": weight_free_factor(block),
                "crossover_horizon_no_floor": {
                    **int_stats(n_star),
                    "additive_over_ours_one_step_min": float(inc.additive_over_ours_one_step().min()),
                },
                "crossing_length_no_floor": crossing_length(inc),
                "crossing_length_no_floor_common_start": crossing_length(inc, "common_start"),
                "additive_at_attacked_floor": {
                    "output_slope_max": float(at_attack.additive_output_slope(None, floor=True).max()),
                    "output_ratio_over_ours": float(
                        at_attack.additive_output_slope(None, floor=True).max() / ours_output.max()),
                    "warp_ratio": stats(warp_ratio_attack),
                    "fraction_of_cells_below_one": float(below_one.double().mean()),
                    # the smallest attacked floor among channels holding such a cell
                    "attacked_floor_min_where_below_one": (
                        float(attacked[below_one.any(1)].min()) if bool(below_one.any()) else None
                    ),
                    "lam_floor_product_max": float(x_attack.max()),
                },
                "output_bound_max": {
                    f"{d:g}": {
                        **{f"layernorm_outputs{'' if v == 'cell' else '_' + v}":
                           float(inc.output_bound(d, variant=v).max()) for v in VARIANTS},
                        **{f"arbitrary{'' if v == 'cell' else '_' + v}":
                           float(inc.output_bound(d, arbitrary=True, variant=v).max()) for v in VARIANTS},
                    }
                    for d in DELTAS
                },
                "euler_gain_max": inc.euler_gain_max(),
                "layernorm": layernorm_lipschitz(block.norm),
                "sampled_looseness": sampled_looseness(block, inc, seed),
            }
        )
    return rows, incs


# --- held-out data ----------------------------------------------------------------------------


def spread_windows(dm, n: int, seq_len: int) -> tuple[torch.Tensor, list[int]]:
    """n test windows of length seq_len whose end rows are spread evenly over the test split,
    the rule scripts/13_incremental_attack.py uses for its own windows."""
    lo, hi = dm.idx_test[0] + seq_len - 1, dm.idx_test[1] - 1
    ends = [hi] if n == 1 else [int(round(lo + i * (hi - lo) / (n - 1))) for i in range(n)]
    old = dm.seq_len
    dm.seq_len = seq_len
    try:
        ds = dm.dataset("test")
    finally:
        dm.seq_len = old
    return torch.stack([ds[e - seq_len + 1 - ds.start][0] for e in ends]), ends


@torch.no_grad()
def realised(models: dict[str, LOBMamba], incs: dict[str, list[BlockIncrement]], dm,
             windows: int, seq_len: int) -> dict:
    """What the data-free constants are set against, on held-out windows, in float64.

    Per block: the largest |h_{t-1} - c_t| per cell along the realised run, which the local
    corollary puts where the cell theorem has 2 M, and LayerNorm's Jacobian at the realised
    block inputs.
    """
    x, ends = spread_windows(dm, windows, seq_len)
    out = {"windows": windows, "seq_len": seq_len, "window_end_rows": ends,
           "test_rows": [int(dm.idx_test[0]), int(dm.idx_test[1])], "arms": {}}
    for arm, model in models.items():
        m64 = copy.deepcopy(model).double().eval()
        h = m64.proj(x.double())
        layers = []
        for li, block in enumerate(m64.layers):
            gap = realised_gap(block, block.norm(h))
            two_M = 2.0 * incs[arm][li].M
            layers.append({
                "layer": li,
                "gap": stats(gap),
                "two_M": stats(two_M),
                "gap_over_two_M_max": float((gap / two_M).max()),
                "layernorm": layernorm_realised(block.norm, h),
            })
            h = block(h)
        out["arms"][arm] = {"layers": layers}
    return out


@torch.no_grad()
def local_comparison(models: dict[str, LOBMamba], dm, windows: int) -> dict:
    """The norm-propagation domain with each recurrence rule, on the held-out windows of
    scripts/08_domains.py."""
    x, _ = next(iter(dm.loader("test", shuffle=False, batch_size=max(8, windows))))
    x = x[:windows].to(torch.float32)

    out = {"windows": int(x.shape[0]), "seq_len": int(x.shape[1]), "arms": {}}
    for arm, model in models.items():
        # the additive rule is the existing step; confirm it before quoting it as that
        old = local_sensitivity(model, x, 1e-6)["lipschitz"]
        new = local_sensitivity_rule(model, x, 1e-6, "additive")["lipschitz"]
        assert torch.equal(old, new), "the additive rule no longer reproduces lipschitz.py"

        model_rows = []
        for eps in LOCAL_EPS:
            row = {"eps": eps}
            for rule in RULES:
                w = local_sensitivity_rule(model, x, eps, rule)["width"]
                row[rule] = float(w.median()) if torch.isfinite(w).all() else math.inf
            model_rows.append(row)

        block_rows = []
        h = model.proj(x)
        for li, block in enumerate(model.layers):
            xn = block.norm(h)
            b64, xn64 = copy.deepcopy(block).double(), xn.double()
            base = contribution(b64, xn64)
            torch.manual_seed(li)
            sampled = 0.0
            for _ in range(20):
                xi = torch.where(torch.rand_like(xn64) > 0.5, 1e-4, -1e-4)
                moved = (contribution(b64, xn64 + xi) - base).abs().max()
                sampled = max(sampled, float(moved) / 1e-4)
            row = {"layer": li, "sampled": sampled}
            for eps in BLOCK_EPS:
                for rule in RULES:
                    lip = local_block_sensitivity(block, xn, eps, rule).lip
                    row[f"{rule}@{eps:g}"] = float(lip.max()) if torch.isfinite(lip).all() else math.inf
            block_rows.append(row)
            h = block(h)
        out["arms"][arm] = {"model": model_rows, "block": block_rows}
    return out


# --- the figure -------------------------------------------------------------------------------


def figure(incs: dict[str, list[BlockIncrement]], path: Path) -> None:
    """Certified block sensitivity against length, layer 0 of each arm."""
    lengths = np.unique(np.round(np.logspace(0, math.log10(5000), 160)).astype(int))
    fig, axes = plt.subplots(1, 2, figsize=HALF, sharey=True)
    for ax, (arm, layers) in zip(axes, incs.items()):
        inc = layers[0]
        ours = float(inc.output_slope().max())
        common = float(inc.output_slope("common_start").max())
        floored = [float(inc.additive_output_slope(int(L), floor=True).max()) for L in lengths]
        free = [float(inc.additive_output_slope(int(L), floor=False).max()) for L in lengths]
        ax.loglog(lengths, free, color=OTHER, linewidth=1.6, linestyle="--",
                  label="additive, no floor")
        ax.loglog(lengths, floored, color=OTHER, linewidth=2.0, label="additive, certified floor")
        ax.loglog(lengths, [ours] * len(lengths), color=OURS, linewidth=2.2,
                  label="log-timescale")
        ax.loglog(lengths, [common] * len(lengths), color=OURS, linewidth=1.6, linestyle="--",
                  label="log-timescale, common start")
        ax.set_xlabel("sequence length $L$")
        ax.text(0.04, 0.96, f"{arm} arm, layer 0", transform=ax.transAxes, va="top",
                fontsize=9, color=INK)
        if arm == "softplus":
            # the certified floor is about 1e-13, so the two additive curves coincide here
            k = int(np.searchsorted(lengths, 200))
            ax.annotate("additive, with or\nwithout the floor", xy=(lengths[k], floored[k]),
                        xytext=(lengths[k] * 4, floored[k] / 40), fontsize=8, color=INK,
                        ha="center", va="top",
                        arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 0.6})
    axes[0].set_ylabel(r"certified $\|F(x) - F(\tilde x)\|_\infty / \delta$")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False, fontsize=8)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


# --- the markdown twin ------------------------------------------------------------------------


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isinf(v):
            return "inf"
        if math.isnan(v):
            return "nan"
        return f"{v:.4g}" if (abs(v) >= 1e4 or (v and abs(v) < 1e-3)) else f"{v:.4f}"
    return str(v)


def table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def _layers(results: dict):
    for arm, a in results["arms"].items():
        for r in a["layers"]:
            yield arm, r


def below_one_note(results: dict) -> str:
    """Where the additive term could win at an exact floor, stated from the recorded floors."""
    parts = [
        f"{arm} layer {r['layer']}, attacked floor at least "
        f"{r['additive_at_attacked_floor']['attacked_floor_min_where_below_one']:.3g}"
        for arm, r in _layers(results)
        if r["additive_at_attacked_floor"]["attacked_floor_min_where_below_one"] is not None
    ]
    if not parts:
        return "There are none."
    return ("They sit in channels the attack could not push down to the certified floor: "
            + "; ".join(parts) + ".")


def attack_check(results: dict) -> str:
    """The attack doubles as a falsification test of the certified timescale box."""
    below = [(arm, r["layer"], r["attacked_below_certified_floor"]) for arm, r in _layers(results)]
    if not any(n for _, _, n in below):
        return "No channel in any block was pushed below its certified floor."
    return "Channels pushed below the certified floor, which falsifies the box: " + ", ".join(
        f"{arm} layer {li}: {n}" for arm, li, n in below if n) + "."


def second_order_gap(results: dict, delta: float) -> float:
    """Largest relative excess of the arbitrary reading over the LayerNorm one at delta."""
    return max(
        r["output_bound_max"][f"{delta:g}"]["arbitrary"]
        / r["output_bound_max"][f"{delta:g}"]["layernorm_outputs"] - 1.0
        for _, r in _layers(results)
    )


def markdown(results: dict) -> str:
    arms = results["arms"]
    rows_all = list(_layers(results))
    md = [
        "# Incremental certificates",
        "",
        "Every certified number here is data-free. It bounds how far a block's contribution",
        "before the residual, `out_proj(y * SiLU(gate))`, can move when its normalised input",
        "moves by at most `delta` in sup-norm at every position, with both normalised inputs",
        "LayerNorm outputs. Slopes are the small-`delta` limits of the bounds divided by `delta`.",
        "The log-timescale certificate is reported with two time-warp terms: the cell theorem's",
        "`2 M (e^rho - 1)`, and `2 M tanh(rho/2)`, a sharper constant for two runs that start from",
        "the same state, which they do here (`h_{-1} = 0`). To first order the second is half the",
        "first. The paper states only the first.",
        "",
        "## Block slopes",
        "",
        table(
            [
                [arm, r["layer"], r["ours"]["state_slope_max"], r["ours"]["output_slope_max"],
                 r["ours_common_start"]["output_slope_max"],
                 r["additive_floored"]["limit"]["output_slope_max"],
                 r["block_output_ratio_floored_over_ours"], r["crossing_length_no_floor"],
                 r["crossing_length_no_floor_common_start"]]
                for arm, r in rows_all
            ],
            ["arm", "layer", "log-timescale, state", "log-timescale, output",
             "log-timescale, common start, output", "additive with floor, output",
             "additive with floor / log-timescale", "additive without floor passes at L",
             "same, common start"],
        ),
        "",
        "The log-timescale columns hold at every sequence length. The additive bound without a",
        "floor is affine in `L`, and the last two columns are the lengths at which its output",
        "slope passes the log-timescale one, with the cell theorem and with a common start; below",
        "them the additive bound is the tighter, and a value below 1 means it is looser at every",
        "length.",
        "",
        "Where the log-timescale output slope comes from, at its largest output coordinate, as",
        "fractions of it:",
        "",
        table(
            [
                [arm, r["layer"], name, s["warp"], s["forcing"], s["C"], s["D"], s["gate"]]
                for arm, r in rows_all
                for name, s in (("cell theorem", r["ours"]["output_slope_shares"]),
                                ("common start", r["ours_common_start"]["output_slope_shares"]))
            ],
            ["arm", "layer", "time-warp term", "time-warp", "forcing", "C perturbation",
             "skip path D", "gate"],
        ),
        "",
        "Time-warp is the term `2 M (e^rho - 1)` (or `2 M tanh(rho/2)`) read out through `C`;",
        "forcing is `sup|c - c~|` read out the same way; C perturbation is `M L_C`; the gate term",
        "is `sup|y|` times the gate's own sensitivity.",
        "",
        "## Against sequence length",
        "",
    ]
    for arm, a in arms.items():
        rows = []
        for r in a["layers"]:
            for L in LENGTHS:
                rows.append([r["layer"], L, r["ours"]["output_slope_max"],
                             r["additive_no_floor"][str(L)]["output_slope_max"],
                             r["additive_floored"][str(L)]["output_slope_max"]])
            rows.append([r["layer"], "limit", r["ours"]["output_slope_max"], math.inf,
                         r["additive_floored"]["limit"]["output_slope_max"]])
        md += [f"{arm} arm, output slope:", "",
               table(rows, ["layer", "L", "log-timescale", "additive, no floor",
                            "additive, certified floor"]), ""]
    md += [
        "In the additive bound with the certified floor, the timescale term is its one-step",
        "cost times `sum_{k<L} exp(-k lam dt_min)`, which saturates at `1/(1 - exp(-lam dt_min))`.",
        "On the softplus arm the certified floor is below `5e-13`, so it does not saturate",
        "within these lengths. The additive bound keeps the factor `exp(-lam dt_min)` in its",
        "one-step cost, which Cao et al.'s Lemma B.3 drops; their Remark 3.4 constant is larger",
        "by `e^(lam dt_min)`, which at the certified floors is at most "
        + " and ".join(f"{fmt(math.exp(max(r['lam_floor_product']['max'] for r in a['layers'])))} "
                       f"on the {arm} arm" for arm, a in arms.items())
        + ".",
        "",
        "## Per cell",
        "",
        "The per-cell comparison is between the two time-warp terms, to first order in `delta`,",
        "for each (channel, state) cell:",
        "",
        "```",
        "ratio = [2 M lam L_delta / (exp(lam dt_min) - 1)] / [2 M L_log_delta]",
        "      = [L_delta / (dt_min L_log_delta)] * x / (e^x - 1),     x = lam dt_min",
        "```",
        "",
        "The numerator is the additive time-warp term with the certified floor `dt_min`, at its",
        "limit in the length, with the factor `exp(-x)` kept; the denominator is the cell",
        "theorem's term. The forcing term is the same in both and is left out, and `M` and the",
        "Lipschitz constant of the pre-activation cancel, so the ratio depends on the",
        "parametrisation, on the certified box of the channel's pre-activation and on `x` only.",
        "`L_delta` and `L_log_delta` are `sup|dt'|` and `sup|(log dt)'|` over that box times the",
        "same `L_z`. Cao et al.'s Remark 3.4 form drops `exp(-x)` and is larger by `e^x`.",
        "Against the common-start term the ratio doubles.",
        "",
        table(
            [
                [arm, r["layer"], r["certified_floor"]["min"],
                 r["lam_floor_product"]["min"], r["lam_floor_product"]["max"],
                 r["warp_ratio_floored_over_ours"]["min"], r["warp_ratio_floored_over_ours"]["median"],
                 r["warp_ratio_floored_over_ours"]["max"], r["weight_free_factor"]]
                for arm, r in rows_all
            ],
            ["arm", "layer", "certified floor", "x min", "x max", "ratio min", "ratio median",
             "ratio max", "weight-free factor"],
        ),
        "",
        "The weight-free factor is the first factor with the slopes taken over all `z`:",
        "`sup|dt'| / (dt_min sup|(log dt)'|)`, which for the bounded parametrisation depends on",
        "`dt_max / dt_min` only. Over the certified boxes the first factor lies between "
        + "; ".join(f"{fmt(r['first_factor']['min'])} and {fmt(r['first_factor']['max'])} "
                    f"({arm} layer {r['layer']})" for arm, r in rows_all)
        + ", and `x / (e^x - 1) <= 1`, so the ratio is at most the first factor of its channel.",
        "The softplus parametrisation has no positive `dt_min`; there the certified floor plays",
        "its part, and the first factor is about `1 / dt_min`.",
        "",
        "Without a floor the additive time-warp term after `n` steps is `2 M lam L_delta n`,",
        "while the cell theorem's, in its finite-horizon form with the worst case over",
        "admissible timescales, is `2 M L_log_delta (1 - exp(-lam dt_hi n))`, `dt_hi` the",
        "certified ceiling. The crossover horizon `n*` is the smallest `n >= 1` at which the",
        "first is at least the second, to first order in `delta`; for fewer steps the additive",
        "term is the smaller.",
        "",
        table(
            [
                [arm, r["layer"], r["crossover_horizon_no_floor"]["min"],
                 r["crossover_horizon_no_floor"]["median"], r["crossover_horizon_no_floor"]["max"],
                 r["crossover_horizon_no_floor"]["additive_over_ours_one_step_min"],
                 r["certified_ceiling"]["max"]]
                for arm, r in rows_all
            ],
            ["arm", "layer", "n* min", "n* median", "n* max",
             "additive / ours after one step, min", "certified ceiling, max"],
        ),
        "",
        "At the attacked floor:",
        "",
        table(
            [
                [arm, r["layer"], r["attacked_floor"]["min"],
                 r["additive_at_attacked_floor"]["lam_floor_product_max"],
                 r["additive_at_attacked_floor"]["warp_ratio"]["min"],
                 r["additive_at_attacked_floor"]["warp_ratio"]["median"],
                 r["additive_at_attacked_floor"]["warp_ratio"]["max"],
                 r["additive_at_attacked_floor"]["fraction_of_cells_below_one"],
                 r["additive_at_attacked_floor"]["output_ratio_over_ours"]]
                for arm, r in rows_all
            ],
            ["arm", "layer", "attacked floor", "x max", "ratio min", "ratio median", "ratio max",
             "fraction of cells with ratio below 1", "additive / log-timescale at the block output"],
        ),
        "",
        "The ratio here is the same per-cell ratio with `dt_min` set to the attacked floor of",
        "the channel. The attacked floor is the smallest `delta` per channel that gradient descent",
        "on the block's raw input reaches, so any valid floor is at most that; the additive term",
        "decreases in the floor, so its ratio there is a lower bound on what the additive rule",
        "could give with an exact floor. A cell with ratio below 1 there is one where an exact",
        "per-channel floor could make the additive term the smaller one. "
        + below_one_note(results),
        "The last column is the block-level comparison: the floored additive output slope, with",
        "the forcing included, at the attacked floor, over the log-timescale output slope. The",
        "floor columns show the minimum over channels. On the bounded arm the certified floor is",
        "`dt_min`, which the clamp in `delta_from_pre` makes a hard lower bound, and the attack",
        "comes within "
        + f"{100 * max(r['attacked_floor']['min'] / r['certified_floor']['min'] - 1 for r in arms['bounded']['layers']):.1f}"
        + " percent of it in the lowest channel. "
        + attack_check(results),
        "",
        "## Second order",
        "",
        table(
            [
                [arm, r["layer"], d, r["output_bound_max"][f"{d:g}"]["layernorm_outputs"],
                 r["output_bound_max"][f"{d:g}"]["arbitrary"],
                 r["output_bound_max"][f"{d:g}"]["layernorm_outputs_common_start"],
                 r["output_bound_max"][f"{d:g}"]["arbitrary_common_start"]]
                for arm, r in rows_all for d in DELTAS
            ],
            ["arm", "layer", "delta", "both LayerNorm outputs", "perturbed input arbitrary",
             "common start, both LayerNorm outputs", "common start, perturbed input arbitrary"],
        ),
        "",
        "Maximum over output coordinates of the bound at `delta`, not its slope. When the",
        "perturbed normalised input need not be a LayerNorm output, the perturbed trajectory can",
        "leave the certified boxes, so they are widened by `L delta` and derivative suprema are",
        "taken globally. With the cell theorem the second reading is larger by at most "
        + f"{100 * second_order_gap(results, DELTAS[0]):.2g} percent at `delta = {DELTAS[0]:g}`"
        + f" and {100 * second_order_gap(results, DELTAS[-1]):.2g} percent at "
        + f"`delta = {DELTAS[-1]:g}`.",
        "",
        "## How loose the data-free bound is",
        "",
        table(
            [
                [arm, r["layer"], *[r["sampled_looseness"][k] for k in
                                    ("u", "log_delta", "B", "C", "c", "h", "h_common_start",
                                     "y", "out", "out_common_start")]]
                for arm, r in rows_all
            ],
            ["arm", "layer", "u", "log delta", "B", "C", "c", "h", "h, common start", "y",
             "output", "output, common start"],
        ),
        "",
        "Each entry is the largest realised difference over its certified bound, sampled over",
        "random inputs and random perturbations of the raw block input. The ratio falls at",
        "every stage from `u` to the output, because each stage applies worst-case row sums of",
        "absolute weights or the certified boxes to the stage before it, and the losses",
        "multiply. Sampling does not search for the worst case, so the certificate is at most",
        "the reciprocal of each entry times the true sensitivity, and may be much less.",
        "",
        "## LayerNorm and the exponential-Euler gain",
        "",
        table(
            [
                [arm, r["layer"], r["layernorm"]["gamma_abs_max"], r["layernorm"]["eps"],
                 r["layernorm"]["l2_global_bound"], r["layernorm"]["l2_at_constant_input"],
                 r["layernorm"]["linf_at_constant_input"], r["euler_gain_max"]]
                for arm, r in rows_all
            ],
            ["arm", "block", "max abs gamma", "eps", "l2 bound, any input", "l2 at a constant input",
             "sup-norm at a constant input", "exponential-Euler gain"],
        ),
        "",
        "With `eps > 0` the block's LayerNorm is globally Lipschitz. Its Jacobian is",
        "`diag(gamma)(I - 1 1^T/d - z z^T/d)/s` with `s = sqrt(var + eps)`, whose l2 norm is at",
        "most `max|gamma|/s <= max|gamma|/sqrt(eps)`. The values at a constant input, where the",
        "variance is 0, are attained, so they bound the global constants from below, in l2 and in",
        "the sup-norm that the block certificate uses. A data-free statement in the raw block",
        "input therefore exists, with these factors; a local one needs a lower bound on the",
        "spread `s`. The last column is `g(lam dt_hi) = lam dt_hi / (1 - exp(-lam dt_hi))`,",
        "maximised over cells with the certified ceiling `dt_hi`, the factor by which",
        "exponential-Euler (`Bbar = dt B`) enlarges the equilibrium; this repository's block uses",
        "the exact zero-order hold.",
        "",
    ]
    real = results.get("realised")
    if real:
        md += [
            "## On held-out windows",
            "",
            f"{real['windows']} windows of length {real['seq_len']} from the test split, rows "
            f"{real['test_rows'][0]} to {real['test_rows'][1] - 1}, ending at rows "
            + ", ".join(str(e) for e in real["window_end_rows"])
            + ". Each block is run in float64 from `h_{-1} = 0`.",
            "",
            table(
                [
                    [arm, r["layer"], r["gap"]["max"], r["gap"]["median"], r["two_M"]["max"],
                     r["two_M"]["median"], r["gap_over_two_M_max"]]
                    for arm, a in real["arms"].items() for r in a["layers"]
                ],
                ["arm", "block", "realised distance, max over cells", "median over cells",
                 "2M, max over cells", "2M, median", "largest distance / 2M in one cell"],
            ),
            "",
            "The realised distance of a cell is `max_t |h_{t-1} - c_t|` over the windows and",
            "positions, with `c_t = B_t u_t / lam`. The local corollary uses it where the cell",
            "theorem has the worst case `2 M`.",
            "",
            table(
                [
                    [arm, r["layer"], r["layernorm"]["rows"], r["layernorm"]["spread_min"],
                     r["layernorm"]["spread_max"], r["layernorm"]["l2_max"],
                     r["layernorm"]["l2_bound_max"], r["layernorm"]["linf_max"]]
                    for arm, a in real["arms"].items() for r in a["layers"]
                ],
                ["arm", "block", "positions", "spread s, min", "spread s, max",
                 "l2 norm of the Jacobian, max", "max|gamma| / s, max", "sup-norm of the Jacobian, max"],
            ),
            "",
            "LayerNorm's Jacobian at the realised input of each block, over every position of",
            "the windows: about 1 in l2, against the global constants above.",
            "",
        ]
    local = results.get("local")
    if local:
        md += ["## The local domain with each recurrence rule", "",
               f"Median certified output width over {local['windows']} held-out windows of length "
               f"{local['seq_len']}, perturbing the network input by `eps`, with the norm-propagation "
               "domain of `src/verification/lipschitz.py` and its recurrence step replaced. The "
               "additive rule reproduces that domain exactly. The running maximum is the local "
               "corollary, `E_t = max(E_{t-1}, dc_t + (e^rho_t - 1)|h_{t-1} - c_t|)`. The "
               "recursive rule is the one-step inequality in its proof, "
               "`E_t = K_t + a*_t (E_{t-1} - K_t)` with `K_t` the second argument of that maximum "
               "and `a*_t` the perturbed transition at its worst value in "
               "`[exp(-lam dt e^rho), exp(-lam dt e^-rho)]`; it is never larger than the running "
               "maximum, and with `rho = 0` it is the exact recursion of the forcing term. The "
               "endpoint rule, the reported one, bounds the error identity itself, "
               "`|e_t| <= s E_{t-1} + (1 - s) dc_t + |Abar_t - s| |h_{t-1} - c_t|` with `s` the "
               "perturbed transition, which is convex in `s`, and takes its larger value at the two "
               "ends of the interval that holds `s`: the intersection of the interval above with the "
               "one the additive bound on `|dt - dt~|` gives. It is never larger than the recursive "
               "rule or the additive rule, and it is finite for every `rho`.", ""]
        rows = [[arm, r["eps"], *[r[rule] for rule in RULES]]
                for arm, a in local["arms"].items() for r in a["model"]]
        md += [table(rows, ["arm", "eps", *[RULE_NAMES[rule] for rule in RULES]]), ""]
        rows = [[arm, r["layer"], r["sampled"], *[r[f"{rule}@{e:g}"] for e in BLOCK_EPS for rule in RULES]]
                for arm, a in local["arms"].items() for r in a["block"]]
        short = {"additive": "additive", "log_delta": "maximum", "log_delta_recursive": "recursive",
                 "log_delta_endpoint": "endpoint"}
        header = ["arm", "layer", "sampled"] + [f"{short[rule]} @ {e:g}" for e in BLOCK_EPS for rule in RULES]
        md += ["Per block, the local Lipschitz bound on the block's contribution around the "
               "realised normalised input, for any perturbation of it of sup-norm `eps`, against "
               "the largest sampled ratio at `eps = 1e-4`:", "", table(rows, header), ""]
    md += [
        "## Scope",
        "",
        "The bounds hold over the reals, not in floating point. They are per block, with respect",
        "to the normalised input. A statement in the raw block input passes through the",
        "LayerNorm, which is globally Lipschitz only through `eps`, with the constants above.",
        "The same per-cell bound follows from Cao et al. (ICML 2026), Lemma 3.1 and Lemma B.2,",
        "with a piecewise-linear time change, and it strengthens their Theorem 3.3(ii). The",
        "additive baseline mirrors the structure of their Theorem 3.3(ii) and Remark 3.4, applied",
        "per cell over the certified boxes, with the factor their Lemma B.3 drops kept. The",
        "induction is Horbatko's Lemma 4.4 (arXiv:2604.18580) applied to the error.",
        "",
    ]
    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental certificates against the additive baseline.")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--windows", type=int, default=4, help="windows of the local comparison")
    parser.add_argument("--gap-windows", type=int, default=GAP_WINDOWS)
    parser.add_argument("--gap-length", type=int, default=GAP_LENGTH)
    parser.add_argument("--no-data", action="store_true", help="skip everything that needs the data")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--fig-dir", default=str(FIG_DIR))
    args = parser.parse_args()

    config = load_config()
    seed = int(config.get("seed", 1337))
    set_seed(seed)
    seq_len = int(config["data"]["seq_length"])
    ckpt_dir = Path(args.checkpoint_dir)

    models = {arm: load_model(ckpt_dir / name, config, arm) for arm, name in ARMS.items()}
    arms, incs = {}, {}
    for arm, model in models.items():
        print(f"certifying the {arm} arm ...", flush=True)
        rows, incs[arm] = certify_arm(model, seq_len, seed)
        arms[arm] = {"checkpoint": str(ckpt_dir / ARMS[arm]), "layers": rows}

    real = local = None
    data_file = Path(config["data"]["processed_file"])
    if args.no_data:
        print("  skipping the held-out measurements")
    elif not data_file.exists():
        print(f"  {data_file} not found; skipping the held-out measurements")
    else:
        from src.dataset import LOBDataModule

        dm = LOBDataModule(config)
        print("realised distances and LayerNorm on held-out windows ...", flush=True)
        real = realised(models, incs, dm, args.gap_windows, args.gap_length)
        print("local comparison on held-out windows ...", flush=True)
        local = local_comparison(models, dm, args.windows)

    fig_dir, out_dir = Path(args.fig_dir), Path(args.out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "incremental_vs_length.pdf"
    figure(incs, fig_path)

    results = {
        "provenance": provenance(seed),
        "perturbation": "sup-norm delta on the block's normalised input, both inputs LayerNorm outputs",
        "lengths": LENGTHS,
        "definitions": {
            "warp_ratio_floored_over_ours":
                "per (channel, state) cell, first order in delta: the additive time-warp term with "
                "the certified floor dt_min at its limit in the length, 2 M lam L_delta / "
                "(exp(lam dt_min) - 1), over the cell theorem's time-warp term 2 M L_log_delta; "
                "equal to [L_delta / (dt_min L_log_delta)] x / (e^x - 1) with x = lam dt_min",
            "crossover_horizon_no_floor":
                "per cell, first order in delta: the smallest n >= 1 at which 2 M lam L_delta n, the "
                "additive time-warp term without a floor after n steps, reaches 2 M L_log_delta "
                "(1 - exp(-lam dt_hi n)), the cell theorem's finite-horizon term, dt_hi the certified "
                "ceiling",
            "block_output_ratio_floored_over_ours":
                "the floored additive output slope at its limit in the length over the log-timescale "
                "output slope (cell theorem), maxima over output coordinates, forcing included",
            "output_slope_shares":
                "fractions of the output slope at its largest coordinate: time-warp, forcing, "
                "C perturbation, skip path D and gate",
        },
        "arms": arms,
        "realised": real,
        "local": local,
        "figure": str(fig_path),
    }
    md = markdown(results)
    (out_dir / "incremental.json").write_text(json.dumps(results, indent=2))
    (out_dir / "incremental.md").write_text(md)
    print("\n" + md)
    print(f"wrote {out_dir / 'incremental.json'}, {out_dir / 'incremental.md'} and {fig_path}")


if __name__ == "__main__":
    main()
