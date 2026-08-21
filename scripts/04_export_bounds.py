"""Export a trained LOBMamba as a verifiable certificate: equations, weights, certified
intervals, realised envelope. See docs/05-handoff.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.dataset import LOBDataModule, Normalisation
from src.models.mamba import LOBMamba
from src.utils.config import load_config
from src.utils.repro import provenance
from src.verification.certify import (
    certify_model,
    empirical_envelope,
    looseness_report,
)
from src.verification.intervals import Interval
from src.verification.export import export_parameters
from src.verification.reference import ReferenceModel

SCHEMA_VERSION = "1.0"

DYNAMICS_SPEC = {
    "description": (
        "Per block: a selective state space model with diagonal, time-invariant A and "
        "input-dependent delta, B, C. Channels d = 0..d_inner-1 evolve independently; "
        "n = 0..d_state-1 indexes the diagonal state."
    ),
    "pre_ssm": [
        "xn      = LayerNorm(x)",
        "[u0, g] = in_proj(xn)",
        "u       = SiLU(DepthwiseCausalConv1d(u0))   # position t reads u0[t-d_conv+1 .. t]",
    ],
    "selective_parameters": [
        "[dt_pre_raw, B, C] = x_proj(u)                       # split dt_rank, d_state, d_state",
        "z                  = dt_proj(dt_pre_raw)",
        "delta              = dt_min * (dt_max/dt_min)**sigmoid(z)   # 'bounded'",
        "delta              = softplus(z)                             # 'softplus'",
    ],
    "discretisation": [
        "A       = -exp(A_log)                       # strictly negative, shape (d_inner, d_state)",
        "A_bar   = exp(delta * A)",
        "B_bar   = (exp(delta * A) - 1) / A * B      # exact zero-order hold",
        "        = delta * phi(delta * A) * B,  phi(v) = expm1(v)/v, phi(0) = 1",
    ],
    "recurrence": [
        "h[t] = A_bar[t] * h[t-1] + B_bar[t] * u[t]      # h[-1] = 0",
        "y[t] = sum_n C[t, n] * h[t, :, n] + D * u[t]",
    ],
    "post_ssm": [
        "out = out_proj(y * SiLU(g)) + x            # residual is the block input, pre-norm",
    ],
    "stability": (
        "A < 0 and delta > 0 give A_bar in (0, 1) elementwise. Exact zero-order hold gives "
        "B_bar = (1 - A_bar) * B / |A| identically, so the update "
        "h[t] = A_bar * h[t-1] + (1 - A_bar) * c[t], with c = B * u / |A|, is a convex "
        "combination. If |c| <= M for every admissible input then |h| <= M for every t, by "
        "induction from h[-1] = 0. That holds at any sequence length and needs no lower "
        "bound on delta. The geometric form sup|B_bar * u| / (1 - sup A_bar) bounds the same "
        "quantity but discards the (1 - A_bar) factor, taking its maximum in the numerator "
        "and its minimum in the denominator, and is about 95 times looser on these weights."
    ),
}


def box_to_dict(box: Interval, reduce: bool = True) -> dict:
    """Serialise an interval. `reduce` collapses to scalars for the large SSM tensors."""
    if reduce:
        return {
            "lo": float(box.lo.min().item()),
            "hi": float(box.hi.max().item()),
            "abs_max": float(box.abs_max().max().item()),
            "shape": list(box.lo.shape),
        }
    return {
        "lo": box.lo.tolist(),
        "hi": box.hi.tolist(),
        "shape": list(box.lo.shape),
    }


def serialise_certificate(cert: dict) -> dict:
    out = {
        k: v for k, v in cert.items() if k not in ("layers",)
    }
    layers = []
    for layer in cert["layers"]:
        boxes = layer["boxes"]
        layers.append(
            {
                "contraction": layer["contraction"],
                "horizon_gain": layer["horizon_gain"],
                "summary": layer["summary"],
                # delta/B/C are small and are what a verifier reasons about, so they go out
                # in full; the per-(channel, state) tensors and the layer outputs are
                # reduced to their extrema.
                "boxes_full": {
                    "delta": box_to_dict(boxes["delta"], reduce=False),
                    "B": box_to_dict(boxes["B"], reduce=False),
                    "C": box_to_dict(boxes["C"], reduce=False),
                    "u": box_to_dict(boxes["u"], reduce=False),
                },
                "boxes_reduced": {
                    k: box_to_dict(boxes[k])
                    for k in (
                        "A_bar",
                        "B_bar",
                        "h_invariant",
                        "h_widened",
                        "y_invariant",
                        "out_invariant",
                        "h_geometric",
                        "h_horizon",
                        "y_geometric",
                        "y_horizon",
                        "out_geometric",
                        "out_horizon",
                    )
                },
            }
        )
    out["layers"] = layers
    return out


def numpy_roundtrip_check(
    model: LOBMamba, artefact: dict, x: torch.Tensor, tol: float = 1e-4
) -> dict:
    """Re-run the whole network in NumPy from the artefact alone and compare against PyTorch.

    This drives src.verification.reference, which imports nothing from the model and reads
    only the exported dict, so it exercises the stem, both blocks and the head rather than
    the scan in isolation. If it disagrees, the artefact does not describe the model.
    """
    model.eval()
    with torch.no_grad():
        y_torch, traces = model(x, trace=True)

    reference = ReferenceModel(artefact)
    y_numpy, rec = reference.forward(x.numpy(), trace=True)

    worst = {"output": float(np.abs(y_numpy - y_torch.numpy()).max())}
    for i, (tr, r) in enumerate(zip(traces, rec)):
        for name, a, b in (
            ("u", tr.u, r["u"]),
            ("delta", tr.delta, r["delta"]),
            ("B", tr.B, r["B"]),
            ("C", tr.C, r["C"]),
            ("A_bar", tr.A_bar, r["A_bar"]),
            ("B_bar", tr.B_bar, r["B_bar"]),
            ("h", tr.h, r["h"]),
            ("y", tr.y, r["y"]),
        ):
            worst[f"layer{i}.{name}"] = float(np.abs(a.numpy() - b).max())

    err = max(worst.values())
    return {
        "max_abs_error": err,
        "max_abs_error_output": worst["output"],
        "per_quantity": worst,
        "tolerance": tol,
        "passed": bool(err < tol),
    }


def soundness_check(cert: dict, envelope: dict) -> dict:
    """Check that every realised value sits inside its certified box."""
    violations = []
    for li, (cl, el) in enumerate(zip(cert["layers"], envelope["layers"])):
        boxes = cl["boxes"]
        for key, cert_key in [
            ("u", "u"),
            ("delta", "delta"),
            ("B", "B"),
            ("C", "C"),
            ("A_bar", "A_bar"),
            ("B_bar", "B_bar"),
            ("h", "h_horizon"),
            ("y", "y_horizon"),
        ]:
            if key not in el:
                continue
            box = boxes[cert_key]
            lo, hi = float(box.lo.min()), float(box.hi.max())
            if el[key]["lo"] < lo - 1e-4 or el[key]["hi"] > hi + 1e-4:
                violations.append(
                    {
                        "layer": li,
                        "quantity": key,
                        "certified": [lo, hi],
                        "realised": [el[key]["lo"], el[key]["hi"]],
                    }
                )
    return {"violations": violations, "sound": not violations}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a verifiable model description.")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_mamba.pt")
    parser.add_argument("--out", default="models/bounds/mamba_certificate.json")
    parser.add_argument("--envelope-batches", type=int, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config") or load_config()
    ver_cfg = config.get("verification", {})
    seq_len = int(ver_cfg.get("seq_len", config["data"]["seq_length"]))
    tight = bool(ver_cfg.get("tight_layernorm", True))
    n_env = args.envelope_batches or int(ver_cfg.get("envelope_batches", 20))

    arch = ckpt.get("architecture", {})
    model = LOBMamba(input_dim=int(arch.get("input_dim", 43)), config=config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded {ckpt_path} | {model.architecture()}")

    print("certifying (input-independent, no data required) ...")
    cert = certify_model(model, seq_len=seq_len, tight_layernorm=tight)

    print(f"measuring the realised envelope on {n_env} held-out batches ...")
    dm = LOBDataModule(config)
    test_loader = dm.loader("test", shuffle=False, batch_size=32)
    batches = []
    for i, (x, _) in enumerate(test_loader):
        if i >= n_env:
            break
        batches.append(x)
    envelope = empirical_envelope(model, batches)
    loose = looseness_report(cert, envelope)
    sound = soundness_check(cert, envelope)

    params = export_parameters(model)

    if not sound["sound"]:
        raise SystemExit(
            f"soundness FAILED: realised values outside the certified box: "
            f"{sound['violations']}"
        )

    norm = Normalisation.from_dict(
        ckpt.get("normalisation") or dm.normalisation.to_dict()
    )
    box_lo, box_hi = norm.input_box()

    artefact = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "export": provenance(config.get("seed")),
            "training": ckpt.get("provenance", {}),
            "checkpoint": str(ckpt_path),
            "training_summary": {
                k: v for k, v in ckpt.get("training", {}).items() if k != "history"
            },
        },
        "architecture": model.architecture(),
        "dynamics_spec": DYNAMICS_SPEC,
        "normalisation": norm.to_dict(),
        "input_box": {
            "feature_names": norm.feature_names,
            "lo": box_lo.tolist(),
            "hi": box_hi.tolist(),
            "enforced_by": (
                "winsorisation to train-split quantiles, applied at inference; the box is a "
                "deployment guarantee, not a historical observation"
            ),
        },
        "certificate": serialise_certificate(cert),
        "empirical_envelope": envelope,
        "looseness": loose,
        "checks": {"soundness": sound},
        "parameters": params,
    }

    print("re-running the whole network in NumPy from the artefact ...")
    roundtrip = numpy_roundtrip_check(model, artefact, batches[0][:4])
    artefact["checks"]["roundtrip"] = roundtrip
    if not roundtrip["passed"]:
        raise SystemExit(
            f"round-trip FAILED: worst {roundtrip['max_abs_error']:.3e} in "
            f"{max(roundtrip['per_quantity'], key=roundtrip['per_quantity'].get)}. "
            "The export does not describe the model; refusing to write it."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artefact, indent=2))

    npz_path = out_path.with_suffix(".npz")
    flat = {}
    for i, layer in enumerate(params["layers"]):
        for k, v in layer.items():
            if isinstance(v, list):
                flat[f"layer{i}.{k}"] = np.asarray(v, dtype=np.float32)
    for section in ("stem", "head"):
        for k, v in params[section].items():
            flat[k] = np.asarray(v, dtype=np.float32)
    np.savez_compressed(npz_path, **flat)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nwrote {out_path} ({size_mb:.1f} MB) and {npz_path.name}")
    print(f"  round-trip worst |numpy - torch| = {roundtrip['max_abs_error']:.3e}")
    print(f"  soundness: {'OK' if sound['sound'] else 'VIOLATED'}")
    print(f"  all layers contractive: {cert['all_layers_contractive']}")
    print(
        f"  certified output range: "
        f"[{cert['output_range']['lo']:.4f}, {cert['output_range']['hi']:.4f}]"
    )
    for i, layer in enumerate(cert["layers"]):
        s = layer["summary"]
        print(
            f"  layer {i}: invariant |h| <= {s['h_abs_max_invariant']:.2f}  "
            f"(induction holds: {s['invariant_holds']}, exact ZOH: {s['zoh_exact']})  "
            f"against widened {s['h_abs_max_widened']:.3e}"
        )


if __name__ == "__main__":
    main()
