"""Try to falsify the certificate by gradient ascent on the input.

Since attacked <= true reachable <= certified, the gap also measures looseness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.dataset import LOBDataModule, Normalisation
from src.models.mamba import LOBMamba
from src.utils.config import load_config
from src.utils.repro import provenance, set_seed
from src.verification.certify import certify_model

OUT_PATH = Path("docs/certification/attack.json")


def attack(
    model: LOBMamba,
    objective: str,
    layer: int,
    steps: int,
    seq_len: int,
    input_dim: int,
    box: tuple[torch.Tensor, torch.Tensor] | None,
    lr: float = 0.5,
    batch: int = 16,
    seed: int = 0,
) -> float:
    """Maximise one scalar summary over the input.

    Six of the eight objectives read a block internal from the trace; output_max and
    output_min read the prediction and skip the trace. Restarts live in the batch dimension,
    so all of them are optimised at once.
    """
    torch.manual_seed(seed)
    if box is not None:
        lo, hi = box
        x = (lo + torch.rand(batch, seq_len, input_dim) * (hi - lo)).clone()
    else:
        x = torch.randn(batch, seq_len, input_dim) * 10.0
    x.requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)

    needs_trace = objective not in ("output_max", "output_min")
    best = -float("inf")
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        if needs_trace:
            _, traces = model(x, trace=True, detach_trace=False)
            tr = traces[layer]
            value = {
                "u_abs": lambda: tr.u.abs().max(),
                "delta_max": lambda: tr.delta.max(),
                "delta_min": lambda: (-tr.delta).max(),
                "B_abs": lambda: tr.B.abs().max(),
                "C_abs": lambda: tr.C.abs().max(),
                "A_bar_max": lambda: tr.A_bar.max(),
                "h_abs": lambda: tr.h.abs().max(),
                "y_abs": lambda: tr.y.abs().max(),
            }[objective]()
        else:
            out = model(x)
            value = out.max() if objective == "output_max" else (-out).max()

        best = max(best, value.item())
        (-value).backward()
        opt.step()
        if box is not None:
            with torch.no_grad():
                x.clamp_(box[0], box[1])

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarially probe the certified bounds.")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_mamba.pt")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--box", action="store_true", help="restrict to the enforced input box")
    args = parser.parse_args()

    config = load_config()
    set_seed(int(config.get("seed", 1337)))

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", config)
        arch = ckpt.get("architecture", {})
        model = LOBMamba(int(arch.get("input_dim", 43)), config)
        model.load_state_dict(ckpt["model_state_dict"])
        norm = Normalisation.from_dict(ckpt["normalisation"]) if "normalisation" in ckpt else None
        source = str(ckpt_path)
    else:
        model = LOBMamba(43, config)
        norm = LOBDataModule(config).normalisation
        source = "randomly initialised"
    model.eval()
    input_dim = model.input_dim

    box = None
    if args.box:
        if norm is None:
            raise SystemExit("--box needs normalisation statistics")
        lo, hi = norm.input_box()
        box = (
            torch.tensor(lo, dtype=torch.float32),
            torch.tensor(hi, dtype=torch.float32),
        )

    cert = certify_model(model, seq_len=args.seq_len)
    summaries = [l["summary"] for l in cert["layers"]]

    # (objective, layer, certified bound, label, kind). Every objective maximises, so
    # soundness is always attacked <= certified. Floors are handled by maximising the
    # negated quantity. "lower" additionally negates both numbers back for display, which
    # inf delta wants; output_min keeps its already-negated form and stays "upper".
    targets = [
        ("output_max", 0, cert["output_range"]["hi"], "max output", "upper"),
        ("output_min", 0, -cert["output_range"]["lo"], "-min output", "upper"),
    ]
    for li, s in enumerate(summaries):
        targets += [
            ("u_abs", li, s["u_abs_max"], f"L{li} sup|u|", "upper"),
            ("delta_max", li, s["delta_hi"], f"L{li} sup delta", "upper"),
            ("delta_min", li, -s["delta_lo"], f"L{li} inf delta", "lower"),
            ("B_abs", li, s["B_abs_max"], f"L{li} sup|B|", "upper"),
            ("C_abs", li, s["C_abs_max"], f"L{li} sup|C|", "upper"),
            ("h_abs", li, s["h_abs_max_invariant"], f"L{li} sup|h|", "upper"),
            ("y_abs", li, s["y_abs_max_horizon"], f"L{li} sup|y|", "upper"),
        ]

    regime = "restricted to the input box" if args.box else "unrestricted input"
    print(f"model: {source}")
    print(f"attack: {args.steps} Adam steps, L={args.seq_len}, {regime}\n")
    print(f"{'quantity':16s} {'attacked':>14s} {'certified':>14s} {'ratio':>9s}  verdict")

    rows, violations = [], 0
    for objective, li, certified, label, kind in targets:
        reached = attack(
            model, objective, li, args.steps, args.seq_len, input_dim, box,
            seed=int(config.get("seed", 1337)),
        )
        violated = reached > certified + 1e-4 * max(1.0, abs(certified))
        violations += violated

        if kind == "lower":
            shown_attacked, shown_certified = -reached, -certified
            ratio = shown_attacked / shown_certified if shown_certified > 0 else float("inf")
        else:
            shown_attacked, shown_certified = reached, certified
            ratio = shown_certified / shown_attacked if shown_attacked > 0 else float("inf")

        rows.append(
            {
                "quantity": label,
                "objective": objective,
                "layer": li,
                "kind": kind,
                "attacked": shown_attacked,
                "certified": shown_certified,
                "looseness": ratio,
                "violated": bool(violated),
            }
        )
        verdict = "VIOLATION" if violated else ("tight" if ratio < 2 else "sound")
        print(
            f"{label:16s} {shown_attacked:14.4e} {shown_certified:14.4e} "
            f"{ratio:9.2f}x  {verdict}"
        )

    print()
    if violations:
        print(f"{violations} CERTIFICATE VIOLATION(S) -- the abstraction is unsound")
    else:
        print("no violations: gradient ascent could not push any quantity outside its box")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "provenance": provenance(config.get("seed")),
                "source": source,
                "regime": regime,
                "steps": args.steps,
                "seq_len": args.seq_len,
                "violations": int(violations),
                "results": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {OUT_PATH}")
    raise SystemExit(1 if violations else 0)


if __name__ == "__main__":
    main()
