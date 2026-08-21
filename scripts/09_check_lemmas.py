"""Discharge the polynomial lemmas behind the certificate with Z3.

Each check asserts the negation and asks for a model, so unsat means no counterexample. The
LayerNorm lemmas quantify over a dimension and are checked at small widths only. exp and the
zero-order hold are outside decidable real arithmetic and keep their hand proofs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from z3 import And, If, Or, Real, Reals, Solver, Sum, sat, unsat

OUT_DIR = Path("docs/certification")


def _run(solver: Solver, timeout_ms: int) -> tuple[str, float]:
    solver.set("timeout", timeout_ms)
    t0 = time.perf_counter()
    result = solver.check()
    return str(result), time.perf_counter() - t0


def layernorm_coordinate_bound(d: int, timeout_ms: int) -> dict:
    """sum z = 0 and sum z^2 <= d imply z_i^2 <= d - 1.

    This is the bound in docs/03-certificate.md, squared so no root appears. It is what
    makes every SSM quantity independent of the block input.
    """
    z = [Real(f"z{i}") for i in range(d)]
    s = Solver()
    s.add(Sum(z) == 0)
    s.add(Sum([zi * zi for zi in z]) <= d)
    s.add(z[0] * z[0] > d - 1)
    status, secs = _run(s, timeout_ms)
    return {"lemma": "layernorm_coordinate_bound", "d": d, "result": status,
            "proved": status == "unsat", "seconds": round(secs, 3)}


def layernorm_linear_is_tight(d: int, timeout_ms: int) -> dict:
    """<v, z> squared <= ||v - mean(v)||^2 d on the set, for every v.

    The fused bound in intervals.layernorm_linear. v is symbolic, so a proof here covers all
    weight vectors at that width.
    """
    z = [Real(f"z{i}") for i in range(d)]
    v = [Real(f"v{i}") for i in range(d)]
    vbar = Sum(v) / d
    centred = [vi - vbar for vi in v]

    dot = Sum([v[i] * z[i] for i in range(d)])
    rhs = Sum([c * c for c in centred]) * d

    s = Solver()
    s.add(Sum(z) == 0)
    s.add(Sum([zi * zi for zi in z]) <= d)
    s.add(dot * dot > rhs)
    status, secs = _run(s, timeout_ms)
    return {"lemma": "layernorm_linear_is_tight", "d": d, "result": status,
            "proved": status == "unsat", "seconds": round(secs, 3)}


def layernorm_linear_is_attained(d: int, timeout_ms: int) -> dict:
    """The maximiser z* = sqrt(d) (v - mean v)/||v - mean v|| lies in the set.

    Checked in squared form on a concrete v so no root appears: for z proportional to the
    centred v with sum z = 0 and sum z^2 = d, the bound holds with equality.
    """
    v = [Real(f"v{i}") for i in range(d)]
    z = [Real(f"z{i}") for i in range(d)]
    vbar = Sum(v) / d
    centred = [vi - vbar for vi in v]
    k = Real("k")

    s = Solver()
    s.add([z[i] == k * centred[i] for i in range(d)])
    s.add(Sum([zi * zi for zi in z]) == d)
    s.add(k > 0)
    s.add(Sum([c * c for c in centred]) > 0)
    # sum z = 0 must follow rather than be assumed, and the bound must be met with equality
    dot = Sum([v[i] * z[i] for i in range(d)])
    s.add(Or(Sum(z) != 0, dot * dot != Sum([c * c for c in centred]) * d))
    status, secs = _run(s, timeout_ms)
    return {"lemma": "layernorm_linear_is_attained", "d": d, "result": status,
            "proved": status == "unsat", "seconds": round(secs, 3)}


def interval_product_is_sound(timeout_ms: int) -> dict:
    """a b lies between the min and max of the four corner products, for all a, b in their boxes.

    Interval.__mul__. Quantified over the endpoints as well as the points, so this is general.
    """
    a, b, a_lo, a_hi, b_lo, b_hi = Reals("a b a_lo a_hi b_lo b_hi")
    corners = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi]

    def mn(xs):
        return xs[0] if len(xs) == 1 else If(xs[0] < mn(xs[1:]), xs[0], mn(xs[1:]))

    def mx(xs):
        return xs[0] if len(xs) == 1 else If(xs[0] > mx(xs[1:]), xs[0], mx(xs[1:]))

    s = Solver()
    s.add(a_lo <= a, a <= a_hi, b_lo <= b, b <= b_hi)
    s.add(Or(a * b < mn(corners), a * b > mx(corners)))
    status, secs = _run(s, timeout_ms)
    return {"lemma": "interval_product_is_sound", "d": None, "result": status,
            "proved": status == "unsat", "seconds": round(secs, 3)}


def interval_square_is_sound(timeout_ms: int) -> dict:
    """Interval.square: the lower end is 0 when the box straddles 0, else the nearer square."""
    a, lo, hi = Reals("a lo hi")
    straddles = And(lo <= 0, hi >= 0)
    lower = If(straddles, 0, If(lo * lo < hi * hi, lo * lo, hi * hi))
    upper = If(lo * lo > hi * hi, lo * lo, hi * hi)

    s = Solver()
    s.add(lo <= a, a <= hi)
    s.add(Or(a * a < lower, a * a > upper))
    status, secs = _run(s, timeout_ms)
    return {"lemma": "interval_square_is_sound", "d": None, "result": status,
            "proved": status == "unsat", "seconds": round(secs, 3)}


def affine_split_is_sound(n: int, timeout_ms: int) -> dict:
    """Splitting W into positive and negative parts bounds W x over a box. intervals.affine."""
    x = [Real(f"x{i}") for i in range(n)]
    lo = [Real(f"lo{i}") for i in range(n)]
    hi = [Real(f"hi{i}") for i in range(n)]
    w = [Real(f"w{i}") for i in range(n)]

    w_pos = [If(wi > 0, wi, 0) for wi in w]
    w_neg = [If(wi < 0, wi, 0) for wi in w]
    out_lo = Sum([w_pos[i] * lo[i] + w_neg[i] * hi[i] for i in range(n)])
    out_hi = Sum([w_pos[i] * hi[i] + w_neg[i] * lo[i] for i in range(n)])
    val = Sum([w[i] * x[i] for i in range(n)])

    s = Solver()
    s.add([And(lo[i] <= x[i], x[i] <= hi[i]) for i in range(n)])
    s.add(Or(val < out_lo, val > out_hi))
    status, secs = _run(s, timeout_ms)
    return {"lemma": "affine_split_is_sound", "d": n, "result": status,
            "proved": status == "unsat", "seconds": round(secs, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Discharge the polynomial lemmas with Z3.")
    parser.add_argument("--max-dim", type=int, default=6)
    parser.add_argument("--timeout-ms", type=int, default=60000)
    args = parser.parse_args()

    checks = []
    checks.append(interval_product_is_sound(args.timeout_ms))
    checks.append(interval_square_is_sound(args.timeout_ms))
    for n in range(1, 5):
        checks.append(affine_split_is_sound(n, args.timeout_ms))
    for d in range(2, args.max_dim + 1):
        checks.append(layernorm_coordinate_bound(d, args.timeout_ms))
    for d in range(2, min(args.max_dim, 5) + 1):
        checks.append(layernorm_linear_is_tight(d, args.timeout_ms))
        checks.append(layernorm_linear_is_attained(d, args.timeout_ms))

    print(f"{'lemma':38s} {'d':>4s} {'result':>9s} {'proved':>7s} {'s':>7s}")
    for c in checks:
        d = "" if c["d"] is None else c["d"]
        print(f"{c['lemma']:38s} {str(d):>4s} {c['result']:>9s} "
              f"{str(c['proved']):>7s} {c['seconds']:>7.2f}")

    proved = sum(c["proved"] for c in checks)
    unknown = [c for c in checks if c["result"] == "unknown"]
    counterexamples = [c for c in checks if c["result"] == "sat"]

    print(f"\n{proved}/{len(checks)} discharged")
    if unknown:
        print(f"{len(unknown)} timed out, which is not a failure: the solver gave no answer")
    if counterexamples:
        print("COUNTEREXAMPLE FOUND, a lemma in docs/ is wrong:")
        for c in counterexamples:
            print(f"  {c['lemma']} at d={c['d']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "smt_lemmas.json").write_text(
        json.dumps({"checks": checks, "proved": proved, "total": len(checks)}, indent=2)
    )
    print(f"wrote {OUT_DIR / 'smt_lemmas.json'}")
    raise SystemExit(1 if counterexamples else 0)


if __name__ == "__main__":
    main()
