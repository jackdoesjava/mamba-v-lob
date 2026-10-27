"""Check that every committed artefact is newer than the code that produced it.

Nothing fails when a figure or a results table goes stale, which is how a plot arguing a
retracted claim survived in this repository for several hours. Run this before submitting
anything or before quoting a number from docs/.

Exits non-zero if an artefact is missing or older than its sources.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

# scripts/14_released.py, on the released checkpoint and the sample text
RELEASED_SOURCES = [
    "scripts/14_released.py", "src/verification/released.py", "src/verification/incremental.py",
    "src/verification/intervals.py", "src/verification/invariant.py",
    "src/verification/lipschitz.py", "src/models/mamba.py", "data/text/public_domain_sample.txt",
]

# scripts/16_cell_tightness.py, a search over the cell alone, with no checkpoint or data
CELL_TIGHTNESS_SOURCES = [
    "scripts/16_cell_tightness.py", "src/verification/incremental.py",
    "src/verification/intervals.py", "src/models/mamba.py",
]

PRODUCED: dict[str, list[str]] = {
    "docs/certification/bounds.json": [
        "scripts/06_certify.py", "src/verification/invariant.py", "src/verification/certify.py",
    ],
    "docs/certification/sweep.json": ["scripts/10_sweep.py", "src/verification/invariant.py"],
    "docs/certification/attack.json": [
        "scripts/07_attack_bounds.py", "src/verification/certify.py",
    ],
    "docs/certification/domains.json": [
        "scripts/08_domains.py", "src/verification/certify.py",
        "src/verification/lipschitz.py", "src/verification/sensitivity.py",
    ],
    "docs/certification/smt_lemmas.json": ["scripts/09_check_lemmas.py"],
    "docs/certification/incremental.json": [
        "scripts/12_incremental.py", "src/verification/incremental.py",
        "src/verification/certify.py", "src/verification/intervals.py",
        "src/verification/invariant.py", "src/verification/lipschitz.py", "src/dataset.py",
        "src/models/mamba.py",
    ],
    "docs/certification/incremental_attack.json": [
        "scripts/13_incremental_attack.py", "src/verification/incremental.py",
        "src/verification/certify.py", "src/verification/intervals.py",
        "src/verification/invariant.py", "src/verification/lipschitz.py", "src/dataset.py",
        "src/models/mamba.py",
    ],
    "models/bounds/mamba_certificate.json": [
        "scripts/04_export_bounds.py", "src/verification/certify.py",
        "src/verification/invariant.py", "src/verification/export.py",
        "src/verification/reference.py",
    ],
    "models/results/test_metrics.json": [
        "scripts/03_evaluate_models.py", "src/utils/stats.py", "src/dataset.py",
    ],
    "docs/figures/bound_vs_floor.pdf": [
        "scripts/05_figures.py", "src/verification/invariant.py",
    ],
    "docs/figures/bound_vs_horizon.pdf": [
        "scripts/05_figures.py", "src/verification/invariant.py",
    ],
    "docs/figures/gap_prediction.pdf": [
        "scripts/05_figures.py", "src/verification/invariant.py",
    ],
    "docs/figures/sweep_seeds.pdf": ["scripts/05_figures.py", "scripts/10_sweep.py"],
    "docs/figures/certified_vs_realised.pdf": ["scripts/05_figures.py"],
    "docs/figures/certified_output_range.pdf": ["scripts/05_figures.py"],
    "docs/figures/dm_test_heatmap.pdf": ["scripts/05_figures.py"],
    "docs/figures/regime_conditioned_ic.pdf": ["scripts/05_figures.py"],
    "docs/figures/latency_complexity.pdf": ["scripts/05_figures.py"],
    "docs/figures/incremental_vs_length.pdf": [
        "scripts/12_incremental.py", "src/verification/incremental.py",
        "src/verification/certify.py", "src/verification/intervals.py",
        "src/verification/invariant.py", "src/models/mamba.py",
    ],
    "docs/figures/sensitivity_vs_length.pdf": [
        "scripts/13_incremental_attack.py", "src/verification/incremental.py",
        "src/verification/certify.py", "src/verification/intervals.py",
        "src/verification/invariant.py", "src/verification/lipschitz.py", "src/dataset.py",
        "src/models/mamba.py",
    ],
    "docs/certification/released.json": RELEASED_SOURCES,
    "docs/figures/released_layers.pdf": RELEASED_SOURCES,
    "docs/figures/sensitivity_local.pdf": [
        "scripts/15_local_figures.py", "docs/certification/incremental_attack.json",
    ],
    "docs/figures/end_to_end.pdf": [
        "scripts/15_local_figures.py", "docs/certification/incremental_attack.json",
    ],
    "docs/figures/released_local.pdf": [
        "scripts/15_local_figures.py", "docs/certification/released.json",
    ],
    "docs/certification/cell_tightness.json": CELL_TIGHTNESS_SOURCES,
    "docs/figures/cell_tightness.pdf": CELL_TIGHTNESS_SOURCES,
}

# Written by the same run as their JSON, so they are current exactly when it is. A table
# that comes out identical when regenerated gets no new commit, which would otherwise make
# it look older than its sources.
TWINS: dict[str, str] = {
    "docs/certification/bounds.md": "docs/certification/bounds.json",
    "docs/certification/sweep.md": "docs/certification/sweep.json",
    "docs/certification/incremental.md": "docs/certification/incremental.json",
    "docs/certification/incremental_attack.md": "docs/certification/incremental_attack.json",
    "docs/certification/released.md": "docs/certification/released.json",
    "docs/certification/cell_tightness.md": "docs/certification/cell_tightness.json",
}

# Phrases from claims this project retracted. None of them should reach an artefact, the README
# or the docs. Matching ignores case.
RETRACTED = [
    "no finite invariant set",
    "makes the state bound usable",
    "delta floor is required",
    "dt_min is necessary",
    # the gap depends on the weights through the slowest pole lam = |A|
    "contains no trained weight",
    "involves no trained weight",
    # the separated and additive bounds are finite wherever their floor is positive
    "no finite bound at all",
    # interval evaluation of the induction step fails at M = M_c when the transition box has
    # width, and can pass for a smaller equilibrium bound
    "fails for every box that is not a single point",
    "for any box that is not a point",
    "for any non-degenerate box",
    # the exponential-Euler step is a convex combination too, with a ceiling in place of a floor
    "needs the exact discretisation to exist",
    "needs this discretisation to exist",
    # the forward invariant is Horbatko's Lemma 4.4 and Cao et al.'s Theorem 3.3(i)
    "first closed-form, architecture-specific",
    # the two Mamba arms share seed, initialisation and batch order; DM p = 0.621
    "favouring the constrained parametrisation",
    "the parametrisation costs no accuracy",
    # the order book data is SPY; the mis-scaled file was named after the wrong instrument
    "gme_features",
    "gme on 2021",
    "gme, 2021",
]

# Prose and artefacts scanned for retracted phrases.
SCANNED = ["README.md", "isabelle/README.md", "docs/**/*.md", "docs/certification/*.json",
           "models/bounds/*.json"]


def timestamps(use_git: bool) -> dict[str, float]:
    """Commit times if available, so a fresh clone does not look stale from checkout order."""
    if not use_git:
        return {}
    try:
        out = subprocess.run(
            ["git", "log", "--name-only", "--format=%ct"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except Exception:
        return {}

    seen: dict[str, float] = {}
    stamp = 0.0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.isdigit():
            stamp = float(line)
        elif line not in seen:
            seen[line] = stamp
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Check artefacts are current.")
    parser.add_argument("--mtime", action="store_true", help="use file times, not commit times")
    args = parser.parse_args()

    commit_time = timestamps(not args.mtime)

    def when(path: str) -> float:
        if path in commit_time:
            return commit_time[path]
        p = Path(path)
        return p.stat().st_mtime if p.exists() else 0.0

    missing, stale, retracted = [], [], []

    for artefact, sources in PRODUCED.items():
        if not Path(artefact).exists():
            missing.append(artefact)
            continue
        made = when(artefact)
        for source in sources:
            if Path(source).exists() and when(source) > made:
                stale.append((artefact, source))
                break

    stale_now = {artefact for artefact, _ in stale}
    for twin, json_file in TWINS.items():
        if not Path(twin).exists():
            missing.append(twin)
        elif json_file in stale_now:
            stale.append((twin, json_file))

    scanned = sorted({path for pattern in SCANNED for path in Path(".").glob(pattern)})
    for path in scanned:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in RETRACTED:
            if phrase.lower() in text:
                retracted.append((str(path), phrase))

    print(f"artefacts checked : {len(PRODUCED) + len(TWINS)}")
    print(f"missing           : {len(missing)}")
    print(f"stale             : {len(stale)}")
    print(f"files scanned     : {len(scanned)}")
    print(f"retracted phrases : {len(retracted)}")

    for a in missing:
        print(f"  MISSING  {a}")
    for a, s in stale:
        print(f"  STALE    {a}  <- {s} is newer")
    for a, phrase in retracted:
        print(f"  CLAIM    {a} contains {phrase!r}")

    if missing or stale or retracted:
        print("\nregenerate with the script sequence in README.md")
        raise SystemExit(1)
    print("\neverything current")


if __name__ == "__main__":
    main()
