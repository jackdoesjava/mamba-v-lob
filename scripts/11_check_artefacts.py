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

PRODUCED: dict[str, list[str]] = {
    "docs/certification/bounds.json": [
        "scripts/06_certify.py", "src/verification/invariant.py", "src/verification/certify.py",
    ],
    "docs/certification/bounds.md": [
        "scripts/06_certify.py", "src/verification/invariant.py", "src/verification/certify.py",
    ],
    "docs/certification/sweep.json": ["scripts/10_sweep.py", "src/verification/invariant.py"],
    "docs/certification/sweep.md": ["scripts/10_sweep.py", "src/verification/invariant.py"],
    "docs/certification/attack.json": [
        "scripts/07_attack_bounds.py", "src/verification/certify.py",
    ],
    "docs/certification/domains.json": [
        "scripts/08_domains.py", "src/verification/certify.py",
        "src/verification/lipschitz.py", "src/verification/sensitivity.py",
    ],
    "docs/certification/smt_lemmas.json": ["scripts/09_check_lemmas.py"],
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
    "docs/figures/certified_vs_realised.pdf": ["scripts/05_figures.py"],
    "docs/figures/certified_output_range.pdf": ["scripts/05_figures.py"],
    "docs/figures/dm_test_heatmap.pdf": ["scripts/05_figures.py"],
    "docs/figures/regime_conditioned_ic.pdf": ["scripts/05_figures.py"],
    "docs/figures/latency_complexity.pdf": ["scripts/05_figures.py"],
}

# Phrases from claims this project retracted. None of them should reach an artefact.
RETRACTED = [
    "no finite invariant set",
    "makes the state bound usable",
    "delta floor is required",
    "dt_min is necessary",
]


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

    for path in Path("docs").rglob("*.md"):
        text = path.read_text(encoding="utf-8").lower()
        for phrase in RETRACTED:
            if phrase in text:
                retracted.append((str(path), phrase))
    for path in list(Path("models/bounds").glob("*.json")):
        text = path.read_text(encoding="utf-8").lower()
        for phrase in RETRACTED:
            if phrase in text:
                retracted.append((str(path), phrase))

    print(f"artefacts checked : {len(PRODUCED)}")
    print(f"missing           : {len(missing)}")
    print(f"stale             : {len(stale)}")
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
