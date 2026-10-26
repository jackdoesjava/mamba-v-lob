# Isabelle Theory — Build Instructions

This directory contains the Isabelle theory files for MambaVLob, along with the
session configuration needed to build them and generate browsable HTML output.

`Mamba_Invariant.thy` machine-checks the zero-order-hold identity, the forward
invariant `|h_t| <= M` with `M = sup |B u| / |A|` at every sequence length, and
the LayerNorm coordinate and fused LayerNorm-Linear bounds for every `d`. The
invariant is Horbatko's Lemma 4.4 (arXiv:2604.18580) and follows from Cao et
al.'s Theorem 3.3(i); the LayerNorm coordinate bound is the Laguerre–Samuelson
inequality. The theory checks these known results; it does not claim them.

## Prerequisites

- [Isabelle2025-2](https://isabelle.in.tum.de/) installed.
- On **Windows**, the `isabelle` command is not available in a normal
  `cmd`/PowerShell terminal. Instead:
  1. Go to your Isabelle installation folder.
  2. Run `Cygwin-Terminal.bat`. This opens a bash shell with `isabelle` on
     the `PATH`.
  3. Use this Cygwin terminal for all commands below. Windows drives are
     mounted under `/cygdrive/`, e.g. `C:\Users\Name\repo` is
     `/cygdrive/c/Users/Name/repo`.
- No LaTeX installation is required — this session is configured for
  **HTML output only** (no PDF document build).

## Directory contents

| File/Folder      | Description                                                       |
|-------------------|--------------------------------------------------------------------|
| `ROOT`            | Isabelle session configuration; lists `Mamba_Invariant` only |
| `Mamba_Invariant.thy` | Theory source file, checked                   |
| `drafts/Mamba_Sensitivity.thy` | **Unchecked draft**, not in `ROOT`; see below |
| `MambaVLob/`      | **Pre-built** HTML output, committed to the repo so collaborators can browse the theory without building it themselves |
| `output/`, `document/` | May appear during build (**not tracked**, see `.gitignore`) |

**Note:** `MambaVLob/` is intentionally tracked in git, unlike a typical
Isabelle setup. It's a copy of Isabelle's generated `browser_info` output,
checked in so collaborators can view the rendered theory directly from the
repo without installing Isabelle or running a build. If you don't have
Isabelle installed and just want to read the theory, open
`MambaVLob/index.html` in a browser and skip the rest of this file.

## The unchecked draft

`drafts/Mamba_Sensitivity.thy` states the lemmas behind the sensitivity bound
of `../docs/02-invariant.md`: the timescale rescaling bound
`|exp(-x) - exp(-x e^s)| <= |e^s - 1| (1 - exp(-x))` (`exp_rescale_bound`),
the error recursion for two runs of the convex update (`error_recursion`), and
the cell bound `|h t - g t| <= max |h 0 - g 0| (eta + 2 M k)`
(`cell_sensitivity_bound`). It was written without access to Isabelle and has
never been loaded, so every proof method in it is unverified and nothing in it
may be cited as machine-checked. It imports `../Mamba_Invariant`.

To check it, add `"drafts/Mamba_Sensitivity"` under `theories` in `ROOT` and
run the build below, or open it in Isabelle/jEdit against this session:

```bash
isabelle jedit -d . -l MambaVLob drafts/Mamba_Sensitivity.thy
```

Move it out of `drafts/`, keep it in `ROOT` and regenerate `MambaVLob/` only
once it builds with no errors.

## Building (only needed if you've changed a `.thy` file)

If you've edited the theory and need to regenerate the HTML, from this
directory in the Cygwin terminal (Windows) or a normal terminal
(Linux/macOS):

```bash
isabelle build -v -D . -o browser_info -c
```

- `-D .` — use the `ROOT` file in the current directory.
- `-o browser_info` — enable HTML presentation output.
- `-c` — clean first, forcing a full rebuild (recommended, since Isabelle
  otherwise skips regenerating output if it thinks the session is already
  up to date).

A successful build prints something like:

```
Session Unsorted/MambaVLob
Running MambaVLob ...
MambaVLob: theory MambaVLob.Mamba_Invariant
MambaVLob: theory MambaVLob.Mamba_Invariant 100% (...s cumulated time)
```

with no errors.

## Viewing the HTML output

Isabelle writes output to the location reported by:

```bash
isabelle getenv ISABELLE_BROWSER_INFO
```

Look for:

```
<ISABELLE_BROWSER_INFO>/Unsorted/MambaVLob/index.html
```

**After regenerating, copy the updated `MambaVLob` folder from
`ISABELLE_BROWSER_INFO` into this directory, replacing the existing one, and
commit the changes.** This keeps the checked-in HTML in sync with the
current `.thy` source. Only `output/` and `document/` (the PDF/LaTeX
build leftovers) stay gitignored — `MambaVLob/` itself should be committed.

## Notes

- If you want PDF output as well, add `options [document = pdf]`
  and a `document_files "root.tex"` line in `ROOT`, and install a LaTeX
  distribution (e.g. MiKTeX or TeX Live) providing `lualatex`.
