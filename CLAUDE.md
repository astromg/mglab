# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal astronomy research lab: a loose collection of Jupyter notebooks for photometry, stellar
population, and observatory data-quality experiments, plus one small shared Python module. There is
no test suite, no linter config, and no CI — treat this as exploratory/scratch research code, not a
packaged library.

## Environment setup

Uses `uv` with `pyproject.toml` / `uv.lock`, `requires-python = ">=3.12"`.

```
uv sync
uv run jupyter notebook   # or: jupyter lab
```

Note: `pyproject.toml` only declares `notebook` as a dependency. Notebooks import several third-party
packages (astropy, photutils, scipy, matplotlib, ephem, tqdm, jsonschema) that are installed ad hoc
into `.venv` rather than declared in `pyproject.toml`/`uv.lock`. Don't assume the lockfile is a
complete or accurate picture of what a given notebook needs — check the notebook's own import cells,
and install missing packages with `uv add <package>` when working on one that fails to import.

There are no build, lint, or test commands — nothing in the repo defines them.

## Layout

- Top-level `*.ipynb` — one notebook per investigation, independent of each other unless noted below.
  Notable ones:
  - `photometry.ipynb`, `cosmic_ray_detection.ipynb`, `satelite_detection.ipynb` — image-level
    analysis on FITS frames using `astropy` + `photutils`/`scipy.ndimage` (source detection, PSF
    photometry, artifact rejection). `hot_piksel_detection.ipynb` is an empty placeholder.
  - `lmc_cmd.ipynb`, `red_lmc.ipynb` — near-duplicate LMC/SMC reddening notebooks (Red Clump fitting
    on a CMD, curve fitting, path-based region selection). `red_lmc.ipynb` is the more complete,
    later version; `lmc_cmd.ipynb` references `f_name`/`result_fname` without defining them and
    can't run standalone.
  - `ocm_data_stats.ipynb` — counts science/flat frames per night for an annual observation report;
    `ocm_dqc.ipynb` — data quality control, validates FITS headers against a JSON schema with
    `jsonschema`.
  - `sky_brightness.ipynb` — sky-background vs. Moon position/phase using `ephem` and
    `astropy.coordinates`.
  - `ffs_test.ipynb` — depends on an `ffs` module that lives in a sibling project
    (`pyaraucaria`/`TOI`, outside this repo), not on anything checked into `mglab`. It won't import
    unless that sibling project is also on `PYTHONPATH`.
  - `colors_trans.ipynb` — small ad hoc color-transformation snippets, Polish comments.
  - `examples.ipynb` — personal Python/astropy/Jupyter cheat-sheet (progress bars, file I/O, FITS
    I/O, coordinates), not an analysis notebook.
  - `syntax_tpg.ipynb` — currently contains unresolved git merge-conflict markers and won't parse as
    valid JSON.
- `trgb/` — Tip-of-the-Red-Giant-Branch (TRGB) distance-indicator work. See
  [`trgb/README.md`](trgb/README.md) for the theory. All reusable logic lives in `trgb_lib.py`; the
  notebooks are thin usage examples on top of it (`from trgb_lib import *`), not where the logic
  lives — don't add analysis code directly into a notebook if it belongs in the library instead.
  - `trgb_lib.py` — the one real shared module in the repo: RGB/AGB luminosity-function simulation,
    magnitude↔flux conversion, blending, photometric-error injection, completeness modeling, the RGB
    slope MLE fit, and the Local Poisson Edge Detector (LPED) + peak-quality diagnostics used to
    locate the TRGB. Also holds the older Sobel/GLOESS/MLA detectors, kept for comparison. The
    simulation half is docstringed with explicit `raise ValueError` input validation — that is the
    target style; the detection half has drifted from it (see "Current state").
  - `lf.ipynb` — walk-through of the simulation chain (RGB → AGB → blending → errors →
    completeness), one plot per stage. `filters.ipynb` — walk-through of the detection chain on a
    single simulated LF.
  - `d_param_mc.ipynb` — Monte Carlo study of the LPED `d` (half-window) parameter: repeats the
    simulate→detect chain over many realizations for a set of `d` values and compares peak height /
    TRGB error.
  - `mi0.ipynb` — unrelated to the simulation/detection code; compiles literature TRGB calibration
    values into forest plots.
- `data/` — gitignored local data dir; don't assume its contents are reproducible or committed.
- `.ipynb_checkpoints/`, `trgb/.ipynb_checkpoints/` — Jupyter autosave checkpoints, not source of truth.

## Working with notebooks

- When editing notebook code, prefer `NotebookEdit` over hand-editing the underlying JSON.
- Since there are no tests, verify numerical/plotting changes by actually re-running the affected
  cells rather than relying on static review.
- `trgb_lib.py` is imported by notebooks via `from trgb_lib import *` with the notebook's CWD on
  `sys.path` (no package install) — keep it a flat, dependency-light module (currently only
  `numpy`/`math`) so that pattern keeps working.

## What the TRGB work is for

`trgb/` is a **laboratory**, not the end product. The goal is to try out and compare TRGB detection
methods, and — just as importantly — to find good ways of **estimating the detection error**. That is
why several "diagnostics" exist side by side: they are under active development and are expected to
change, be replaced, or be thrown out.

The main instrument is **Monte Carlo**: simulate a luminosity function with known input parameters,
run detection on it, repeat over many realizations, and see how the recovered TRGB and its error
behave as a function of the input data. One notebook per MC case; so far there is one,
`d_param_mc.ipynb` (choice of the LPED half-window `d`).

Two consequences that should drive design decisions here:

- **`trgb_lib.py` is the deliverable.** When the method settles, the library gets lifted out of this
  repo into a separate application that analyses real data. The notebooks stay behind. So the library
  must remain self-contained, portable and free of notebook-specific assumptions — and analysis logic
  belongs in it, not in a cell.
- **The notebooks are the experiments.** Each one is a case study, kept separate on purpose. Don't
  merge them or generalize one into a framework unless asked.

## How we work here

- **Language.** The user talks Polish; reply in Polish. Everything written to a file — this document,
  READMEs, docstrings, code comments, commit messages — is in **English**.
- **One session = one topic.** The user closes the session when the topic is done. There is no
  running log of sessions: "Current state" below is the only status carry-over, and it is
  **rewritten in place**, never appended to. At the end of a session (the user says e.g. "podsumuj i
  zapisz") update it to the new reality, drop what is no longer true, keep it to a few bullets. Git
  history is the record of what changed; that section is the record of where things stand.
- **The user makes the commits.** Never run `git commit`/`git push`. Say what is done and leave it
  staged-or-not for them.
- **This file is editable without asking.** Update `CLAUDE.md` whenever something here goes stale.
- **Running code is allowed** without asking — run notebook cells or scripts to verify a change.
  Read-only shell commands are pre-approved in `.claude/settings.local.json`.
- **Missing packages:** install them yourself with `uv add <package>`, no need to ask.
- **Stay in scope.** Work is scoped to `trgb/` unless the user says otherwise. Problems noticed
  elsewhere in the repo get mentioned, not fixed (e.g. `syntax_tpg.ipynb` has unresolved merge
  markers — leave it alone).

## Code style

- Match the user's own style; parts of this code came from an LLM, parts are hand-written, and the
  hand-written parts win as the reference. Notably: **do not break a simple statement across several
  lines** just because it is long — if it is one straightforward command, keep it on one line.
- Write code the user can read at a glance. **If a method or algorithm is not obvious, ask before
  using it** — an unfamiliar clever trick in this codebase is a defect, not a feature.
- Follow the existing conventions of `trgb/trgb_lib.py`: full docstrings and explicit `raise
  ValueError` input validation.
- **Don't silently break what already works.** Renaming a function, reordering or renaming its
  arguments, or changing what it returns breaks the user's notebooks and invalidates results they
  have already computed and understood. Propose such a change and wait for a yes; the same goes for
  quietly swapping the method behind an existing function.

## Current state (last updated: 2026-08-07)

- **Only the LPED chain is under active development**: `fit_alpha_mle` → `local_poisson_filter` →
  `detect_local_peak`. Sobel / GLOESS / MLA sit in `trgb_lib.py` for comparison only — don't spend
  effort on them.
- **The plan for the whole study now lives in `trgb/README.md`** ("What is to be studied"): sweep the
  simulation parameters one at a time, then in pairs as heatmaps, then the method parameters, then
  pick the diagnostic that best predicts the error, then check what bootstrap misses. Follow that
  order unless the user says otherwise.
- **The open question is still the half-window `d`**, and it has grown into the question of a
  `d`-independent quality statistic. `peak_height` scales as `sqrt(d)`, so raw heights don't compare
  across `d`; `d_param_mc.ipynb` now records a hand-scaled `peak_height_scale` (`peak_height *
  sqrt(0.1/d)`) next to `sharpness`, which carries the same factor implicitly. The test is whether
  either of them collapses the `absolute_error`-vs-quality relation for all `d` onto one curve. No
  conclusion recorded yet.
- **`detect_local_peak` was reworked on 2026-08-06/07, so anything computed before that date is not
  comparable with what it returns now.** Every diagnostic except `peak_height` and `peak_prominence`
  either changed meaning or changed name; `trgb/README.md` documents the current set. Git history has
  the details. What matters going forward: widths are plain fractions of the peak height (no
  prominence), the peak zone is `±d`, and the two wings are reported apart.
- Known wart, agreed but not fixed: the area diagnostics are tied to `d` inside a fixed ±1 mag window
  (peak zone `±d`, wings the rest), so part of any trend against `d` is geometry, not a cleaner
  response. Needs a `d`-independent geometry before the areas can decide anything about `d`.
- Documentation is current as of this date: `trgb/README.md`, library docstrings, the LPED cells and
  the final plot of `filters.ipynb`, the final plot of `d_param_mc.ipynb`. Still to do:
  - `d_param_mc.ipynb` — Polish comment in the first cell.
  - `lf.ipynb` — redefines `random_seed()` locally and reuses one seed for every stage; plot titles
    on the AGB and RGB+AGB panels both say "RGB Luminosity Function"; `10` hard-coded instead of
    `m_trgb`; `m` silently changes meaning from RGB to RGB+AGB mid-notebook.
  - `mi0.ipynb` — the JWST rows are placeholders (`# przykład: wpisz swoje...`), not adopted
    literature values, but the saved figure looks like a real compilation.
