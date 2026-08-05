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
    locate the TRGB. Fully docstringed with explicit input validation (`raise ValueError` on
    out-of-range params) — match that style if extending it.
  - `filters.ipynb`, `lf.ipynb` — example notebooks exercising `trgb_lib.py`. `filters.ipynb` still
    has older Sobel-filter/GLOESS/MLA edge-detection code defined inline rather than in the library —
    those are earlier experiments, not the current method (MLE + LPED).
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
