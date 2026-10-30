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

JupyterLab comes with the `notebook` dependency, so `uv run jupyter lab` starts it from `.venv`. The
`.venv` is also registered as the kernel **`mglab (.venv)`**
(`.venv/bin/python -m ipykernel install --user --name mglab`) for a JupyterLab started from elsewhere;
notebooks must use that kernel or the `.venv` `python3`.

`pyproject.toml` declares every third-party package the notebooks import (astropy, photutils, scipy,
matplotlib, ephem, tqdm, jsonschema, ipywidgets for `tqdm.notebook`, pandas, pyarrow, nats-py for `ocm/allsky/fetch_nats.py`) — keep it that
way: add new ones with `uv add <package>`, not `pip install`. The one exception is `pyaraucaria`,
needed only by `ocm/`: it is still in a testing state (`ffs` is being developed on branch
`mgorski_dev`), so for now it is an editable install of the local checkout `~/src/pyaraucaria` (see
`ocm/` below) rather than a declared dependency. On a fresh machine: clone it
(`git clone -b mgorski_dev git@github.com:araucaria-project/pyaraucaria.git ~/src/pyaraucaria`) and
`uv pip install -e ~/src/pyaraucaria`. `uv sync` removes it, so reinstall it after every sync. Once
pyaraucaria/`ffs` settles, make it a normal dependency in `pyproject.toml`.

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
  - `sky_brightness.ipynb` — sky-background vs. Moon position/phase using `ephem` and
    `astropy.coordinates`.
  - `colors_trans.ipynb` — small ad hoc color-transformation snippets, Polish comments.
  - `examples.ipynb` — personal Python/astropy/Jupyter cheat-sheet (progress bars, file I/O, FITS
    I/O, coordinates), not an analysis notebook.
  - `syntax_tpg.ipynb` — currently contains unresolved git merge-conflict markers and won't parse as
    valid JSON.
- `ocm/` — OCM observatory data-quality notebooks, working on the raw frames in
  `/work/vela/oca/fits/<tel>/raw/<night>/` (`<night>` = `int(JD) - 2460000`; the date-named dirs there
  are symlinks to the numbered ones). They use `pyaraucaria.ffs` (Fast FITS Statistics) from the
  local checkout `~/src/pyaraucaria`, installed editable into `.venv` and on branch `mgorski_dev`,
  where `ffs` is under development — results from different pyaraucaria commits may not be comparable.
  - `ffs_scan.ipynb` — current one: runs the full FFS pipeline over the last N nights of one telescope
    (wk06/zb08/jk15), raw frames or ofp-reduced `*_zdf.fits` (`SOURCE`), newest first, and appends one
    row per frame (selected header cards, full header as text, all scalar `stats["frame"]`, line/star
    counts, Moon/Sun, `ffs_version` = pyaraucaria commit) to a parquet database outside the repo,
    `~/data/ffs_db/<tel>/` (raw) or `~/data/ffs_db/<tel>_zdf/` (chunked files; read with its
    `load_db(tel, source)`, which keeps the last row per frame). Re-running skips frames already there.
  - `ffs_scan.py` — the same scan as a command-line script for long unattended runs
    (`python ffs_scan.py --telescopes zb08 --sources raw zdf --nights 15`), writing the same database.
    Its functions are a copy of the notebook's, so a change to the measurement must go into both.
  - `ffs_test.ipynb`, `ocm_dqc.ipynb` — older versions of the same idea, on the old `ffs` API
    (`from ffs import FFS`, pickles of `stats` dicts). Superseded by `ffs_scan.ipynb`; the analysis
    notebook for the new database is still to be written.
  - `ocm_data_stats.ipynb` — counts science/flat frames per night for an annual observation report.
  - `allsky/` — cloud detection on the OCM allsky camera archive `/data/misc/allsky/<YYYYMMDD>/<YYYYMMDD>_HHMM.jpg`
    (NFS; UT names, ~2.4 min cadence at night, 1240x1256 JPG, effectively greyscale, burned-in overlays).
    - `allsky_lib.py` — the deliverable, to be lifted into the cloud service (conventions of `trgb_lib.py`: docstrings,
      `raise ValueError`). Frame access, camera model (`CAMERA_PARAMS`), masks, derotation, star measurement and the
      clear-sky expectation, and the method: `SkyAnalyzer.analyse` (one frame: `transp`, `star_seen`, `diff_ls_n`, star
      maps, difference map, `detections`) and `StateTracker` (state in time, `veil`); `detections(..., veil)` gives the
      observers' marks `sure` (red) and `possible` (yellow). Rebuilt 2026-10-29 (segments and old indicators removed,
      one star pass and one derotation per frame, ~0.8 s; values identical to before).
    - `allsky_case.ipynb` — the current notebook: one case (a run of frames, chosen in `CASES`) animated: the frame with
      the marks, the raw difference and the expected stars with their own detections, indicators + areas with the state
      band. `CASE = "live"` = the newest frames.
    - `allsky_regions.py` — study around the method: `RegionAnalyzer` (analyser + tracker + table row), whole nights into
      `~/data/allsky/regions/<night>.{parquet,npz,png}` (`nights`, `plot`), drawing (`draw*`), the report's `cases`, and
      the check against the telescopes (`telescopes` → `regions/telescopes.parquet`, `marks`).
    - `allsky_diff.py` — study: the difference against a clear reference frame (`reference`; one run ≤ 1 h, it keeps every
      frame in memory).
    - `allsky_calib.py` — optional clear-sky star calibration from nights the user declares clear →
      `~/data/allsky/calib/allsky_calib.npz`; by hand after cleaning or a camera change.
    - `allsky_camera.py` — fits `CAMERA_PARAMS` on bright stars of clear frames; once per camera.
    - `allsky_report.py` — the HTML report of the method as it is now → `~/data/allsky/report/allsky_report.html`.
    - `fetch_nats.py` — downloads (read-only, NATS keeps 30 days) the telescopes' zero points and raw-frame pointings into
      `~/data/allsky/nats/`.
    - `~/data/allsky/attic/` — the study up to 2026-10-29 (old report HTML and scripts, `allsky_cache.py`; caches in
      `~/data/allsky/cache/`). Not used any more.
    - **Memory**: never build an astropy `Time` from a string without `format=` in per-frame code — astropy guesses the
      format by raising and catching exceptions, whose tracebacks keep the callers' frames (with the full images) in
      reference cycles (+36 MB per frame). Closed matplotlib figures need `gc.collect()`. Run heavy jobs one at a time,
      capped (`systemd-run --user --scope -p MemoryMax=6G ...`): two parallel runs OOM-crashed the laptop on 2026-10-29.
    - Times: sky positions from the file name (`frame_time`); intervals between frames from the mtimes (`frame_mtime`,
      off by 15-220 s as an absolute time).
    - Agreed with the user: reading many frames is discussed first; batches of nights are reported before going on.
    - The service `allsky_overwatch` lives in the oca_overwatch repo (v2 on this library, on `master`, not yet deployed);
      requirements in the vault note `Projects/Automation/allsky_overwatch.md`; image / mask storage is a question for
      majkelx (vault `Inbox/For majkelx — allsky_overwatch image storage (2026-10-28).md`).
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
  - `global_param_mc.ipynb` — the same idea for the simulation parameters: a scenario says of each of
    `N_1mag`, `R`, `photometric_error` whether it is held at a number or drawn from a `[lo, hi]`
    range, all scenarios share the realizations, and each is a colour on the plot.
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
  `numpy`/`scipy`/`math`) so that pattern keeps working.

## What the TRGB work is for

`trgb/` is a **laboratory**, not the end product. The goal is to try out and compare TRGB detection
methods, and — just as importantly — to find good ways of **estimating the detection error**. That is
why several "diagnostics" exist side by side: they are under active development and are expected to
change, be replaced, or be thrown out.

The main instrument is **Monte Carlo**: simulate a luminosity function with known input parameters,
run detection on it, repeat over many realizations, and see how the recovered TRGB and its error
behave as a function of the input data. One notebook per MC case; so far there are two,
`d_param_mc.ipynb` (choice of the LPED half-window `d`) and `global_param_mc.ipynb` (the simulation
parameters `N_1mag`, `R`, `photometric_error`).

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
- **Stay in scope.** Work is scoped to `trgb/` unless the user says otherwise (`ocm/` is the other
  active area, when the user asks for it). Problems noticed
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

## Current state (last updated: 2026-10-30)

### `ocm/allsky/` — cloud detection (sessions of 2026-10-27..30)

- **The method is settled and documented** (`allsky_lib.py`, report `~/data/allsky/report/allsky_report.html`): stars
  expected from the frame's own zero point (optional calibration) → `transp`, `star_seen`, star maps; derotated difference
  → `diff_ls_n`, difference regions; `StateTracker` → clear / thin / cloudy / overcast + **veil** (transp below thin while
  star_seen ≥ 0.8); `zp_clear` **learned** (median of clear-looking frames, 14 nights; (9.03, −2.45) on Oct 2026 vs the
  hand fit (8.95, −2.10)) — to be watched for two weeks and checked back. Marks (user): red = hides everything (star sure,
  or difference sure only where stars cannot tell), yellow = cirrus / uncertain / whole sky under a veil, light cyan area = Moon.
  Fixed segments dropped for good. Checked on ten nights and against the telescopes (report 3–4); levels kept (too few
  cloudy telescope points to tune).
- **Service `allsky_overwatch` v2 in production on services01 since 2026-10-30** (unit `oca_allsky_overwatch`, run as
  poweruser; the user updates it himself: `git pull`, restart). Publishes `…allsky_overwatch.clouds` per night frame (state, veil,
  marks shares, indicators, `zp_coeffs`, `files`); messages on `tic.status.ocm.messages` only when the sky gets worse /
  clears: thin/veil → `CLOUDS INFO`, cloudy → `CLOUDS WARNING`, overcast → `CLOUDS ALERT`, clear → `INFO`; `CLOUDS INFO`
  "<tel> is pointing into (thick) clouds" for an observing telescope in yellow/red. Display pictures (not an archive) for
  the last 10 frames in `/data/misc/oca_overwatch/allsky/` (dir made by the user, poweruser 775): `_marks.png` (codes 0–3),
  `_stars.jpg`, `_diff.jpg` (on the frame itself: star rings, difference blended in colour; Moon = light cyan area, the
  whole circle above the horizon). `marks` shares in the message exclude the veil (flag `veil`); `telescopes` field =
  where each observing telescope points (red / yellow / null). zp history `~/.local/state/allsky_overwatch/`, seeded
  from the repo. Memory flat so far (the user watches it for a few days). All committed and pushed (oca_overwatch
  `master` 1b159bb); services01 needs `git pull` + restart for the latest commits.
- **Around it, all committed and pushed (2026-10-30)**: TOI (`main` d1e0fa4): field Clouds (one word, grey / yellow /
  orange / red), lamp ALLSKY OW, colours of the CLOUDS labels, cloud icon in the telescope table. weather_overwatch's
  zero point dated by `meta.ts` (zero_monitor arrives 5–45 min late). oca_monitor (`master` bd6fc45): mask overlay on the
  allsky page, tabs AllskyStars / AllskyDiff out of the rotation, page **Sky** in panel 02 (clouds, phot_zero, fwhm;
  Weather keeps wind, temperature, humidity, power), chart `clouds` (state band — veil hatched —, yellow / red sky
  shares, current-time line), TOI's icons in the telescope table incl. the cloud. Convention (user): allsky pictures
  shown as the camera takes them (north turned by theta from the burned-in N/E/S/W); telescope dots and the wind
  arrow follow the camera model. Sounds for the CLOUDS labels (`pages/message.py`) still the user's to pick.
- **Tone curve (2026-10-30, in mglab `allsky_lib.py` only — the service copy now differs)**: false star marks next to the
  Moon come from the JPEG tone curve (a star on a brighter background looks up to ~1 mag fainter). `RESPONSE_BKG` /
  `RESPONSE_DIM` + `response_dim` (relative to the zero point's stars: zp / transp unchanged) in `star_expectation(dim=)`
  and the star dim map; measured on clear twilights during telescope flats and clear moonlit nights (they agree).
  Checked on 10-27/29/30 (half of the yellow near the Moon gone, red gone, cloudy night ~-12 % yellow). Study scripts in
  the session scratchpad only. To do: whole nights + report, then copy to the service; learn the curve automatically from
  flat twilights; maybe ring noise around the Moon for the rest.
- **Next**: the clear reference frame study (`allsky_diff.py`: ageing, cloudy night
  start, join the marks?); a Sun term in zp_clear.
- **This machine**: `~/src/pyaraucaria` missing, so `ocm/ffs_*` cannot run. Old study material in `~/data/allsky/attic/`.
  Nothing committed in mglab: `ocm/allsky/` untracked, `CLAUDE.md`, `pyproject.toml`, `uv.lock` modified.

### `trgb/` (unchanged since 2026-08-11)

- **Only the LPED chain is under active development**: `fit_alpha_mle` → `local_poisson_filter` →
  `detect_local_peak`. Sobel / GLOESS / MLA sit in `trgb_lib.py` for comparison only — don't spend
  effort on them.
- **The plan and the results so far live in `trgb/README.md`** — "What is to be studied" for the
  order of work, "What is already established" for what is settled. Follow that order unless the user
  says otherwise.
- **Work has moved to the simulation parameters** (point 1 of the plan) in the new
  `global_param_mc.ipynb`. The `d` / `sigma` question is parked by the user's decision and comes back
  in a later session; what `d_param_mc.ipynb` was testing — whether `peak_height_scale` or
  `sharpness` collapses the error-vs-quality relation for all `d` onto one curve — still has no
  conclusion recorded.
- **The conclusion that reframed the study**, written up in the README: at fixed global parameters the
  diagnostics still order *individual realizations* by their error, so the binned curve is a
  conditional error estimate rather than a summary of the parameter class. That turns point 1 from
  "does the mean error move with the parameter" into "does the quality → error curve move, or is it
  invariant". The README lists what it does not yet settle — bias vs noise, per-`d` calibration,
  dropped detections, whether it beats plain star counts, and the truth-informed search window.
- **Error is now read as core plus catastrophic tail**, not as one number: percentile interval against
  standard deviation, drawn apart on the same plot. This should stay the convention in later MC
  notebooks.
- `R` means `N_agb_1mag / N_rgb_1mag` (so `R = 0.2` is 20 % AGB contamination), defined in
  `trgb/README.md`. Asked about in this session and deliberately left as it is, even though the same
  letter names the filter's count ratio `10**(alpha*d)` in `trgb_lib.py` docstrings.
- **`detect_local_peak` was reworked on 2026-08-06/07, so anything computed before that date is not
  comparable with what it returns now.** Every diagnostic except `peak_height` and `peak_prominence`
  either changed meaning or changed name; `trgb/README.md` documents the current set. Git history has
  the details. What matters going forward: widths are plain fractions of the peak height (no
  prominence), the peak zone is `±d`, and the two wings are reported apart.
- Known wart, agreed but not fixed: the area diagnostics are tied to `d` inside a fixed ±1 mag window
  (peak zone `±d`, wings the rest), so part of any trend against `d` is geometry, not a cleaner
  response. Needs a `d`-independent geometry before the areas can decide anything about `d`.
- Documentation is current as of this date: `trgb/README.md`, library docstrings, the LPED cells and
  the final plot of `filters.ipynb`, the final plot of `d_param_mc.ipynb`, and `global_param_mc.ipynb`
  throughout. Still to do:
  - `d_param_mc.ipynb`, `global_param_mc.ipynb` — Polish comment in the first cell (the environment
    note, the same one in both).
  - `lf.ipynb` — redefines `random_seed()` locally and reuses one seed for every stage; plot titles
    on the AGB and RGB+AGB panels both say "RGB Luminosity Function"; `10` hard-coded instead of
    `m_trgb`; `m` silently changes meaning from RGB to RGB+AGB mid-notebook.
  - `mi0.ipynb` — the JWST rows are placeholders (`# przykład: wpisz swoje...`), not adopted
    literature values, but the saved figure looks like a real compilation.
