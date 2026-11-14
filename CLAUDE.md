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
way: add new ones with `uv add <package>`, not `pip install`. `pyaraucaria` (needed by `ocm/`) is
declared too, but as an **editable path source** (`[tool.uv.sources]`: `../pyaraucaria`, i.e. the
local checkout `~/src/pyaraucaria` on branch `mgorski_dev`, where `ffs` is developed), so `uv sync`
keeps it and edits there are live. On a fresh machine clone it first
(`git clone -b mgorski_dev git@github.com:araucaria-project/pyaraucaria.git ~/src/pyaraucaria`), then
`uv sync`. Once pyaraucaria/`ffs` settles, switch the source to git or PyPI.

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
  - `calibration_check/` — detector stability from the calibration frames (zeros, darks; flats later), notebooks only,
    no library, written simply in the user's style (the user's own cell text stays as written).
    - `calib_stack.ipynb` — **the user's working notebook (ask before editing it)**. For nights `N_START..N_STOP`
      (convention: blocks of 10 nights, e.g. 1330_1340) the zeros and darks from the frames index, then per-pixel maps of
      the zeros → `~/data/ocm_bias/<tel>/<tel>_<n1>_<n2>_{mean,std,nclip}.fits` (ADU), then of the darks minus that
      bias mean, divided by each dark's EXPTIME → `~/data/ocm_dark/<tel>/<tel>_<n1>_<n2>_{mean,std_excess,nclip}.fits`
      (ADU/s, EXPTIME only in the header; one exposure time: most darks among >= 600 s with >= 4 darks; if none, a
      message and the dark cells do nothing). Images cut in bands of `SEG` rows (memory).
      Clipping: reject |x - pixel median| > 5 x max(pixel MAD, read noise) — the floor matters, with ~10 frames the
      pixel MAD alone over-clips (std 4-6x too low in 0.3 % of pixels). `std_excess` = sqrt(std_dark² - bias_std²).
    - `frames_index.ipynb` — headers of all raw frames of a night → `~/data/ocm_frames/<tel>/<night>.parquet` (one row
      per frame, full path + selected cards; skips nights already there). `calib_stack`, `bias_qc`, `dark_qc` read it;
      their old header-scan cell is kept commented out.
    - `bias_view.ipynb`, `dark_view.ipynb` — one cell per map (image + histogram, limits set by hand at the top of the
      cell); `calib_trends.ipynb` — level of every stack in time, bar = per-pixel noise.
    - `bias_qc.ipynb`, `dark_qc.ipynb` — every zero / dark of one night against a reference stack: z = (frame - ref mean)
      / expected noise; offset, zsig (MAD sigma of z; ~0.93 is normal on wk06), pixels beyond +-5, cosmic groups > 10.
    - `bias_stack.ipynb`, `dark_stack.ipynb` — the earlier separate versions (median/std_mad maps, `sigma_clip`).
  - `ghost/` — anomalies (ghosts, scattered light, periodic patterns) in reduced OCM frames (`processed-ofp/science/<night>/<frame>/<frame>_zdf.fits`), started 2026-11-01. The user is learning the method step by step: keep it simple, one field at a time, **notebook text and comments in Polish for now** (the user's request).
    - `fft_basics.ipynb` — synthetic examples and their power spectra (sines, stars: one, two, random, random with ghosts, a 3x3 grid; a line, noise), why fringes and speckle appear, "donuts" (centred, decentred, with a bright spot, rotated), vignetting + dust + ghost + noise in one flat (frame-edge cross, Hann window, radial spectrum per component), how FFT orders frequencies (1D, N = 8, fftshift, rfft).
    - `fft_fields.ipynb` — six real fields from nights 1344-1345, all at once: image and its 2D power spectrum per row (`ZOOM`); no window, no radial profile.
    - `zb08_red.ipynb` — zb08 in Ic / i / z: flats, night sky raw vs `_zdf` (fringing, not removed by OFP), FFT of a flat and a star field, and the **pupil ghost** ("donut", hole ~6 % in z) over many frames: weaker with a bright high Moon, it rotates/moves with `ROT-MECH`; sections 10-12 (2026-11-03, on night-1347 dome flats at `ROT-MECH` 79.9 and the 1345 master flats): edge profile and spectrum of the hole, band correlation between filters, the first ghost model (Fourier series in angle per ring around the hole centre), 13 the ghost separated from the Ic / V colour difference and the first corrected Ic flat, 14 the all-angle test (dome), 15 the same on OFP twilight master flats Ic and z, 16 the same-star photometry against the ghost model, each with its own "Co wyszło"; overall findings of 1-9 at the end (2026-11-01).
    - `samestar_phot.py` — same-star differential photometry from the OFP `_ap.txt` of fields seen on >= 5 nights; residual vs the change of 2.5 log(flat_z / flat_r) at the star → `~/data/ghost/<tel>_<filter>_samestar.parquet` (z, Ic, r done). Result (section 9 of `zb08_red`): slope z 0.53 ± 0.11, Ic 0.22, r 0.03 against the z/r flat ratio; against the ghost model alone (section 16) z 0.89, Ic 0.96 — the ghost is almost entirely added light.
    - `sky_maps.py` — block-median maps (16 px) of master flats and night-sky frames (raw - zero, `_zdf`, flat used) of one filter + header values → `~/data/ghost/<tel>_<filter>_{maps.npz,frames.parquet}`; frames chosen from the frames index `~/data/ocm_frames/<tel>/`. z done (43 flats, 56 frames, nights 1290-1345).
  - `flats/flat_windows.ipynb` — (2026-11-10, Polish text at the user's request) when sky flats are possible per
    telescope / filter: TOI flat log from NATS (`tic.status.<tel>.toi.flat`, appended to `~/data/flats/flat_log.parquet`,
    the stream keeps ~30 days), sky rate model vs geometric Sun altitude, windows within TIC's exposure / ADU limits,
    7/7 sequence times, a greedy evening / morning plan, figure `~/data/flats/flat_windows.png`. Vault note
    `Domain/Instruments/OCM Sky Flat Windows.md`. Summary and how to reproduce: `flats/README.md`.
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
    - `fits_*.py` — the raw FITS study (2026-11-12, `/data/misc/allsky_raw/`, kept 7 days): `fits_lib.py` (reader,
      `CAMERA_PARAMS_FITS`), `fits_photometry.py` / `fits_diff.py` per night → `~/data/allsky/fits/phot/`, then
      `fits_analyze`, `fits_local` (per-star factors), `fits_calib`, `fits_vs_telescopes` → `~/data/allsky/fits/report/`.
    - **`README.md` — the summary to keep in the repo**: data sources, method, conclusions (jpg + FITS), service status,
      how to reproduce, what can no longer be reproduced. Keep it current when the allsky work changes.
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

## Current state (last updated: 2026-11-06)

### `ocm/ghost/` — anomalies in reduced frames, zb08 pupil ghost (sessions of 2026-11-01..03, paused)

- **Done:** `fft_basics` / `fft_fields` (the user's learning notebooks), `zb08_red` sections 1-12 with findings; vault note
  `Domain/Instruments/zb08 Red Fringing and Pupil Ghost.md` is the full write-up (kept current).
- **Goal (user):** remove the ghost from the Ic / z flats without spoiling the rest of the flat.
- **Findings:** red fringing (z/Ic/i, additive, not removed by OFP); pupil ghost "donut" (hole ~6 % z, ~3 % Ic twilight)
  roughly camera-fixed, weaker under a high Moon / towards the Sun (stray-light hypothesis); same-star test: largely
  **additive** (against the ghost model alone, section 16: z 0.89, Ic 0.96, r 0.02) → OFP z photometry off by up to ~0.12 mag between hole and ring, a few % for usual shifts.
- **Dome-flat rotation series, night 1346** (science frames, OBJECT `domeflat`, Ic/V alternating, `zb08c_1346_76584` ..
  `85423`; analysed ad hoc only — block maps `~/data/ghost/zb08_1346_dome_maps.npz`, scripts
  `~/data/ghost/scratch_1346/`). The user fixed the rotator telemetry mid-series: up to `80168` `ROTATOR` == `ROT-MECH`
  (old wrong reading, angles 20-100), from `82034` on `ROT-MECH` = `ROTATOR` - 180 (angles 31.7-149.9). Ic frames order
  smoothly with angle across both scales: old ≈ new `ROT-MECH` + ~70-100 deg (rough). Hole r ~395 px shifts by tens of
  px (decentration), inner crescent and spider rotate; dome Ic hole 0.94-0.95; V nearly free (~0.3 %). Ic 300 s only
  ~1100 ADU, V 300 s saturates.
- **Night 1347 dome flats** (B / V 100 s / Ic 600 s at `ROT-MECH` 19.9, 79.9, 119.9, 159.9; z 1200 s at 19.9 only ~200 ADU;
  V 600 s saturated) added to the same block-map file; scripts `~/data/ghost/scratch_1347/`. The hole jumps ~30-40 px
  between frames independently of the angle (also within 1347) — suspect filter re-seating at each wheel move (or focus);
  a ghost model needs its centre per flat. Proposed to the user (not yet taken): Ic,Ic,Ic vs Ic,V,Ic,V at one angle, a
  focus offset, z with ~1500+ ADU, i, all at 79.9.
- **Sections 10-12:** hole edge is a circle (centre (1082, 811), r ~407 px at 79.9), steep part 30-40 px, crest ~100 px
  out; < ~32 px (PRNU) and mostly 32-64 px (dust) common to all filters, ~128-512 px the ghost band, blue filters share a
  detector-fixed ~250 px lattice absent in red → working split: < ~64 px from V, ~64-500 px ghost to model, > ~500 px
  from the filter's own flat. **First model** (section 12): per 4 px ring a Fourier series in angle (M = 2-4, 4-sigma
  clip) on Ic / V removes the ghost (large-scale rms 3.4 % → 0.2 %) and keeps small scales (corr 0.96-0.98).
- **Section 13:** the colour difference Ic / V taken as a 2nd-order polynomial S fitted where the ghost is assumed absent
  (r < 340 px and r > R_END); shape of S stable to ~0.2 % for R_END 800-1000 (700 too small); ghost model on Ic / V / S,
  corrected Ic = Ic / model — the donut is nearly gone, corrected Ic / V ≈ S.
- **Section 14 (all-angle test, 15 dome Ic flats 1346/1347, centre + S + model per flat):** rms of Ic / V between angles
  median 1.78 % → 0.43 % (same-angle floor 0.2 %). Left: spider and the inner crescent arc (both rotate).
- **Section 15 (twilight master flats, 12 evenings 1300-1345 + 2 mornings, Ic / V and z / V):** rms between sessions
  Ic 0.63 → 0.14 %, z 0.71 → 0.16 % (median). z ghost = wide annulus with an outer edge at ~850-950 px (hence R_END 1000,
  model to 1100 px there), centre nearly fixed ~(930, 958) whatever the angle; Ic twilight centres from the four cuts are
  unreliable (residual arcs where they are off). Session 1305 w (and 1306 w) is a bad twilight.
- **Section 16 (same-star photometry vs the ghost model alone):** slope z 0.89 ± 0.08, Ic 0.96 ± 0.08, r control 0.02 —
  the ghost is (almost) entirely added light; the k 0.53 / 0.22 of section 9 was diluted by real z/r flat differences in
  P. Flat / G removes the dependence; rms of moved stars z 35.0 → 29.2 mmag, Ic 36.0 → 32.0. OFP error in z up to ~0.12 mag
  between hole and ring crest (Ic ~0.07), a few % for usual field shifts. One z model (1345 w) served nights 1290-1345.
- **Open / next:** centre from a circle fit to the whole edge (Ic); spider and crescent arc as their own components;
  how to deliver it (a ghost map per filter / per twilight for OFP?) — the user's call. Whether the ghost shape is common
  to Ic / i / z (then a median over filters) waits for a good z dome flat. Nothing in `ghost/` is committed.

### `ocm/calibration_check/` — detector stability from zeros and darks (session of 2026-10-30..11-01)

- **Done:** frames index → per-pixel stacks (bias `mean/std/nclip`, dark `mean/std_excess/nclip`) → views, trends,
  per-frame QC of one night (`bias_qc`, `dark_qc`). The user works in the notebooks directly and simplifies them in
  their own style (cells without functions, settings at the top of each cell); keep their edits, don't "fix" them.
- **Data so far:** frames index for wk06 nights 1200..1340 (88 nights with frames); none for zb08 yet (the run failed:
  `/data` was gone for a while: on Wi-Fi "oca internet" the resolver 192.168.20.1 timed out, `perun.oca.lan` did not
  resolve and the autofs NFS mount vanished; it came back by itself — "oca vlan" or cable is safer). Before a scan,
  check that `/data/fits/<tel>/raw` exists: an empty archive looks like nights without frames. Bias stacks wk06 1300..1340 (four blocks; recomputed with the current clipping only where the user
  ran them) and zb08 1300_1310; dark stack wk06 1330_1340 (900 s; the only darks nearby are night 1332; earlier ones
  1263/1264/1267). Old files from earlier methods (`*_900s_*`, `*_median`, `*_std_mad`) may still lie in `~/data/ocm_*`.
- **Findings:** wk06 bias ~300.1 ADU, read noise ~7 ADU, each frame's level jumps ~0.75 ADU (no trend between
  series/nights/T-CAM), pattern incl. ~256 px blocks identical over 40 nights; no overscan. wk06 dark ~0.005 ADU/s with
  a strong gradient (bottom ~3, top ~7 ADU / 900 s), bands of 512 columns, ~1900 hot pixels > 0.1 ADU/s (hottest 29.8
  ADU/s, Poisson-stable, slightly non-linear at 1800 s). zb08 zeros have the known interference stripes (vault
  `Domain/Instruments/zb08 Camera Noise.md`): ~3 clipped values per pixel, std 8.3 vs 7.0. wk06 zero
  `wk06c_1343_50355` had extra noise only in columns 0-470 (a partial readout disturbance; global zsig hardly sees it).
- **Open / next:** user's decision on a local zsig (per column/row strips) and a local-noise panel in `bias_qc`
  (tried, reverted at the user's request); why dark zsig jumps 0.87-1.07 between darks; build the zb08 index and
  stacks; threaded header reading in `frames_index` (offered, not decided); `calib_trends` not run on a full set yet;
  flats later; OFP's per-series dirs (`processed-ofp/{zeros,darks}/`) as an alternative index, not used.
- Don't recompute stacks / scan many nights on my own: one verification case only, never while the user's kernel reads
  `/data`.

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
- **Twilight -18 (2026-10-31, uncommitted in mglab, oca_overwatch, TOI, oca_monitor)**: the camera's auto exposure lags
  at dusk (a clear sky looked overcast→thin from Sun -16 to -18). Above -18 clouds are not measured: `StateTracker(sun_max
  =-18)`, twilight out of the smoothing (both lib copies), service state "twilight" with no marks / telescopes / alarms,
  "twilight" light blue in TOI and oca_monitor. oca_monitor tabs: Clouds, Allsky (plain, out of rotation), CloudsDiff,
  CloudsStars. Veil shown as a dashed yellow circle at 30 deg + the word, no longer a whole-sky fill (lib renderers `veil=`, service, monitor).
- **Camera change 2026-11-06 (night 11-05/06, a test night: cloud values on NATS untrustworthy ~05:00 UT until the
  service swap)**: camera on RAW16; `process_ocm` (fork `~/src/allsky`, deployed on the RPi 06:59 UT) writes the 16-bit
  Bayer frame as Rice FITS with exposure / gain / temperature in the header → `/data/misc/allsky_raw/<YYYYMMDD>/
  <YYYYMMDD>_HHMM.fits` (1520 x 1021, kept 7 days), and rotates the jpg archive by -39.1 instead of -30 (north up).
  `CAMERA_PARAMS` (611.18, 623.42, 0.031, ...) prepared in oca_overwatch and oca_monitor (uncommitted, the user deploys);
  **mglab `allsky_lib.py` still has the old model** — nights from 2026-11-06 need the new one (a per-date model). To do:
  refit with `allsky_camera.py` on clear frames after the switch; use the FITS exposure (twilight, tone curve). Vault:
  `Projects/Automation/allsky_pipeline.md`.
- **Raw FITS study (2026-11-12, nights 11-06..12, no Moon yet)**: FITS star photometry (constant zp 13.86, per-star
  factors, local transparency within 12 deg) beats the jpg against the telescopes (18/21 vs 10/21); the difference
  gains nothing. In the service since 2026-11-13 (star part from FITS when valid). Details in `ocm/allsky/README.md`.
- **The analysis laptop stays at the observatory (2026-11-14)**: `~/data/` is not synchronised; the repo carries the
  conclusions and the reproduction steps in `ocm/allsky/README.md` and `ocm/flats/README.md`.
- **Next**: the clear reference frame study (`allsky_diff.py`: ageing, cloudy night
  start, join the marks?); a Sun term in zp_clear.
- **This machine**: `~/src/pyaraucaria` (mgorski_dev) cloned, a declared editable dependency since 2026-11-01. Old study material in `~/data/allsky/attic/`.
  mglab `ocm/allsky/` committed and pushed 2026-10-30 at the user's request (8e9fa6b); `ocm/allsky/.idea/` left out.

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
