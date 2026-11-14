# Allsky cloud detection — summary (status 2026-11-14)

Cloud detection on the OCM allsky camera, as a library (`allsky_lib.py`) plus study scripts. The library became the
production service `allsky_overwatch` (repo oca_overwatch). This file records **what was found and how to reproduce
it** on a machine with access to the observatory data; all intermediate results live outside the repo
(`~/data/allsky/` on the analysis laptop) and are not synchronised. Full history: vault notes
`Projects/Automation/allsky_overwatch.md`, `allsky_pipeline.md`, `allsky FITS study (2026-11-12).md`.

## Data sources

| source | where | kept | used by |
|---|---|---|---|
| jpg archive | `/data/misc/allsky/<YYYYMMDD>/<YYYYMMDD>_HHMM.jpg` (NFS) | long (back to 2026-01) | `allsky_*.py` |
| raw FITS (RAW16, from 2026-11-06 06:59 UT) | `/data/misc/allsky_raw/<YYYYMMDD>/<YYYYMMDD>_HHMM.fits` | **7 days** | `fits_*.py` |
| telescope zero points, raw-frame pointings | NATS JetStream `tic_status` (`nats://nats.oca.lan:4222`) | **30 days** | `fetch_nats.py`, `fits_fetch_nats.py` |
| service messages `…allsky_overwatch.clouds` | NATS | 30 days | `fits_fetch_nats.py` |
| Yale Bright Star Catalogue | `~/data/allsky/bsc5.csv`, made by `allsky_lib.fetch_bsc()` from GitHub | — | everything |

jpg: 1240 x 1256, R = G = B, UT names, ~2.4 min cadence at night, burned-in overlays, a non-linear tone curve, and
from 2026-11-06 a min-max stretched 8-bit view of the RAW16 frame rotated by -39.1 deg (north up). FITS: 1520 x 1021
uint16, Rice compressed, true exposure start in CAPDATE/CAPTIME (RPi local clock), EXPTIME, GAIN, CCD-TEMP.

## The method (jpg, `allsky_lib.py`)

- **Stars**: BSC stars expected at their model positions (`CAMERA_PARAMS`, fitted by `allsky_camera.py`); measured
  against the expectation from the frame's own zero point → `transp` (whole-sky, mag), `star_seen` (share of the
  expected stars found), star maps. Optional clear-sky calibration per (alt, az) and vs. Moon distance
  (`allsky_calib.py`, nights 20261007, 20261011, 20261029 → `~/data/allsky/calib/allsky_calib.npz`).
- **Difference**: consecutive frames derotated onto each other; `diff_ls_n` = std of the smoothed difference over the
  pixel noise; difference regions = moving cloud.
- **State in time** (`StateTracker`): clear / thin / cloudy / overcast + **veil** (transp below "thin" while
  star_seen ≥ 0.8 = a uniform thin layer). `zp_clear` is learned (median of clear-looking frames). Above Sun −18 deg
  nothing is measured (the camera's auto exposure lags at dusk; state "twilight").
- **Marks** for the observers (`detections`): red = hides everything (stars sure, or difference sure where stars
  can't tell), yellow = cirrus / uncertain / whole sky under a veil, light cyan = Moon area.
- One frame ≈ 0.8 s (one star pass, one derotation).

## Conclusions

**jpg study (10 nights 2026-10-01..29, checked against the telescopes' zero points):**
- Stars and difference are complementary: stars catch still veils and forming cloud (difference blind); the
  difference catches moving cirrus and arriving cloud ~40 min before the weather alert (stars weak there).
- Fixed sky segments were dropped for good; local star maps + difference regions instead.
- False star marks near the Moon come from the JPEG tone curve (a star on a brighter background looks up to ~1 mag
  fainter); a measured correction (`RESPONSE_BKG` / `RESPONSE_DIM`, `star_expectation(dim=)`) removes about half of
  the yellow near the Moon — in this repo's `allsky_lib.py` only, **not** in the service copy.
- Evening auto exposure lags: from Sun −13 to −18 a clear sky looks overcast → thin; hence the −18 deg rule.
- Learned `zp_clear` drifts when nights from before a camera change get mixed in (2026-11-09: 9.10 → 9.24 → false
  thin/veil); the service now learns only from frames after 2026-11-06 07 UT.
- Too few cloudy telescope points to tune the levels; they stay as set by hand.

**Raw FITS study (nights 2026-11-06..12, all near new Moon — no Moon test yet):**
- The FITS are linear but **not raw sensor data**: dark-subtracted upstream (`darkSubtract.sh` picks the dark by
  temperature only → black clipping at short exposures, up to 18 % of pixels = 0 at 1-9 s) and resized 3096 → 1520
  (Bayer phases averaged). Usable only with zenith sky ≥ ~1500 ADU and exposure ≥ ~3 s until fixed on the RPi.
- Camera model for FITS: `(773.32, 500.51, -39.077, 5.5165, -3.775e-05)` (`fits_lib.CAMERA_PARAMS_FITS`, rms 1.3 px
  above alt 30).
- **Star photometry works after dividing by EXPTIME**: constant clear-sky zp ≈ 13.86 (13.81-13.89 per night) with no
  learning, no Moon model, no tone curve; stars V 3.5-5.5, amplitude < 30 000 ADU; per-star clear-sky factors absorb
  colour and flat (`fits_local.build_star_factors`).
- **Local transparency at a telescope's pointing** (flux sum / expected of stars V 4-5.5 within 12 deg) at −0.15 mag
  catches 18/21 telescope losses > 0.2 mag with 2.9 % false; the jpg service state 10/21 with 8.3 % false
  (only 3 cloud episodes — not proven).
- The derotated difference gains nothing from FITS (follows the jpg one, Spearman 0.80-0.90).
- Sky brightness (ADU/s) is no cloud indicator on moonless nights.

**In production** (oca_overwatch, services01, unit `oca_allsky_overwatch`): v2 on this library since 2026-10-30;
since 2026-11-13 the star part uses the FITS when valid (`"source": "fits"`, constant zp 13.86, per-star factors in
`fits_star_factors.csv`, telescope icons from transparency within 12 deg: −0.15 yellow, −0.8 red, overcast −2.5),
difference and marks still from the jpg.

## Reproducing

Setup: `uv sync` in the repo root; `/data` (NFS from perun) mounted — use "oca vlan" or cable, on Wi-Fi "oca
internet" the mount drops. Run scripts from `ocm/allsky/`. **Heavy jobs one at a time, memory-capped**
(`systemd-run --user --scope -p MemoryMax=6G python ...`): two parallel runs crashed the laptop.

1. Star catalogue: `python -c "import allsky_lib; allsky_lib.fetch_bsc()"` → `~/data/allsky/bsc5.csv`.
2. Camera model (only after a camera change): `python allsky_camera.py <clear moonless jpg frames>`; for FITS
   `python fits_camera.py <frames.fits>`. Paste the result into `CAMERA_PARAMS` / `CAMERA_PARAMS_FITS`. Note: the lib
   still has the pre-2026-11-06 jpg model; nights after that need `(611.18, 623.42, 0.031, 6.21458, -3.7e-05)`.
3. Clear-sky calibration (optional): `python allsky_calib.py calibrate 20261011 20261007 20261029`.
4. Telescope comparison data (NATS, last 30 days only): `python fetch_nats.py` → `~/data/allsky/nats/`.
5. Whole nights: `python allsky_regions.py nights|plot|telescopes 20261001 20261005 20261007 20261011 20261017
   20261018 20261025 20261027 20261028 20261029` → `~/data/allsky/regions/`.
6. Clear-reference study: `python allsky_diff.py reference 20261005 0130 0230 out.png`.
7. Report: `python allsky_report.py` → `~/data/allsky/report/allsky_report.html`.
8. Single cases interactively: `allsky_case.ipynb` (`CASES`, or `CASE = "live"`).

FITS study (only for nights still in the 7-day FITS archive; the 2026-11-06..12 frames are already gone):
`python fits_inventory.py inv.csv DAY ...`, `python fits_photometry.py DAY ...` and `python fits_diff.py DAY ...` →
`~/data/allsky/fits/phot/`; `python fits_fetch_nats.py` (set `START`); then `fits_analyze.py` →
`report/frames_all.parquet`, `fits_local.py` (star factors), `fits_calib.py`, `fits_vs_telescopes.py` →
`report/tel_match.parquet`.

**Not reproducible any more** (sources expired): the FITS photometry of 2026-11-06..12 and every NATS dump older than
30 days (telescope zero points of October, service messages of 2026-11-06..12). The numbers above for those are as
recorded here and in the vault.

## Open

- Test FITS photometry on moonlit nights (short exposures) and stop the black clipping on the RPi (darks off, or save
  the FITS before `saveImage.sh`).
- Refit the jpg camera model after 2026-11-06 and make `CAMERA_PARAMS` per date; tone curve learned from flat
  twilights, then copied to the service.
- With FITS (known exposure) the −18 deg twilight rule could go; a Sun term in `zp_clear`.
- Clear-reference difference (`allsky_diff.py`): ageing, start of a cloudy night.

`image-20261106*.png` are screenshots of RAW16 test frames from the camera reconfiguration night.
