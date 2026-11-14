# Sky flat windows — summary (status 2026-11-14)

When twilight sky flats are possible for each OCM telescope (wk06, zb08, jk15) and filter, how long a 7/7 sequence
takes, and in what order to take the filters. One notebook, `flat_windows.ipynb` (analysis 2026-11-10; text in
Polish at the user's request). Vault note: `Domain/Instruments/OCM Sky Flat Windows.md`; related
`Projects/Automation/flat_overwatch.md`, `Projects/Science & Processing/sfs (Sky Flat Selector).md`.

## Data

- TOI flat log from NATS: `tic.status.<tel>.toi.flat` (one message per frame: `mean`, `exp_time`, `filter`,
  `timestamp_utc`); after the 2026-11-10 migration also `data.toi.<tel>.camera.flat_log`. The stream keeps ~30 days.
- The notebook appends new frames to `~/data/flats/flat_log.parquet` (outside the repo), so the file collects a
  longer history than NATS. The analysis used **1025 frames, 2026-10-29..11-07**; nothing came on either subject
  after 11-07 although flats were taken on 11-08/09 (not investigated).
- TIC limits (from TIC config `sky_flats`): exposure 2-120 s and 20-25 kADU for wk06 / zb08; 10-120 s and 40-60 kADU
  for jk15.

## Method

- Sky rate = (mean − bias) / exp_time; bias 300 (wk06, zb08), 3817 (jk15, mode g-1_r-3).
- Sun altitude computed by hand (ephem, geometric, no refraction) at the exposure midpoint = timestamp − exp/2 − 5 s.
  **TOI's `h_sun` is up to 1.5 deg too high near the horizon** (refraction applied below it) — don't use it.
- Model per telescope: log10 rate = a1·h + a2·h² + s·(morning) + c_filter, frames with h > −9.5 deg; residuals
  0.04-0.07 dex. Morning sky slightly darker than evening at the same h.
- Window = Sun altitudes where the exposure (target level / rate, clipped to the TIC range) gives a level inside the
  ADU range. Overheads from the log: per frame wk06 ~12 s, zb08 ~15 s, jk15 ~35 s; filter change + test ~31 s
  (wk06/zb08), ~83 s (jk15). Greedy evening / morning plan for one night (2026-11-10/11).

## Conclusions

- **wk06, zb08**: each filter's window ~16-22 min, a 7/7 sequence ~2-2.5 min at its fastest → all 9 wk06 filters in
  one session (~22 min), 7 of 8 for zb08. Evening order darkest first — wk06: Hb_n, u_s, Hb_w, v_s, y_s, b_s, V, Ic,
  B; zb08: u, z, i / Ic / r / V, B, g. Morning reversed.
- **jk15**: windows only ~11-13 min (10-120 s is only a factor 12, plus 35 s readout) and 7/7 takes ~7 min → about
  4 filters per session (u_s, Hb_n, Hb_w, then one broadband). Darkness order (darkest first): u_s, Hb_n, Ha_n, u,
  Hb_w, v_s, z, Ha_w, y_s, b_s, i, Ic, V, r, B, g. Lever: a shorter minimum exposure or a faster readout mode.
- Windows start (evening) at Sun ~−1.5 to −5 deg and end at ~−6 to −8.5 deg — inside the sfs session [+1, −15].

## Reproducing

`uv sync`, NATS reachable (`nats://nats.oca.lan:4222`, OCA network), run `flat_windows.ipynb` top to bottom: cell 1
reads the log from NATS (read-only) and appends to `DATA`, the rest fits the model and draws the windows / plan
(`NIGHT` sets the night; figure → `~/data/flats/flat_windows.png`). Paths `DATA` and the figure are hard-coded to
`/home/mgorski/data/flats/` — adjust on another machine. **The 2026-10-29..11-07 frames expire from NATS around
2026-12-07**; after that this exact sample can only come from a saved `flat_log.parquet`. A rerun on a newer sample
should give the same windows within a minute or two (the model has only a few parameters per telescope).

## Open

- Why the flat log stopped after 2026-11-07 (TOI / migration).
- Refit with more nights and seasons (the twilight shape depends on the season / Moon); jk15 exposure range.
