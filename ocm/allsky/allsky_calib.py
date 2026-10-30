"""Clear-sky calibration of the star expectation from nights the user declares clear.

The star statistics count an expected star as missing whatever the reason. On a clear sky some stars are lost all the
same: on the low northern edge (thicker air, lens vignetting), next to the Moon, close doubles and stars in crowded
fields. A night the user knows was clear shows how often each star is seen where it is expected; the stars and places a
clear sky does not show reliably are then left out (allsky_lib.calibrated_out).

To be redone by hand only when the camera changes: suspected dirt on the dome, after it is cleaned, a new lens or
camera. The clear nights should together have frames without the Moon (for the stars and the alt x az map) and with a
bright Moon (for the distance from the Moon); a passing cloud does not spoil them, frames with fewer than MIN_SEEN of
the expected stars seen are left out.

    python allsky_calib.py calibrate 20261011 20261007 20261029    # -> CALIB_NPZ (also collects what is missing)
    python allsky_calib.py collect 20261011 ...                    # per-star tables only -> CALIB_DIR/stars_<night>.parquet
"""

import os
import sys

import numpy as np
import pandas as pd
from allsky_lib import *

CALIB_DIR = os.path.dirname(CALIB_NPZ)
ALT_EDGES = np.arange(30.0, 90.1, 5.0)
AZ_EDGES = np.arange(0.0, 360.1, 22.5)
DIST_EDGES = np.array([0.0, 30, 35, 40, 45, 50, 60, 80, 180])
MIN_RATE = 0.8        # a star, alt x az cell or Moon distance seen in fewer of the cases is left out
MIN_UNRELIABLE = 0.5  # a star seen in fewer than this share of its chances is unreliable
MIN_CHANCES = 10      # chances a star needs to be judged
MIN_CELL = 20         # expected stars a cell needs to be judged
MIN_SEEN = 0.8        # frames with fewer of the expected stars seen are not taken as clear


def star_table(an, img, t):
    """Per-star measurements of one frame (SkyAnalyzer.analyse with details, without calibration).

    Returns
    -------
    pandas.DataFrame
        star (index into the catalogue), v, alt, az, x, y, amp, sigma, peak, detected, expected (by zp_clear); plus the
        frame's moon_x, transp and zp_ref repeated on every row.
    """
    r = an.analyse(img, t, details=True)
    st = r["stars"]
    return pd.DataFrame(dict(star=st["idx"], v=st["v"], alt=st["alt"], az=st["az"], x=st["x"], y=st["y"], amp=st["amp"], sigma=st["sigma"], peak=st["peak"],
                             detected=st["detected"], expected=st["expected"], moon_x=r["moon_x"], transp=r["transp"], zp_ref=r["zp_ref"]))


def collect_night(an, day, sun_max=-16.0):
    """star_table of every dark frame of `day` -> CALIB_DIR/stars_<day>.parquet (one row per star per frame)."""
    parts = []
    for p in dark_frames(day, sun_max=sun_max):
        d = star_table(an, read_frame(p), frame_time(p))
        d.insert(0, "frame", os.path.basename(p))
        parts.append(d)
    if not parts:
        raise ValueError(f"no dark frames on {day}")
    os.makedirs(CALIB_DIR, exist_ok=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(f"{CALIB_DIR}/stars_{day}.parquet")
    print(day, len(parts), "frames", len(out), "rows", flush=True)


def chances(day):
    """The expected stars of the clear frames of a collected night, with seen and the distance from the Moon.

    Expected as in SkyAnalyzer (star_expectation on frame_zero_point), without calibration.
    """
    tab = pd.read_parquet(f"{CALIB_DIR}/stars_{day}.parquet")
    rows = []
    for fr, d in tab.groupby("frame", sort=True):
        st = {k: d[k].values for k in ("v", "amp", "sigma", "peak", "detected")}
        zp = frame_zero_point(st)
        e = star_expectation(st, zp if np.isfinite(zp) else d.zp_ref.iloc[0])
        if not e["expected"].any() or e["seen"][e["expected"]].mean() < MIN_SEEN:
            continue
        t = frame_time("/x/" + fr)
        moon_alt, moon_az = (float(a) for a in body_altaz("moon", t))
        a1, a2 = np.radians(moon_alt), np.radians(d.alt.values)
        dist = np.degrees(np.arccos(np.clip(np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(np.radians(d.az.values - moon_az)), -1, 1)))
        with np.errstate(invalid="ignore", divide="ignore"):
            clean = e["seen"] & (st["peak"] < 240) & (st["v"] > 2) & (st["v"] < 4.5) & (st["amp"] > 0)
            dim = np.where(clean, st["v"] + 2.5 * np.log10(np.clip(st["amp"], 1e-3, None)) - zp, np.nan)
        rows.append(pd.DataFrame(dict(frame=fr, star=d.star.values, alt=d.alt.values, az=d.az.values % 360, dist=np.where(moon_alt > 0, dist, np.nan),
                                      seen=e["seen"], dim=dim))[e["expected"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["frame", "star", "alt", "az", "dist", "seen", "dim"])


def calibrate(days, path=CALIB_NPZ):
    """Calibration from the clear nights `days` -> path (npz, read by load_calibration).

    Step by step, each on what the previous ones leave: stars seen in fewer than MIN_UNRELIABLE of at least
    MIN_CHANCES chances are unreliable; alt x az cells (ALT_EDGES x AZ_EDGES) and distances from the Moon (DIST_EDGES)
    get the share of the expected stars seen there (NaN with fewer than MIN_CELL stars). The dimming of the seen
    stars (their zero point minus the frame's): its median per star (at least MIN_CHANCES measurements), then the
    median of the rest per alt x az cell.

    Returns
    -------
    dict
        As load_calibration.
    """
    if not days:
        raise ValueError("no nights given")
    d = pd.concat([chances(day) for day in days], ignore_index=True)
    if d.empty:
        raise ValueError("no clear frames in the given nights")
    per_star = d.groupby("star").seen.agg(["mean", "size"])
    unreliable = per_star.index[(per_star["size"] >= MIN_CHANCES) & (per_star["mean"] < MIN_UNRELIABLE)].values.astype(int)
    d = d[~d.star.isin(unreliable)]
    ia = np.digitize(d.alt, ALT_EDGES) - 1
    iz = np.digitize(d.az, AZ_EDGES) - 1
    rate = np.full((len(ALT_EDGES) - 1, len(AZ_EDGES) - 1), np.nan)
    cells = d.assign(ia=ia, iz=iz).groupby(["ia", "iz"]).seen.agg(["mean", "size"])
    for (a, z), r in cells.iterrows():
        if r["size"] >= MIN_CELL and 0 <= a < rate.shape[0] and 0 <= z < rate.shape[1]:
            rate[a, z] = r["mean"]
    good = np.nan_to_num(rate[np.clip(ia, 0, rate.shape[0] - 1), np.clip(iz, 0, rate.shape[1] - 1)], nan=1.0) >= MIN_RATE
    m = d[good & d.dist.notna().values]
    by_dist = m.groupby(np.digitize(m.dist, DIST_EDGES) - 1).seen.agg(["mean", "size"])
    moon_rate = np.full(len(DIST_EDGES) - 1, np.nan)
    for i, r in by_dist.iterrows():
        if r["size"] >= MIN_CELL and 0 <= i < len(moon_rate):
            moon_rate[i] = r["mean"]
    dm = d[d.dim.notna()]
    per_star_dim = dm.groupby("star").dim.agg(["median", "size"])
    per_star_dim = per_star_dim[per_star_dim["size"] >= MIN_CHANCES]["median"]
    star_dim = np.full(int(d.star.max()) + 1, np.nan)
    star_dim[per_star_dim.index.values.astype(int)] = per_star_dim.values
    dm = dm.assign(rest=dm.dim - star_dim[dm.star.values.astype(int)]).dropna(subset=["rest"])
    altaz_dim = np.full(rate.shape, np.nan)
    cells = dm.assign(ia=np.digitize(dm.alt, ALT_EDGES) - 1, iz=np.digitize(dm.az, AZ_EDGES) - 1).groupby(["ia", "iz"]).rest.agg(["median", "size"])
    for (a, z), r in cells.iterrows():
        if r["size"] >= MIN_CELL and 0 <= a < rate.shape[0] and 0 <= z < rate.shape[1]:
            altaz_dim[a, z] = r["median"]
    cal = dict(unreliable=unreliable, alt_edges=ALT_EDGES, az_edges=AZ_EDGES, altaz_rate=rate, dist_edges=DIST_EDGES, moon_rate=moon_rate,
               min_rate=np.float64(MIN_RATE), star_dim=star_dim, altaz_dim=altaz_dim, nights=np.array(days))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **cal)
    print(f"{path}: {d.frame.nunique()} clear frames of {len(days)} nights, {len(unreliable)} unreliable stars, "
          f"{int((rate < MIN_RATE).sum())} of {int(np.isfinite(rate).sum())} alt x az cells below {MIN_RATE}, seen vs Moon distance {np.round(moon_rate, 2)}; "
          f"dimming known for {int(np.isfinite(star_dim).sum())} stars (spread {np.nanstd(star_dim):.2f} mag), per cell {np.nanmin(altaz_dim):.2f}..{np.nanmax(altaz_dim):.2f} mag")
    return cal


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("collect", "calibrate"):
        sys.exit(__doc__)
    an = SkyAnalyzer(stars=load_bsc(BSC_CSV))
    days = sys.argv[2:]
    if sys.argv[1] == "collect":
        for day in days:
            collect_night(an, day)
    else:
        for day in days:
            if not os.path.exists(f"{CALIB_DIR}/stars_{day}.parquet"):
                collect_night(an, day)
        calibrate(days)
