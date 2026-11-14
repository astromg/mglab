"""Derotated difference of consecutive allsky FITS frames (count rates), one night at a time.

Per frame pair (Sun below SUN_MAX, gap < MAX_GAP): diff_ls_n as in allsky_lib (std of the 13-px box-smoothed difference
over the pixel noise of the difference), plus linear-data variants: diff_rel = the same smoothed std over the median
sky rate (fractional change of the sky brightness), diff_pois = smoothed std over the expected smoothed noise from
the noise of the current frame (not damped by the change itself). Output ~/data/allsky/fits/phot/diff_<day>.parquet.

    python fits_diff.py DAY [DAY ...]
"""
import os, sys, time
import numpy as np
import pandas as pd
from astropy.utils import iers
iers.conf.auto_download = False
iers.conf.auto_max_age = None
from allsky_lib import derotate, body_altaz, _smooth_in, _pixel_noise
from fits_lib import read_fits, day_fits, CAMERA_PARAMS_FITS, FITS_SHAPE, SATURATION
from fits_photometry import ALT, OUT

P = CAMERA_PARAMS_FITS
SUN_MAX = -6.0
MAX_GAP = 400.0
SMOOTH_PX = 13           # ~15 jpg px
SKY = ALT >= 30
ZEN = ALT >= 60


def ls_std(d, valid):
    d0 = np.where(valid, d - np.median(d[valid]), 0.0)
    return float(np.std(_smooth_in(d0, valid, SMOOTH_PX)[valid]))


def main(days):
    for day in days:
        rows, prev = [], None
        t0 = time.time()
        for path in day_fits(day):
            try:
                img, h, t = read_fits(path)
            except Exception as e:
                print("ERR", path, e, file=sys.stderr); prev = None; continue
            if float(body_altaz("sun", t)[0]) > SUN_MAX:
                prev = None; continue
            exp = float(h["EXPTIME"])
            sat = img >= SATURATION
            rate = img / exp
            r = dict(frame=os.path.basename(path), t=t.isot, exptime=exp)
            if prev is not None and 0 < t.unix - prev[1] < MAX_GAP:
                dt = t.unix - prev[1]
                der = derotate(prev[0], dt, P)
                d = rate - der
                valid = SKY & np.isfinite(d) & ~sat & ~prev[2]
                sky = float(np.median(rate[valid]))
                ls = ls_std(d, valid)
                pix = _pixel_noise(d, valid)
                pix_cur = _pixel_noise(rate, valid) * np.sqrt(2)     # noise of a difference of two such frames
                r.update(dt=dt, sky_rate=sky, ls=ls, pix=pix, diff_ls_n=ls / pix if pix > 0 else np.nan,
                         diff_rel=ls / sky, diff_pois=ls / (pix_cur / SMOOTH_PX),
                         diff_rel_zen=ls_std(d, valid & ZEN) / float(np.median(rate[valid & ZEN])),
                         dmed_rel=float(np.median(d[valid])) / sky)
                del der, d
            rows.append(r)
            prev = (rate, t.unix, sat)
            del img
        pd.DataFrame(rows).to_parquet(os.path.join(OUT, f"diff_{day}.parquet"))
        print(day, len(rows), "frames", f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
