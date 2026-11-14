"""Allsky raw FITS (RAW16, /data/misc/allsky_raw, from 2026-11-06 06:59 UT): reading, camera model, linear photometry.

The FITS hold the camera image as captured (1520 x 1021 uint16, 14-bit ADC scaled to 16 bit, vertical flip done by
the capture program); the jpg archive is a crop [10:990, 290:1270] of the same image, rotated by -39.1 deg and
re-rendered by matplotlib (per-frame min-max stretch), so the two have different pixel geometry.
"""
import os
from datetime import datetime, timedelta

import numpy as np
from astropy.io import fits
from astropy.time import Time

RAW_DIR = "/data/misc/allsky_raw"
FITS_SHAPE = (1021, 1520)
SATURATION = 65000.0
# (x0, y0, theta, k1, k2) as in allsky_lib.altaz_to_xy; fitted on stars by fits_camera.py
CAMERA_PARAMS_FITS = (773.32, 500.51, -39.077, 5.5165, -3.775e-05)   # 2026-11-12, 815 stars V<4 on 4 frames, rms 1.3 px above alt 30


def exposure_start(h):
    """Exposure start (UT datetime) from CAPDATE/CAPTIME (RPi local clock); the local-UT offset from DATE-UT."""
    loc = datetime.strptime(h["CAPDATE"] + h["CAPTIME"].strip(), "%Y%m%d%H%M%S")
    ut_proc = datetime.strptime(h["DATE-UT"], "%Y%m%d_%H%M")
    off = round((ut_proc - loc).total_seconds() / 3600)
    return loc + timedelta(hours=off)


def read_fits(path):
    """Frame as float32, its header, and the time of mid exposure (astropy Time, scalar).

    Returns
    -------
    img : numpy.ndarray, float32, shape FITS_SHAPE
    hdr : astropy.io.fits.Header
    t : astropy.time.Time
    """
    with fits.open(path) as f:
        hdr = f[1].header.copy()
        img = f[1].data.astype(np.float32)
    t0 = exposure_start(hdr)
    tmid = t0 + timedelta(seconds=float(hdr["EXPTIME"]) / 2)
    return img, hdr, Time(tmid.isoformat(), format="isot", scale="utc")


def day_fits(day, raw_dir=RAW_DIR):
    d = os.path.join(raw_dir, day)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".fits")) if os.path.isdir(d) else []


def aperture_photometry(img, x, y, r_ap=3.0, r_in=6.0, r_out=10.0):
    """Aperture flux minus the annulus median, at given positions.

    Returns
    -------
    flux, bkg, bkg_sigma, peak : numpy.ndarray
        flux [ADU, sum over the aperture minus bkg x area], bkg [ADU/px, annulus median], bkg_sigma [ADU/px, MAD],
        peak [ADU, max in the aperture; >= SATURATION means saturated]. NaN near the edge.
    """
    x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
    R = int(np.ceil(r_out))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    out = np.full((4, len(x)), np.nan)
    for n, (xi, yi) in enumerate(zip(x, y)):
        if not np.isfinite(xi) or not np.isfinite(yi):
            continue
        cx, cy = int(round(xi)), int(round(yi))
        if not (R <= cx < img.shape[1] - R and R <= cy < img.shape[0] - R):
            continue
        win = img[cy - R:cy + R + 1, cx - R:cx + R + 1]
        rr = np.hypot(xx - (xi - cx), yy - (yi - cy))
        ap = rr <= r_ap
        ann = win[(rr >= r_in) & (rr <= r_out)]
        b = np.median(ann)
        out[0, n] = win[ap].sum() - b * ap.sum()
        out[1, n] = b
        out[2, n] = 1.4826 * np.median(np.abs(ann - b))
        out[3, n] = win[ap].max()
    return out
