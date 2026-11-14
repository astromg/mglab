"""Linear photometry of BSC stars on the allsky raw FITS, one night at a time.

Per frame (Sun below SUN_MAX): every BSC star V < 6 above alt 15 is measured by aperture photometry at the model
position (fits_lib.CAMERA_PARAMS_FITS); per star rows go to stars_<day>.parquet, per frame summaries to
frames_<day>.parquet in ~/data/allsky/fits/phot/. One frame in memory at a time.

    python fits_photometry.py DAY [DAY ...]
"""
import os, sys, time
import numpy as np
import pandas as pd
from astropy.utils import iers
iers.conf.auto_download = False     # IERS download costs ~20 s per start; precision irrelevant here
iers.conf.auto_max_age = None
from allsky_lib import load_bsc, radec_to_altaz, body_altaz, altaz_to_xy, xy_to_altaz, BSC_CSV
from fits_lib import read_fits, day_fits, aperture_photometry, CAMERA_PARAMS_FITS, FITS_SHAPE, SATURATION

OUT = os.path.expanduser("~/data/allsky/fits/phot")
SUN_MAX = -6.0
P = CAMERA_PARAMS_FITS

yy, xx = np.mgrid[0:FITS_SHAPE[0], 0:FITS_SHAPE[1]]
ALT, AZ = xy_to_altaz(xx.astype(float), yy.astype(float), P)
ZEN = ALT >= 70
RING = (ALT >= 30) & (ALT < 50)
UP30 = ALT >= 30
del yy, xx
PIX_AS2 = (3600.0 / P[3]) ** 2          # pixel area at the zenith [arcsec^2]


def airmass(alt):
    z = np.radians(90 - np.asarray(alt))
    return 1 / (np.cos(z) + 0.50572 * (96.07995 - np.degrees(z)) ** -1.6364)


def main(days):
    os.makedirs(OUT, exist_ok=True)
    ra, dec, v = load_bsc(BSC_CSV)
    sel = v < 6.0
    ra, dec, v, hr = ra[sel], dec[sel], v[sel], np.nonzero(sel)[0]
    for day in days:
        frows, srows = [], []
        t_start = time.time()
        for path in day_fits(day):
            try:
                img, h, t = read_fits(path)
            except Exception as e:
                print("ERR", path, e, file=sys.stderr); continue
            sun = float(body_altaz("sun", t)[0])
            if sun > SUN_MAX:
                continue
            moon_alt, moon_az = (float(q) for q in body_altaz("moon", t))
            exp = float(h["EXPTIME"])
            alt, az = radec_to_altaz(ra, dec, t)
            up = alt > 15
            x, y = altaz_to_xy(alt[up], az[up], P)
            flux, bkg, bsig, peak = aperture_photometry(img, x, y)
            name = os.path.basename(path)
            st = pd.DataFrame(dict(frame=name, idx=hr[up], v=v[up], alt=alt[up], az=az[up], x=x, y=y,
                                   flux=flux / exp, bkg=bkg / exp, bsig=bsig, peak=peak))
            srows.append(st)
            # frame zero point: V 2-5, alt > 30, unsaturated, S/N > 10
            noise = bsig * np.sqrt(np.pi * 9)
            ok = (st.v > 2) & (st.v < 5) & (st.alt > 30) & (st.peak < SATURATION) & (flux > 10 * noise)
            zp_star = st.v[ok] + 2.5 * np.log10(st.flux[ok])
            X = airmass(st.alt[ok])
            zp, zp0, k = np.nan, np.nan, np.nan
            if ok.sum() >= 15:
                zp = float(np.median(zp_star))
                A = np.c_[np.ones(ok.sum()), -X]
                c = np.linalg.lstsq(A, zp_star, rcond=None)[0]
                for _ in range(2):    # clip outliers
                    r = zp_star - A @ c; m = np.abs(r) < 3 * 1.4826 * np.median(np.abs(r - np.median(r)))
                    c = np.linalg.lstsq(A[m], zp_star[m], rcond=None)[0]
                zp0, k = float(c[0]), float(c[1])
            # detection fraction of V<4.5 stars above 30 deg
            e = (st.v < 4.5) & (st.alt > 30)
            seen = float((flux[e.values] > 5 * noise[e.values]).mean()) if e.sum() else np.nan
            sky_zen = float(np.median(img[ZEN])) / exp
            frows.append(dict(frame=name, t=t.isot, sun_alt=sun, moon_alt=moon_alt, exptime=exp, gain=h.get("GAIN"),
                              autoexp=h.get("AUTOEXP"), capmean=h.get("CAPMEAN"), temp=h.get("CCD-TEMP"),
                              sky_zen=sky_zen, sky_ring=float(np.median(img[RING])) / exp,
                              sky_zen_mag=(zp0 - 2.5 * np.log10(sky_zen / PIX_AS2)) if sky_zen > 0 else np.nan,
                              sat_frac=float((img[UP30] >= SATURATION).mean()), p99=float(np.percentile(img[UP30], 99)),
                              n_zp=int(ok.sum()), zp=zp, zp0=zp0, k_ext=k, seen45=seen))
            del img
        pd.DataFrame(frows).to_parquet(os.path.join(OUT, f"frames_{day}.parquet"))
        if srows:
            pd.concat(srows).to_parquet(os.path.join(OUT, f"stars_{day}.parquet"))
        print(day, len(frows), "frames", f"{time.time() - t_start:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
