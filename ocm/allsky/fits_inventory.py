"""Inventory of the allsky raw FITS archive (/data/misc/allsky_raw): header values and simple pixel stats per frame.

One frame in memory at a time; output CSV, one row per frame.
    python fits_inventory.py OUT.csv DAY [DAY ...]
"""
import csv, glob, os, sys
from datetime import datetime, timedelta
import numpy as np
from astropy.io import fits
from astropy.time import Time
from allsky_lib import body_altaz

RAW_DIR = "/data/misc/allsky_raw"
KEYS = ["EXPTIME", "AUTOEXP", "GAIN", "AUTOGAIN", "CCD-TEMP", "BINNING", "BITDEPTH", "FLIP", "CAPMEAN", "DARKFRM"]


def exposure_start(h):
    """Exposure start [UT datetime] from CAPDATE/CAPTIME (RPi local clock), offset to UT from DATE-UT (processing)."""
    loc = datetime.strptime(h["CAPDATE"] + h["CAPTIME"].strip(), "%Y%m%d%H%M%S")
    ut_proc = datetime.strptime(h["DATE-UT"], "%Y%m%d_%H%M")
    off = round((ut_proc - loc).total_seconds() / 3600)
    return loc + timedelta(hours=off), off


def main(out, days):
    rows = []
    for day in days:
        for p in sorted(glob.glob(os.path.join(RAW_DIR, day, "*.fits"))):
            try:
                with fits.open(p) as f:
                    h = f[1].header
                    d = f[1].data
                    t0, off = exposure_start(h)
                    c = d[310:710, 560:960]           # central 400x400 around the zenith
                    r = dict(file=os.path.basename(p), ut_start=t0.isoformat(), tz_off=off,
                             proc_lag=(datetime.strptime(h["DATE-UT"], "%Y%m%d_%H%M") - t0).total_seconds(),
                             size=os.path.getsize(p), mtime=os.path.getmtime(p),
                             med_all=float(np.median(d)), med_c=float(np.median(c)),
                             p999_c=float(np.percentile(c, 99.9)), nsat=int((d >= 65535).sum()), dmin=int(d.min()))
                    r.update({k: h.get(k) for k in KEYS})
                    del d, c
                rows.append(r)
            except Exception as e:
                print("ERR", p, e, file=sys.stderr)
    # Sun / Moon at mid exposure, vectorised
    t = Time([r["ut_start"] for r in rows], format="isot", scale="utc") + \
        np.array([float(r["EXPTIME"] or 0) / 2 for r in rows]) / 86400
    for body in ("sun", "moon"):
        alt, az = body_altaz(body, t)
        for r, a in zip(rows, alt):
            r[body + "_alt"] = round(float(a), 2)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(len(rows), "frames ->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
