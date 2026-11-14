"""Same-star differential photometry of zb08 frames from the OFP aperture photometry (`_ap.txt`).

For every field observed in the filter on >= MIN_NIGHTS nights (up to 2 frames per night): the stars of each frame are
matched by RA/Dec to the frame with most stars. Then, alternately, frame zero points (median over stars) and star means
are removed, so the residual of a star on a frame is its deviation from its own mean. For every measurement we store the
flat ratio P = 2.5 log10(flat_z / flat_r) at the star's position (block maps of the night-1345 master flats) and
Pc = P minus the star's mean P. A slope of residual vs Pc says how much of the z-vs-r flat difference is not real for
stars (1 = none of it is real, 0 = all of it is).

Output: ~/data/ghost/<tel>_<filter>_samestar.parquet

    python samestar_phot.py --filter z
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import ascii, fits

OFP = "/data/fits/{tel}/processed-ofp"
INDEX = os.path.expanduser("~/data/ocm_frames/{tel}")
OUT = os.path.expanduser("~/data/ghost")
FLAT_Z = "z/zb08c_1345_46571/zb08c_1345_46571_master_f_z.fits"
FLAT_R = "r/zb08c_1345_47681/zb08c_1345_47681_master_f_r.fits"
BLOCK = 16


def blocks(d, b=BLOCK):
    """Median of the image in b x b px blocks (edges that do not fill a block are dropped)."""
    if d.ndim != 2:
        raise ValueError("image must be 2D")
    ny, nx = d.shape[0] // b, d.shape[1] // b
    return np.nanmedian(d[:ny * b, :nx * b].reshape(ny, b, nx, b).transpose(0, 2, 1, 3).reshape(ny, nx, b * b), axis=2)


def read_ap(path):
    """Columns x, y, ra, dec, mag, err, snr of an OFP `_ap.txt`, or None if missing / old format."""
    if not os.path.exists(path):
        return None
    try:
        t = ascii.read(path, format="fixed_width_two_line").to_pandas()
        return t[["xcenter", "ycenter", "ra", "dec", "mag", "mag_err", "snr"]].rename(columns={"xcenter": "x", "ycenter": "y", "mag_err": "err"})
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tel", default="zb08")
    ap.add_argument("--filter", default="z")
    ap.add_argument("--night-min", type=int, default=1290)
    ap.add_argument("--min-nights", type=int, default=5)
    ap.add_argument("--min-snr", type=float, default=50)
    a = ap.parse_args()
    ofp = OFP.format(tel=a.tel)

    fz = blocks(fits.getdata(f"{ofp}/flats/{FLAT_Z}").astype(np.float64))
    fr = blocks(fits.getdata(f"{ofp}/flats/{FLAT_R}").astype(np.float64))
    P = 2.5 * np.log10((fz / np.median(fz)) / (fr / np.median(fr)))

    d = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{INDEX.format(tel=a.tel)}/*.parquet")])
    d = d[(d.night.astype(int) >= a.night_min) & (d.IMAGETYP == "science") & (d.FILTER == a.filter)].copy()
    d["base"] = d.file.map(lambda f: os.path.basename(f)[:-5])

    rows = []
    for obj, g in d.groupby("OBJECT"):
        g = g.sort_values("JD").groupby("night").head(2)
        if g.night.nunique() < a.min_nights:
            continue
        tabs = [(b, read_ap(f"{ofp}/science/{b.split('_')[1]}/{b}/{b}_ap.txt")) for b in g.base]
        tabs = [(b, t[t.snr > a.min_snr]) for b, t in tabs if t is not None and len(t) > 10]
        if len(tabs) < a.min_nights:
            continue
        ref = max(tabs, key=lambda x: len(x[1]))[1]
        rc = SkyCoord(ref.ra.values * u.deg, ref.dec.values * u.deg)
        for b, t in tabs:
            i, s, _ = match_coordinates_sky(SkyCoord(t.ra.values * u.deg, t.dec.values * u.deg), rc)
            m = s.arcsec < 1.5
            rows.append(pd.DataFrame(dict(obj=obj, frame=b, star=i[m], x=t.x.values[m], y=t.y.values[m], mag=t.mag.values[m], err=t.err.values[m])))
        print(obj, len(tabs), flush=True)

    c = pd.concat(rows)
    c = c[(c.x > 50) & (c.x < 2000) & (c.y > 50) & (c.y < 2000)].copy()
    c["P"] = P[np.clip((c.y / BLOCK).astype(int), 0, P.shape[0] - 1), np.clip((c.x / BLOCK).astype(int), 0, P.shape[1] - 1)]
    c["key"] = c.obj + "_" + c.star.astype(str)
    c = c[c.groupby("key").frame.transform("count") >= a.min_nights].copy()
    for _ in range(3):                      # frame zero points and star means, alternately
        c["zp"] = (c.mag - c.groupby("key").mag.transform("mean")).groupby(c.frame).transform("median")
        c["m0"] = (c.mag - c.zp).groupby(c.key).transform("mean")
    c["res"] = c.mag - c.zp - c.m0
    c["Pc"] = c.P - c.groupby("key").P.transform("mean")
    c = c[np.abs(c.res) < 0.2]
    os.makedirs(OUT, exist_ok=True)
    c.to_parquet(f"{OUT}/{a.tel}_{a.filter}_samestar.parquet")
    print("saved", len(c), "measurements of", c.key.nunique(), "stars in", c.obj.nunique(), "fields")


if __name__ == "__main__":
    main()
