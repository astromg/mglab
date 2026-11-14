"""Block-median sky maps of zb08 frames for the ghost study.

For every selected science frame: sky before the flat (raw - master zero), the reduced frame (_zdf) and the master flat
used, each as a median in BLOCK x BLOCK px blocks. For every master flat of the filter: the flat map. Plus header values
(rotator, exposure, altitude, Sun, Moon). Frames are read one at a time; only the small maps are kept.

Output: ~/data/ghost/<tel>_<filter>_maps.npz (maps) and ~/data/ghost/<tel>_<filter>_frames.parquet (one row per map).

    python sky_maps.py --filter z --night-min 1290 --per-night 3
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from astropy.io import fits

OFP = "/data/fits/{tel}/processed-ofp"
INDEX = os.path.expanduser("~/data/ocm_frames/{tel}")
OUT = os.path.expanduser("~/data/ghost")
BLOCK = 16
CARDS = ["OBJECT", "FILTER", "EXPTIME", "ROTATOR", "ROT-MECH", "ALT_TEL", "AZ_TEL", "SUN-ALT", "MOON-ALT", "MOON-PHA", "JD"]


def blocks(d, b=BLOCK):
    """Median of the image in b x b px blocks.

    Parameters
    ----------
    d : 2D array
        Image; must be square-ish, the edges that do not fill a whole block are dropped.
    b : int
        Block size in pixels.

    Returns
    -------
    2D array of shape (ny // b, nx // b).
    """
    if d.ndim != 2:
        raise ValueError("image must be 2D")
    ny, nx = d.shape[0] // b, d.shape[1] // b
    return np.nanmedian(d[:ny * b, :nx * b].reshape(ny, b, nx, b).transpose(0, 2, 1, 3).reshape(ny, nx, b * b), axis=2)


def header_values(h):
    """The CARDS of a header as a dict (None where missing)."""
    return {k: h.get(k) for k in CARDS}


def science_list(tel, flt, night_min, per_night, min_exptime):
    """Science frames of one filter from the frames index: up to per_night different objects per night."""
    files = sorted(glob.glob(f"{INDEX.format(tel=tel)}/*.parquet"))
    if not files:
        raise ValueError(f"no frames index for {tel}")
    d = pd.concat([pd.read_parquet(f) for f in files])
    d = d[(d.night.astype(int) >= night_min) & (d.IMAGETYP == "science") & (d.FILTER == flt) & (d.EXPTIME >= min_exptime)]
    d = d.sort_values("JD").groupby(["OBJECT", "night"]).head(1)
    return d.groupby("night").head(per_night).file.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tel", default="zb08")
    ap.add_argument("--filter", default="z")
    ap.add_argument("--night-min", type=int, default=1290)
    ap.add_argument("--per-night", type=int, default=3)
    ap.add_argument("--min-exptime", type=float, default=30)
    a = ap.parse_args()
    ofp = OFP.format(tel=a.tel)
    os.makedirs(OUT, exist_ok=True)

    rows, raw_maps, zdf_maps, flat_maps = [], [], [], []

    # master flats of the filter
    for d in sorted(glob.glob(f"{ofp}/flats/{a.filter}/{a.tel}c_*")):
        base = os.path.basename(d)
        if int(base.split("_")[1]) < a.night_min:
            continue
        master = glob.glob(f"{d}/*_master_f_{a.filter}.fits")
        raws = sorted(glob.glob(f"{d}/{a.tel}c_*[0-9].fits"))
        if not master or not raws:
            continue
        m = blocks(fits.getdata(master[0]).astype(np.float64))
        row = header_values(fits.getheader(raws[len(raws) // 2]))
        row.update(kind="flat", frame=base, flat=base, sky=np.nan)
        rows.append(row)
        raw_maps.append(m / np.median(m))      # a master flat is already zero-subtracted
        zdf_maps.append(np.full_like(m, np.nan))
        flat_maps.append(m / np.median(m))
        print("flat", base, row["ROT-MECH"], flush=True)

    # night sky on science frames
    for f in science_list(a.tel, a.filter, a.night_min, a.per_night, a.min_exptime):
        base = os.path.basename(f)[:-5]
        d = f"{ofp}/science/{base.split('_')[1]}/{base}"
        zero, flat = glob.glob(f"{d}/*_master_z.fits"), glob.glob(f"{d}/*_master_f_*.fits")
        if not (os.path.exists(f"{d}/{base}_zdf.fits") and zero and flat):
            continue
        raw = fits.getdata(f"{d}/{base}.fits").astype(np.float64)
        rm = blocks(raw - fits.getdata(zero[0]).astype(np.float64))
        h = fits.getheader(f"{d}/{base}.fits")
        del raw
        zm = blocks(fits.getdata(f"{d}/{base}_zdf.fits").astype(np.float64))
        fm = blocks(fits.getdata(flat[0]).astype(np.float64))
        row = header_values(h)
        row.update(kind="science", frame=base, flat=os.path.basename(flat[0]).split("_master")[0], sky=float(np.median(rm)))
        rows.append(row)
        raw_maps.append(rm / np.median(rm))
        zdf_maps.append(zm / np.median(zm))
        flat_maps.append(fm / np.median(fm))
        print("science", base, row["OBJECT"], row["ROT-MECH"], f"{row['sky']:.0f}", flush=True)

    name = f"{OUT}/{a.tel}_{a.filter}"
    np.savez_compressed(f"{name}_maps.npz", raw=np.array(raw_maps, np.float32), zdf=np.array(zdf_maps, np.float32), flat=np.array(flat_maps, np.float32), block=BLOCK)
    pd.DataFrame(rows).to_parquet(f"{name}_frames.parquet")
    print("saved", name, len(rows))


if __name__ == "__main__":
    main()
