"""FFS scan of OCM frames - the command-line version of ffs_scan.ipynb.

Runs the full pyaraucaria.ffs pipeline (branch mgorski_dev) on the raw or ofp-reduced (zdf) frames of
the last N nights and appends one row per frame to the parquet database in DB_DIR (<tel>/ for raw,
<tel>_zdf/ for zdf), skipping frames already there. Same rows as the notebook, which reads the
database with load_db().

    python ffs_scan.py --telescopes zb08 wk06 jk15 --sources raw zdf --nights 15
"""
import os
import time
import glob
import argparse
import warnings
import subprocess

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_body

import pyaraucaria
from pyaraucaria.ffs import FFS

FITS_DIR = "/work/vela/oca/fits"
DB_DIR = os.path.expanduser("~/data/ffs_db")

CHUNK = 20                    # frames per parquet file
SATURATION = {"zb08": 60000, "wk06": 60000, "jk15": 150000}  # ADU, raw; SATURATE is empty in the headers, jk15 is BITPIX 32 with star peaks up to ~170k; used also for zdf
AUTOBIN = {"zb08": False, "wk06": False, "jk15": True}       # jk15 4k: ~27 s/frame at bin 1, ~10 s with autobin
LINES_NSIGMA = 5              # threshold mask for line detection; FFS default 3 gives false lines on galaxies and dense fields
N_STARS = 100                 # brightest stars measured
STARS_MASK_K = 3              # second pass: stats and sky without discs of k * FWHM around stars

# header cards stored as columns (the full header is stored as text anyway)
HDR_KEYS = ["DATE-OBS", "JD", "IMAGETYP", "OBSTYPE", "OBJECT", "FILTER", "EXPTIME", "TEST",
            "RA_TEL", "DEC_TEL", "ALT_TEL", "AZ_TEL", "AIRMASS", "FOCUS", "ROTATOR",
            "T-CAM", "T-CAMSET", "READ-MOD", "GAIN-MOD", "GAIN", "RON", "SCALE",
            "DOME-SHT", "DOME-AZ", "MIRR-COV", "TRACKING", "FAN-MIRR", "FAN-DOME",
            "T-WS", "P-WS", "RHUM-WS", "WIND-DIR", "WIND-AVG", "WIND-GUS",
            "M1-POS1", "M1-POS2", "M1-POS3", "NAXIS1", "NAXIS2",
            "OFP_DATE", "OFP_ZERO", "OFP_DARK", "OFP_FLAT"]   # OFP_* only in zdf

# which ffs measured a frame: FFS on mgorski_dev changes, results of different versions may not be comparable
_repo = os.path.dirname(os.path.dirname(pyaraucaria.__file__))
FFS_VERSION = subprocess.run(["git", "-C", _repo, "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()


def current_night():
    """OCM night number (directory name in raw/) of the coming / ongoing night: JD - 2460000 at noon UTC."""
    return int(Time.now().jd) - 2460000


def list_files(telescope, n_nights, source):
    """Raw ("raw") or reduced ("zdf") frames of the last n_nights nights, newest first."""
    root = os.path.join(FITS_DIR, telescope, "raw" if source == "raw" else "processed-ofp/science")
    pattern = "*.fits" if source == "raw" else "*/*_zdf.fits"
    first = current_night() - n_nights + 1
    nights = sorted((d for d in os.listdir(root) if d.isdigit() and int(d) >= first), reverse=True)  # date-named dirs in raw/ are symlinks to these
    files = []
    for n in nights:
        files += sorted(glob.glob(os.path.join(root, n, pattern)), reverse=True)
    return files


def db_name(telescope, source):
    """Directory of the database in DB_DIR: zb08 for raw, zb08_zdf for zdf."""
    return telescope if source == "raw" else f"{telescope}_{source}"


def load_db(telescope, source="raw"):
    """All rows measured for the telescope and source; for a frame measured more than once only the last row."""
    files = sorted(glob.glob(os.path.join(DB_DIR, db_name(telescope, source), "*.parquet")))
    if not files:
        return pd.DataFrame()
    db = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return db.sort_values("measured_at").drop_duplicates("file", keep="last").sort_values("file", ignore_index=True)


def save_chunk(rows, telescope, source):
    """Write rows as a new parquet file in DB_DIR/db_name(telescope, source)."""
    if not rows:
        return
    path = os.path.join(DB_DIR, db_name(telescope, source))
    os.makedirs(path, exist_ok=True)
    name = os.path.join(path, Time.now().strftime("%Y%m%d_%H%M%S") + f"_{len(rows)}.parquet")
    pd.DataFrame(rows).to_parquet(name, index=False)


def hdr_value(hdr, key):
    """Header value; missing or empty ('') -> None."""
    v = hdr.get(key)
    return None if v is None or v == "" else v


def sky_geometry(hdr):
    """Moon and Sun altitude, Moon illumination and Moon distance from the field [deg]."""
    location = EarthLocation(lat=hdr["OBS-LAT"] * u.deg, lon=hdr["OBS-LONG"] * u.deg, height=hdr["OBS-ELEV"] * u.m)
    t = Time(hdr["DATE-OBS"], format="isot", scale="utc")
    altaz = AltAz(obstime=t, location=location)
    field = SkyCoord(ra=hdr["RA_TEL"] * u.deg, dec=hdr["DEC_TEL"] * u.deg, frame="icrs")
    sun = get_sun(t)
    moon = get_body("moon", t, location)
    elongation = moon.separation(sun, origin_mismatch="ignore").rad
    return {
        "sun_alt": sun.transform_to(altaz).alt.deg,
        "moon_alt": moon.transform_to(altaz).alt.deg,
        "moon_illum": (1 - np.cos(elongation)) / 2,
        "moon_sep": field.separation(moon, origin_mismatch="ignore").deg,
    }


def measure(path, telescope, imagetypes=None):
    """One row of the database: file, header, FFS frame statistics, lines, stars, Moon/Sun.
    None if the frame type is not in imagetypes."""
    row = {"file": os.path.basename(path), "path": path, "telescope": telescope,
           "night": int(os.path.basename(path).split("_")[1]),
           "measured_at": Time.now().isot, "ffs_version": FFS_VERSION, "error": None}
    t0 = time.time()
    try:
        with fits.open(path) as hl:
            hdr = hl[0].header
            data = hl[0].data.astype(float)
        row["header"] = hdr.tostring(sep="\n")
        row.update({key: hdr_value(hdr, key) for key in HDR_KEYS})
        if imagetypes is not None and hdr.get("IMAGETYP") not in imagetypes:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # nanmedian of empty slices on frames without stars
            ffs = FFS(data, gain=hdr_value(hdr, "GAIN") or 1., rn_noise=hdr_value(hdr, "RON") or 0., saturation=SATURATION[telescope], autobin=AUTOBIN[telescope])
            # the steps of calc_frame_fwhm, with our own nsigma of the threshold mask
            ffs.mk_saturation_mask()
            ffs.mk_stats()
            ffs.sky_map()
            ffs.mk_cosmics_mask()
            ffs.mk_threshold_mask(nsigma=LINES_NSIGMA)
            ffs.find_lines()
            ffs.classify_lines()
            ffs.find_stars(threshold=10, fwhm=10)
            ffs.star_info(box=10, N_stars=N_STARS)
            ffs.calc_star_stats()
            if np.isfinite(ffs.frame_fwhm):
                # second pass: frame statistics and sky without the stars
                ffs.mk_stars_mask(k=STARS_MASK_K)
                ffs.mk_stats()
                ffs.sky_gradient()
            if ffs.frame_used_stars >= 5:
                ffs.calc_star_field_stats()

        for key, v in ffs.stats["frame"].items():
            row[key] = v.tolist() if isinstance(v, np.ndarray) else v
        lines = ffs.lines
        trails = lines[~lines["spike"]] if len(lines) else lines
        row["bin"] = ffs.bin
        row["n_stars"] = len(ffs.coo)
        row["n_lines"] = len(trails)
        row["n_spikes"] = len(lines) - len(trails)
        row["line_max_len"] = float(np.max(np.hypot(trails["x1"] - trails["x0"], trails["y1"] - trails["y0"]))) if len(trails) else 0.
        row["line_max_val"] = float(np.max(trails["val"])) if len(trails) else 0.
        row["fwhm_arcsec"] = row["fwhm"] * row["SCALE"] if row.get("SCALE") else None
        if hdr_value(hdr, "RA_TEL") is not None:
            row.update(sky_geometry(hdr))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    row["ffs_time"] = time.time() - t0
    return row


COLS = f"{'file':26s} {'type':8s} {'object':16s} {'flt':4s} {'exp':>6s} {'sky':>7s} {'grad':>7s} {'lines':>5s} {'stars':>5s} {'fwhm':>5s} {'fw\"':>5s} {'ell':>5s} {'e_t':>6s} {'moon':>5s} {'t[s]':>5s}"


def fmt(x, f):
    """Number formatted with f, or '-' when missing / NaN."""
    return format("-", ">" + f.split(".")[0]) if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, f)


def show(r):
    """One line with the most interesting values of a database row."""
    if r["error"]:
        return f"{r['file']:26s} ERROR {r['error']}"
    return (f"{r['file']:26s} {str(r['IMAGETYP'])[:8]:8s} {str(r['OBJECT'])[:16]:16s} {str(r['FILTER'])[:4]:4s} {fmt(r['EXPTIME'], '6.1f')} "
            f"{fmt(r.get('median'), '7.0f')} {fmt(r.get('bkg_frame_gradient'), '7.1f')} {r['n_lines']:5d} {r['n_stars']:5d} "
            f"{fmt(r.get('fwhm'), '5.2f')} {fmt(r.get('fwhm_arcsec'), '5.2f')} {fmt(r.get('ellipticity'), '5.2f')} {fmt(r.get('psf_e_t'), '6.3f')} "
            f"{fmt(r.get('moon_alt'), '5.0f')} {r['ffs_time']:5.1f}")


def scan(telescope, source, n_nights, imagetypes=None, max_files=None, retry_errors=False):
    """Measure the frames of the last n_nights nights that are not in the database yet, newest first."""
    db = load_db(telescope, source)
    done = set(db["file"]) if len(db) else set()
    if retry_errors and len(db):
        done -= set(db.loc[db["error"].notna(), "file"])

    files = list_files(telescope, n_nights, source)
    todo = [f for f in files if os.path.basename(f) not in done]
    print(f"{telescope} {source}: {len(files)} frames in the last {n_nights} nights, {len(files) - len(todo)} already in the database, {len(todo)} to measure", flush=True)

    rows = []
    n = 0
    t0 = time.time()
    print(COLS, flush=True)
    try:
        for f in todo:
            if max_files is not None and n >= max_files:
                break
            r = measure(f, telescope, imagetypes)
            if r is None:
                continue
            print(show(r), flush=True)
            rows.append(r)
            n += 1
            if len(rows) >= CHUNK:
                save_chunk(rows, telescope, source)
                rows = []
    finally:
        save_chunk(rows, telescope, source)
    print(f"{telescope} {source}: {n} frames measured in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFS scan of OCM frames into the parquet database (see ffs_scan.ipynb).")
    parser.add_argument("--telescopes", nargs="+", default=["zb08", "wk06", "jk15"])
    parser.add_argument("--sources", nargs="+", default=["raw", "zdf"], choices=["raw", "zdf"])
    parser.add_argument("--nights", type=int, default=15, help="how many nights back from today (1 = last night only)")
    parser.add_argument("--imagetypes", nargs="+", default=None, help="only these IMAGETYP values (default: all)")
    parser.add_argument("--max-files", type=int, default=None, help="stop after this many measured frames per telescope and source")
    parser.add_argument("--retry-errors", action="store_true", help="measure again frames that failed last time")
    parser.add_argument("--db-dir", default=DB_DIR)
    args = parser.parse_args()

    DB_DIR = args.db_dir
    print("pyaraucaria", FFS_VERSION, pyaraucaria.__file__, "->", DB_DIR, flush=True)
    for telescope in args.telescopes:
        for source in args.sources:
            scan(telescope, source, args.nights, args.imagetypes, args.max_files, args.retry_errors)
