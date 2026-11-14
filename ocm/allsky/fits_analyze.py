"""Frame-level indicators from the FITS photometry tables (fits_photometry.py, fits_diff.py), joined with the service
messages from NATS (fits_fetch_nats.py) by frame name. Writes ~/data/allsky/fits/report/frames_all.parquet.

zp_clean: median over stars V 3.5-5.5, alt > 30, amplitude < AMP_MAX (no raw saturation), S/N > 10, of
V + 2.5 log10(flux rate) + K_EXT (X - 1). seen_f: share of the stars V < 4.5 above 30 deg detected at S/N > 5.
"""
import glob, os
import numpy as np
import pandas as pd

PH = os.path.expanduser("~/data/allsky/fits/phot")
OUT = os.path.expanduser("~/data/allsky/fits/report")
AMP_MAX = 30000.0
K_EXT = 0.20


def airmass(alt):
    z = np.radians(90 - np.asarray(alt))
    return 1 / (np.cos(z) + 0.50572 * (96.07995 - np.degrees(z)) ** -1.6364)


def frame_table():
    rows = []
    for fp in sorted(glob.glob(os.path.join(PH, "frames_*.parquet"))):
        day = fp[-16:-8]
        f = pd.read_parquet(fp)
        sp = os.path.join(PH, f"stars_{day}.parquet")
        if not os.path.exists(sp) or f.empty:
            continue
        s = pd.read_parquet(sp, columns=["frame", "v", "alt", "flux", "bkg", "bsig", "peak"])
        s = s.merge(f[["frame", "exptime"]], on="frame")
        noise = s.bsig * np.sqrt(np.pi * 9)
        amp = s.peak - s.bkg * s.exptime
        snr = s.flux * s.exptime / noise
        s["zps"] = s.v + 2.5 * np.log10(s.flux.where(s.flux > 0)) + K_EXT * (airmass(s.alt) - 1)
        clean = (s.v > 3.5) & (s.v < 5.5) & (s.alt > 30) & (amp < AMP_MAX) & (snr > 10)
        z = s[clean].groupby("frame").zps.agg(["median", "count"]).rename(columns={"median": "zp_clean", "count": "n_clean"})
        exp = (s.v < 4.5) & (s.alt > 30)
        seen = (snr[exp] > 5).groupby(s.frame[exp]).mean().rename("seen_f")
        zen = (s.v < 4.5) & (s.alt > 60)
        seen_z = (snr[zen] > 5).groupby(s.frame[zen]).mean().rename("seen_zen")
        f = f.merge(z, left_on="frame", right_index=True, how="left").merge(seen, left_on="frame", right_index=True, how="left") \
             .merge(seen_z, left_on="frame", right_index=True, how="left")
        dp = os.path.join(PH, f"diff_{day}.parquet")
        if os.path.exists(dp):
            d = pd.read_parquet(dp).drop(columns=["t", "exptime"])
            f = f.merge(d, on="frame", how="left")
        rows.append(f)
        del s
    f = pd.concat(rows, ignore_index=True)
    f["t"] = pd.to_datetime(f.t)
    f["stem"] = f.frame.str[:-5]
    return f


def with_service(f):
    c = pd.read_parquet(os.path.expanduser("~/data/allsky/fits/nats_clouds.parquet"))
    c["stem"] = c.frame.str[:-4]
    c = c.drop_duplicates("stem")[["stem", "state", "veil", "transp", "transp_s", "star_seen", "diff_ls_n", "zp_frame", "zp_ref",
                                  "marks.red", "marks.yellow"]]
    c.columns = ["stem"] + ["svc_" + k for k in c.columns[1:]]
    return f.merge(c, on="stem", how="left")


if __name__ == "__main__":
    f = with_service(frame_table())
    f["night"] = (f.t - pd.Timedelta("12h")).dt.strftime("%m-%d")
    f.to_parquet(os.path.join(OUT, "frames_all.parquet"))
    print(len(f), "frames;", f.svc_state.notna().sum(), "with a service message")
