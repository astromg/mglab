"""Clear-sky calibration of the FITS star photometry: median of zps - ZP_CLEAR per (alt, az) bin over clear frames
(service state clear, FITS seen_f > 0.97, whole-sky |transp| < 0.1, Sun < -20). Saves ~/data/allsky/fits/report/calib.npz
with alt_edges, az_edges, corr (NaN where too few stars). Usage of the map: zps_corr = zps - corr(alt, az)."""
import os
import numpy as np
import pandas as pd
from fits_analyze import airmass, AMP_MAX, K_EXT

D = os.path.expanduser("~/data/allsky/fits")
ZP_CLEAR = 13.86
ALT_E = np.arange(20, 91, 5.0)
AZ_E = np.arange(0, 361, 20.0)

F = pd.read_parquet(f"{D}/report/frames_all.parquet")
good = F[(F.sun_alt < -20) & (F.svc_state == "clear") & (F.seen_f > 0.97) & ((F.zp_clean - ZP_CLEAR).abs() < 0.1)
         & ~F.frame.str.startswith("20261106")]
print(len(good), "clear frames")
acc = []
for day in sorted(good.frame.str[:8].unique()):
    s = pd.read_parquet(f"{D}/phot/stars_{day}.parquet", columns=["frame", "idx", "v", "alt", "az", "flux", "bkg", "bsig", "peak"])
    s = s[s.frame.isin(good.frame)].merge(F[["frame", "exptime"]], on="frame")
    amp = s.peak - s.bkg * s.exptime
    snr = s.flux * s.exptime / (s.bsig * np.sqrt(np.pi * 9))
    s = s[(s.v > 3.5) & (s.v < 5.5) & (s.alt > 20) & (amp < AMP_MAX) & (snr > 10)]
    s = s.assign(d=s.v + 2.5 * np.log10(s.flux) + K_EXT * (airmass(s.alt) - 1) - ZP_CLEAR)
    acc.append(s[["idx", "alt", "az", "d"]])
s = pd.concat(acc)
ia = np.digitize(s.alt, ALT_E) - 1
iz = np.digitize(s.az % 360, AZ_E) - 1
corr = np.full((len(ALT_E) - 1, len(AZ_E) - 1), np.nan)
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        m = (ia == i) & (iz == j)
        if m.sum() >= 30:
            corr[i, j] = np.median(s.d.values[m])
g = s.groupby("idx").d.agg(["median", "count"])
g = g[g["count"] >= 20]
np.savez(f"{D}/report/calib.npz", alt_edges=ALT_E, az_edges=AZ_E, corr=corr, star_idx=g.index.values, star_corr=g["median"].values)
print(len(g), "stars with their own correction; sd of star corrections", round(float(g["median"].std()), 3))
print("rows alt 20..90 step 5, cols az 0..360 step 20 [mag]:")
for i, a in enumerate(ALT_E[:-1]):
    print(f"{a:4.0f} " + " ".join(f"{c:5.2f}" if np.isfinite(c) else "  nan" for c in corr[i]))
# residual extinction: median d per alt bin
print("median by alt:", [round(float(np.nanmedian(corr[i])), 2) for i in range(corr.shape[0])])
