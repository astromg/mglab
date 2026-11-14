"""Telescope zero-point deficits (zero_monitor.lc, relative, 0 = clear) against the allsky: FITS local transparency at
the telescope's pointing (stars within R_LOC deg of it, nearest allsky frame within 150 s), FITS whole-sky transparency,
and the service's (jpg) whole-sky transp. Writes ~/data/allsky/fits/report/tel_match.parquet."""
import os
import numpy as np
import pandas as pd
from fits_analyze import airmass, AMP_MAX, K_EXT

D = os.path.expanduser("~/data/allsky/fits")
R_LOC = 12.0
ZP_CLEAR = 13.86
C = np.load(f"{D}/report/calib.npz")

z = pd.read_parquet(f"{D}/nats_zero_points.parquet")
z = z[z.zero_value != 0].drop_duplicates(["telescope", "oca_jd", "filter"])
r = pd.read_parquet(f"{D}/nats_raw_frames.parquet")
r["oca_jd"] = r.JD.astype(float) - 2460000
r = r[r.ALT_TEL != "None"]
z = z.merge(r[["telescope", "oca_jd", "ALT_TEL", "AZ_TEL", "EXPTIME"]], on=["telescope", "oca_jd"], how="left")
z["t"] = pd.to_datetime((z.oca_jd + 2460000 - 2440587.5) * 86400, unit="s")
z = z.dropna(subset=["ALT_TEL"])
z["alt_t"], z["az_t"] = z.ALT_TEL.astype(float), z.AZ_TEL.astype(float)

F = pd.read_parquet(f"{D}/report/frames_all.parquet")
F = F[F.sun_alt < -18].sort_values("t")
F["t"] = F.t.astype("datetime64[ns]")
z["t"] = z.t.astype("datetime64[ns]")
z = z.sort_values("t")
z = pd.merge_asof(z, F[["t", "frame", "zp_clean", "seen_f", "svc_transp", "svc_state", "svc_diff_ls_n", "night"]],
                  on="t", direction="nearest", tolerance=pd.Timedelta("150s")).dropna(subset=["frame"])

from fits_local import load_stars, local_transp
factors = pd.read_parquet(f"{D}/report/star_factors.parquet")
loc, nloc = [], []
cache = {}
for _, row in z.iterrows():
    day = row.frame[:8]
    if day not in cache:
        cache = {day: load_stars(day)}
    q = cache[day][cache[day].frame == row.frame]
    tr, n = local_transp(q, row.alt_t, row.az_t, R_LOC, factors)
    loc.append(tr); nloc.append(n)
z["fits_loc"], z["n_loc"] = loc, nloc
z["fits_glob"] = z.zp_clean - ZP_CLEAR
z.to_parquet(f"{D}/report/tel_match.parquet")
print(len(z), "matched zero points")
ok = z.dropna(subset=["fits_loc"])
print("spearman with telescope zero_value:", ok[["zero_value", "fits_loc", "fits_glob", "svc_transp"]].corr("spearman").zero_value.round(3).to_dict())
bad = ok[ok.zero_value < -0.2]
print(len(bad), "telescope points with deficit < -0.2 mag")
print(bad[["t", "telescope", "filter", "zero_value", "alt_t", "fits_loc", "n_loc", "fits_glob", "seen_f", "svc_transp", "svc_state", "svc_diff_ls_n"]]
      .round(2).to_string(index=False))
