"""Local sky transmission from the FITS star photometry, noise-robust: T = sum(flux) / sum(expected flux) over the stars
(V 4.0-5.5, alt > 20) within a radius of a direction; expected flux = 10^(-0.4 (V - ZP_CLEAR)) x the star's own clear-sky
factor (median flux / nominal over clear frames; stars without one are left out). transp = 2.5 log10 T (negative = light lost).

    build_star_factors()  -> ~/data/allsky/fits/report/star_factors.parquet
    local_transp(stars_of_frame, alt, az, radius) -> (transp, n_stars)
"""
import os
import numpy as np
import pandas as pd
from fits_analyze import airmass, K_EXT

D = os.path.expanduser("~/data/allsky/fits")
ZP_CLEAR = 13.86
V_LO, V_HI = 4.0, 5.5


def nominal(v, alt):
    """Clear-sky flux rate [ADU/s] of a star of V at alt, before its own factor."""
    return 10 ** (-0.4 * (v - ZP_CLEAR + K_EXT * (airmass(alt) - 1)))


def load_stars(day, frames=None):
    s = pd.read_parquet(f"{D}/phot/stars_{day}.parquet", columns=["frame", "idx", "v", "alt", "az", "flux"])
    s = s[(s.v >= V_LO) & (s.v < V_HI) & (s.alt > 20)]
    if frames is not None:
        s = s[s.frame.isin(frames)]
    return s.assign(nom=nominal(s.v, s.alt))


def build_star_factors():
    F = pd.read_parquet(f"{D}/report/frames_all.parquet")
    good = F[(F.sun_alt < -20) & (F.svc_state == "clear") & (F.seen_f > 0.97) & ((F.zp_clean - ZP_CLEAR).abs() < 0.1)
             & ~F.frame.str.startswith("20261106")]
    acc = [load_stars(d, set(good.frame))[["idx", "flux", "nom", "alt"]] for d in sorted(good.frame.str[:8].unique())]
    s = pd.concat(acc)
    s["r"] = s.flux / s.nom
    g = s.groupby("idx").r.agg(["median", "count"])
    g = g[g["count"] >= 20].rename(columns={"median": "factor"})
    g.to_parquet(f"{D}/report/star_factors.parquet")
    print(len(g), "stars; factor p5/p50/p95", g.factor.quantile([.05, .5, .95]).round(2).tolist())
    return g


def local_transp(q, alt, az, radius, factors):
    """q: star rows of one frame (load_stars); returns (transp [mag], n stars, expected flux sum)."""
    q = q[q.idx.isin(factors.index)]
    a1, z1, a2, z2 = map(np.radians, (q.alt.values, q.az.values, alt, az))
    sep = np.degrees(np.arccos(np.clip(np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(z1 - z2), -1, 1)))
    w = q[sep < radius]
    if len(w) < 3:
        return np.nan, len(w)
    exp = (w.nom * factors.factor.reindex(w.idx).values).sum()
    T = w.flux.sum() / exp
    return float(2.5 * np.log10(max(T, 0.05))), len(w)


if __name__ == "__main__":
    build_star_factors()
