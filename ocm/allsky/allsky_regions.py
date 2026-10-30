"""Study of the cloud regions (the method itself is in allsky_lib: star_maps, diff_map, regions, detections): whole nights
into tables and maps on disk, drawing of the marks, the cases of the report, and the check against the telescopes.

Drawing (user, 2026-10-29): on the frame, as for the observers, red outline = cloud that hides everything, light yellow
fill = cirrus, semi-transparent or uncertain cloud (or a veil over the whole sky), light cyan area = the Moon mask (no
detection inside); draw_diff and draw_stars show each method alone on its raw data with its own Moon mask.

    python allsky_regions.py cases <out.png>
    python allsky_regions.py nights 20261001 20261005 ...    # -> ~/data/allsky/regions/<night>.parquet + .npz
    python allsky_regions.py plot 20261001 ...               # -> ~/data/allsky/regions/<night>.png
    python allsky_regions.py telescopes 20261001 ...         # -> ~/data/allsky/regions/telescopes.parquet
"""

import gc
import os
import sys

import numpy as np
import pandas as pd
from allsky_lib import *

OUT_DIR = os.path.expanduser("~/data/allsky/regions")
CASES = [("20261017", "0300", "clear"), ("20261007", "0257", "thin veil"), ("20261027", "0613", "moving cloud, Moon"),
         ("20260928", "0539", "altocumulus"), ("20261029", "0127", "clear, Milky Way NW"), ("20261029", "0602", "clear, bright Moon"),
         ("20261005", "0155", "cloud forming in S")]


class RegionAnalyzer:
    """SkyAnalyzer + StateTracker frame by frame, with the veil added to the detections and one table row per frame."""

    def __init__(self, calib_path=CALIB_NPZ):
        self.an = SkyAnalyzer(stars=load_bsc(BSC_CSV), calib=load_calibration(calib_path))
        self.tracker = StateTracker()
        self.sky, self.g = self.an.sky, self.an.g

    def frame(self, path, prev_path):
        """One frame (prev_path None = no difference).

        Returns
        -------
        dict
            row (indicators, state, veil, region areas), img, stars, maps, zs, diff (the derotated difference or None),
            noise (its pixel noise), det (detections with the veil), moon_stars, moon_diff.
        """
        img, t = read_frame(path), frame_time(path)
        prev = None if prev_path is None else read_frame(prev_path)
        dt = None if prev_path is None else (frame_mtime(path) - frame_mtime(prev_path)).sec
        r = self.an.analyse(img, t, prev=prev, dt=dt, details=True)
        s = self.tracker.update(t, r["transp"], r["diff_ls_n"], r["sun_alt"], r["moon_x"], r["star_seen"])
        det = detections(r["maps"], r["zs"], self.g, veil=s["veil"])
        a = det_areas(det, self.g)
        row = dict(frame=os.path.basename(path), mtime=frame_mtime(path).unix, sun_alt=r["sun_alt"], moon_alt=r["moon_alt"], moon_x=r["moon_x"], transp=r["transp"],
                   star_seen=r["star_seen"], zen_star_seen=r["zen_star_seen"], zp_stars=r["zp_stars"], zp_ref=r["zp_ref"], diff_ls_n=r["diff_ls_n"], **s,
                   n_exp=r["n_exp"], sigma=r["sigma"], star_known=np.isfinite(r["maps"]["missing"][self.g]).mean(),
                   star_area=a["star_possible"], star_sure=a["star_sure"], star_cirrus=a["star_cirrus"],
                   diff_area=a["diff_possible"], diff_sure=a["diff_sure"], possible=a["possible"], sure=a["sure"], diff_noise=r["diff_noise"])
        return dict(row=row, img=img, stars=r["stars"], maps=r["maps"], zs=r["zs"], diff=r["diff_image"], noise=r["diff_noise"], det=det,
                    moon_stars=r["moon_stars"], moon_diff=r["moon_diff"])


def night_veil(d):
    """The veil of every frame of a computed night (its table), as StateTracker decides it: the smoothed transp below the
    thin limit and star_seen (median of 3 frames, as the tracker smooths it) at least VEIL_SEEN."""
    if "veil" in d:
        return d.veil.values.astype(bool)
    seen = d.star_seen.rolling(3, min_periods=2).median()
    with np.errstate(invalid="ignore"):
        return ((d.transp_s < STATE_LIMITS["thin_transp"] + STATE_LIMITS["thin_transp_moon"] * d.moon_x) & (seen >= VEIL_SEEN)).values


def pack(maps, zs):
    """Maps packed for storage: missing and missing_sure * 250 as uint8 (255 = unknown), dim * 80 clipped to +-1.5 as
    int8 (-128 = unknown), zs * 12 clipped to +-120 as int8."""
    share = lambda m: np.where(np.isfinite(m), np.round(np.nan_to_num(m) * 250), 255).astype(np.uint8)
    dim = np.where(np.isfinite(maps["dim"]), np.round(np.clip(np.nan_to_num(maps["dim"]), -1.5, 1.5) * 80), -128).astype(np.int8)
    return dict(missing=share(maps["missing"]), missing_sure=share(maps["missing_sure"]), dim=dim, zs=np.round(np.clip(zs, -10, 10) * 12).astype(np.int8))


def load_night(day):
    """(table, star maps as a dict of (frames, rows, cols) arrays with NaN = unknown, signed difference maps zs) of a
    computed night."""
    d = pd.read_parquet(f"{OUT_DIR}/{day}.parquet")
    m = np.load(f"{OUT_DIR}/{day}.npz")
    share = lambda a: np.where(a == 255, np.nan, a / 250)
    maps = dict(missing=share(m["missing"]), missing_sure=share(m["missing_sure"]), dim=np.where(m["dim"] == -128, np.nan, m["dim"] / 80))
    return d, maps, m["zs"] / 12


def night_maps(maps, i):
    """The star maps of frame i of load_night."""
    return {k: v[i] for k, v in maps.items()}


def run_night(ra, day):
    """All frames of `day` with the Sun below -12 deg -> OUT_DIR/<day>.parquet and .npz (the two maps)."""
    paths = day_frames(day)
    dark = set(dark_frames(day, sun_max=-12.0))
    rows, packed = [], []
    ra.tracker = StateTracker()
    for k, p in enumerate(paths):
        if p not in dark or k == 0:
            continue
        out = ra.frame(p, paths[k - 1])
        rows.append(out["row"]); packed.append(pack(out["maps"], out["zs"]))
        if len(rows) % 20 == 0:
            print(day, out["row"]["frame"], out["row"]["state"], f"possible {out['row']['possible']:.2f} sure {out['row']['sure']:.2f}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_parquet(f"{OUT_DIR}/{day}.parquet")
    np.savez_compressed(f"{OUT_DIR}/{day}.npz", **{k: np.array([p[k] for p in packed]) for k in packed[0]})


def _frame_axes(ax, title):
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(150, 1080); ax.set_ylim(1090, 160)


def _grid_xy(shape):
    gy, gx = np.mgrid[0:shape[0], 0:shape[1]] * GRID
    return gx, gy


def _moon_area(ax, moon, sky):
    """Light cyan area of a Moon mask (moon_masks) within the sky; nothing when moon is None or empty."""
    if moon is None:
        return
    m = (moon & sky)[::GRID, ::GRID]
    if m.any():
        gx, gy = _grid_xy(m.shape)
        ax.contourf(gx, gy, m.astype(float), levels=[0.5, 1.5], colors="cyan", alpha=0.3)


def draw(ax, img, det, sky, title, moon=None):
    """The frame with the combined detections, as for the observers: light yellow fill = cirrus, semi-transparent or
    uncertain cloud (possible, either method), red outline = cloud that hides everything (sure, either method), cyan
    area = the Moon mask `moon` (no detection inside; pass moon_masks()[1])."""
    gx, gy = _grid_xy(det["sure"].shape)
    ax.imshow(np.where(sky, img, np.nan), cmap="gray", vmin=np.percentile(img[sky], 1), vmax=np.percentile(img[sky], 99.5))
    if det["possible"].any():
        ax.contourf(gx, gy, det["possible"].astype(float), levels=[0.5, 1.5], colors="yellow", alpha=0.25)
    if det["sure"].any():
        ax.contour(gx, gy, det["sure"].astype(float), levels=[0.5], colors="red", linewidths=1.5)
    _moon_area(ax, moon, sky)
    _frame_axes(ax, title)


def draw_diff(ax, diff, noise, det, sky, title="difference", moon=None):
    """The raw derotated difference (no smoothing, in colour as allsky_lib.render_diff: red brighter than the previous
    frame, blue darker, +-6 pixel noises) with the difference detections: yellow outline = possible, black = sure;
    light cyan area = the Moon mask of the difference (moon_masks()[1])."""
    gx, gy = _grid_xy(det["sure"].shape)
    if diff is not None:
        ax.imshow(np.where(sky, diff, np.nan), cmap="bwr", vmin=-6 * noise, vmax=6 * noise)
    for key, color in (("diff_possible", "gold"), ("diff_sure", "black")):
        if det[key].any():
            ax.contour(gx, gy, det[key].astype(float), levels=[0.5], colors=color, linewidths=1.3)
    _moon_area(ax, moon, sky)
    _frame_axes(ax, title)


def draw_stars(ax, st, det, sky, title="stars", moon=None):
    """The expected stars (green seen, red missing; the sure ones larger) with the star-map detections: yellow outline =
    possible (cirrus included), red = sure; light cyan area = the Moon mask of the star map (moon_masks()[0])."""
    gx, gy = _grid_xy(det["sure"].shape)
    ax.imshow(np.where(sky, 0.15, np.nan), cmap="gray", vmin=0, vmax=1)
    for sel, size in ((st["expected_f"] & ~st["sure_f"], 4), (st["sure_f"], 11)):
        ax.scatter(st["x"][sel & st["seen"]], st["y"][sel & st["seen"]], s=size, c="lime", lw=0)
        ax.scatter(st["x"][sel & ~st["seen"]], st["y"][sel & ~st["seen"]], s=size + 2, c="red", lw=0)
    for key, color in (("star_possible", "yellow"), ("star_sure", "red")):
        if det[key].any():
            ax.contour(gx, gy, det[key].astype(float), levels=[0.5], colors=color, linewidths=1.3)
    _moon_area(ax, moon, sky)
    _frame_axes(ax, title)


def fig_cases():
    """Figure of the CASES frames: combined detections, the difference, the stars."""
    import matplotlib.pyplot as plt
    ra = RegionAnalyzer()
    fig, axs = plt.subplots(3, len(CASES), figsize=(4.2 * len(CASES), 12.5))
    for j, (day, hhmm, title) in enumerate(CASES):
        paths = day_frames(day)
        k = frame_index(paths, hhmm)
        ra.tracker = StateTracker()
        for i in (k - 2, k - 1):   # the state (and the veil) is smoothed over 3 frames, as in a run
            ra.frame(paths[i], paths[i - 1])
        o = ra.frame(paths[k], paths[k - 1])
        row = o["row"]
        print(f"{day} {hhmm} {title}: expected {row['n_exp']}, sigma {row['sigma']:.1f} deg; possible {row['possible']:.0%}, sure {row['sure']:.0%}")
        draw(axs[0, j], o["img"], o["det"], ra.sky, f"{day} {hhmm} {title}", o["moon_diff"])
        draw_diff(axs[1, j], o["diff"], o["noise"], o["det"], ra.sky, moon=o["moon_diff"])
        draw_stars(axs[2, j], o["stars"], o["det"], ra.sky, f"stars (map sigma {row['sigma']:.1f} deg, {row['n_exp']} expected)", o["moon_stars"])
    fig.tight_layout()
    return fig


def fig_night(day):
    """Figure of a computed night: time series of the detection areas against the state, and frames at chosen moments
    with the combined detections."""
    import matplotlib.pyplot as plt
    d, maps, zs = load_night(day)
    sky = sky_mask(FRAME_SHAPE, CAMERA_PARAMS, 30.0)
    g = sky[::GRID, ::GRID]
    veil = night_veil(d)
    dets = [detections(night_maps(maps, i), zs[i], g, veil=veil[i]) for i in range(len(d))]
    area = pd.DataFrame([det_areas(x, g) for x in dets])
    t = pd.to_datetime(d.frame.str[:13], format="%Y%m%d_%H%M")
    colors = dict(clear="#4caf50", thin="#ffc107", cloudy="#ff7043", overcast="#9e9e9e")
    picks = np.unique(np.linspace(0, len(d) - 1, 8).round().astype(int))
    fig = plt.figure(figsize=(17, 11))
    ax = fig.add_axes([0.05, 0.62, 0.9, 0.33])
    for key, color, ls, label in (("star_possible", "tab:blue", "--", "stars possible"), ("star_sure", "tab:blue", "-", "stars sure"),
                                  ("star_cirrus", "c", ":", "stars cirrus"),
                                  ("diff_possible", "tab:orange", "--", "difference possible"), ("diff_sure", "tab:orange", "-", "difference sure")):
        ax.plot(t, area[key], c=color, ls=ls, lw=1, label=label)
    ax.plot(t, d.star_known, c="gray", lw=0.8, label="star map known")
    for tt, s in zip(t, d.state):
        ax.axvspan(tt - pd.Timedelta(minutes=1.2), tt + pd.Timedelta(minutes=1.2), ymin=0, ymax=0.04, color=colors.get(s, "k"), lw=0)
    for i in picks:
        ax.axvline(t[i], c="k", lw=0.5, ls=":")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8, loc="upper left", ncol=6); ax.set_title(f"{day}: sky share of the detections, state band at the bottom")
    paths = {os.path.basename(p): p for p in day_frames(day)}
    for j, i in enumerate(picks):
        a = fig.add_axes([0.01 + j * 0.124, 0.03, 0.12, 0.55])
        draw(a, read_frame(paths[d.frame[i]]), dets[i], sky, f"{d.frame[i][9:13]} {d.state[i]}", moon_masks(frame_time(paths[d.frame[i]]))[1])
    return fig


NATS_DIR = os.path.expanduser("~/data/allsky/nats")
DIFF_LEVELS = [(2.0, 1.0), (3.0, 1.5), (4.0, 2.0), (6.0, 3.0), (8.0, 4.0)]  # (seed, grow) levels of the difference regions tried
MAX_DT = 180.0  # s; a telescope frame is matched with the allsky frame named within this


def telescope_points(nights):
    """The region maps at the place each telescope pointed, against its zero point (fetch_nats.py), for the computed
    nights; to check the meaning of the marks and to try other levels.

    The telescope zero point (zero_monitor.lc, mag, ~0 = clear, negative = light lost) of a science frame is joined with
    its pointing (ALT_TEL, AZ_TEL of the raw frame) and with the allsky frame of the night closest in time (file name,
    within MAX_DT); the maps are read at the grid point nearest to the pointing. Points below 30 deg, in twilight or
    without an allsky frame are left out; a failed zero point (NaN) is kept.

    Parameters
    ----------
    nights : sequence of str
        Nights computed by run_night.

    Returns
    -------
    pandas.DataFrame
        One row per telescope frame: night, frame (allsky), telescope, zero_value, alt, az, moon (inside the difference's
        Moon mask), veil (night_veil), missing, missing_sure, dim (star maps, NaN = unknown), zs (signed difference),
        and diff_<hi> (bool: inside the difference regions of each DIFF_LEVELS level).
    """
    zp = pd.read_parquet(f"{NATS_DIR}/zero_points.parquet")
    raw = pd.read_parquet(f"{NATS_DIR}/raw_frames.parquet")
    raw["oca_jd"] = raw.JD - 2460000
    zp = zp[zp["mode"] == "lc"].sort_values("oca_jd")
    pts = pd.merge_asof(zp, raw.sort_values("oca_jd")[["oca_jd", "telescope", "ALT_TEL", "AZ_TEL"]], on="oca_jd", by="telescope", tolerance=1e-5, direction="nearest")
    pts = pts.dropna(subset=["ALT_TEL"])
    pts = pts[pts.ALT_TEL >= 30.0]
    pts["unix"] = (pts.oca_jd + 2460000 - 2440587.5) * 86400.0
    sky = sky_mask(FRAME_SHAPE, CAMERA_PARAMS, 30.0)
    g = sky[::GRID, ::GRID]
    rows = []
    for day in nights:
        d, maps, zs = load_night(day)
        veil = night_veil(d)
        names = d.frame.str[:13]
        ft = (pd.to_datetime(names, format="%Y%m%d_%H%M") - pd.Timestamp("1970-01-01")).dt.total_seconds().values
        for _, p in pts[(pts.unix > ft.min() - MAX_DT) & (pts.unix < ft.max() + MAX_DT)].iterrows():
            i = int(np.argmin(np.abs(ft - p.unix)))
            if abs(ft[i] - p.unix) > MAX_DT or d.twilight[i]:
                continue
            x, y = altaz_to_xy(p.ALT_TEL, p.AZ_TEL, CAMERA_PARAMS)
            gy, gx = int(round(float(y) / GRID)), int(round(float(x) / GRID))
            if not (0 <= gy < g.shape[0] and 0 <= gx < g.shape[1] and g[gy, gx]):
                continue
            moon = moon_masks(frame_time(d.frame[i]))[1][int(round(float(y))), int(round(float(x)))]
            z = np.abs(zs[i])
            row = dict(night=day, frame=d.frame[i], telescope=p.telescope, zero_value=p.zero_value, alt=p.ALT_TEL, az=p.AZ_TEL, moon=bool(moon),
                       moon_x=d.moon_x[i], veil=bool(veil[i]), missing=maps["missing"][i][gy, gx], missing_sure=maps["missing_sure"][i][gy, gx], dim=maps["dim"][i][gy, gx], zs=zs[i][gy, gx])
            for hi, lo in DIFF_LEVELS:
                row[f"diff_{hi:g}"] = bool(regions(z, g, hi, lo)[gy, gx])
            rows.append(row)
        del d, maps, zs
        gc.collect()
    return pd.DataFrame(rows)


def marks(t, star_possible=STAR_POSSIBLE, star_sure=STAR_SURE, diff_possible=HI, diff_sure=HI_SURE):
    """The observers' mark at each telescope point (telescope_points) for given levels: "red" (hides everything: star
    sure, or difference sure where the star map cannot tell), "yellow" (possible by either method or the veil, not red), "none",
    or "moon" (inside the Moon mask). diff_possible / diff_sure are seed levels from DIFF_LEVELS."""
    s_sure = t.missing_sure > star_sure
    s_poss = (t.missing > star_possible) | (t.dim < CIRRUS_DIM) | s_sure
    red = s_sure | (t[f"diff_{diff_sure:g}"] & t.missing_sure.isna())
    poss = s_poss | t[f"diff_{diff_possible:g}"] | red | t.veil
    return pd.Series(np.select([t.moon, red, poss], ["moon", "red", "yellow"], "none"), index=t.index)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("cases", "nights", "plot", "telescopes"):
        sys.exit(__doc__)
    import matplotlib
    matplotlib.use("Agg")
    if sys.argv[1] == "cases":
        fig_cases().savefig(sys.argv[2], dpi=70)
    elif sys.argv[1] == "telescopes":
        t = telescope_points(sys.argv[2:])
        t.to_parquet(f"{OUT_DIR}/telescopes.parquet")
        print(len(t), "telescope points")
    elif sys.argv[1] == "nights":
        ra = RegionAnalyzer()
        for day in sys.argv[2:]:
            run_night(ra, day)
    else:
        import matplotlib.pyplot as plt
        for day in sys.argv[2:]:
            fig = fig_night(day)
            fig.savefig(f"{OUT_DIR}/{day}.png", dpi=70)
            plt.close(fig)
            gc.collect()  # a closed figure (8 full frames) is freed only by the cycle collector
