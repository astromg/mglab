"""Study: the difference against a clear reference frame instead of the previous one (report: "Clear reference frame").

Each frame minus the last frame of the run that was clear over the whole sky (star map possible < CLEAR_STAR_AREA,
difference regions < CLEAR_DIFF_AREA, star_seen > CLEAR_SEEN), derotated to now, with a gain and an offset fitted on
the part of the sky both frames call clear; its regions mark where the sky differs from the clear sky, including a
cloud that stands still or forms in place, which the step difference misses. A run keeps all its frames in memory:
keep it to about an hour.

Tried and dropped (2026-10-29, in the report): other scales of diff_ls_n (no better), a reference built per pixel
(it takes in the clouds the other detections miss).

    python allsky_diff.py reference 20261005 0130 0230 out.png
"""

import os
import sys

import numpy as np
from scipy import ndimage
from allsky_lib import *
from allsky_lib import _pixel_noise
import allsky_regions as R



def smoothed(d, valid, sigma=DIFF_SIGMA):
    """Gaussian smoothing of d over the valid pixels only (normalised convolution); NaN outside valid."""
    w = ndimage.gaussian_filter(valid.astype(float), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(valid, ndimage.gaussian_filter(np.where(valid, d, 0.0), sigma) / w, np.nan)


CLEAR_STAR_AREA = 0.02     # a frame is a clear reference when the star map marks less than this share as possible cloud,
CLEAR_DIFF_AREA = 0.02     # the difference regions less than this,
CLEAR_SEEN = 0.9           # and star_seen is above this


def fit_gain_offset(img, ref, use, n_iter=3, k=3.0):
    """img ~ gain * ref + offset on the pixels `use`, least squares with k-sigma clipping (stars, residual clouds).
    Pass smoothed frames: with the pixel noise in ref the slope comes out too low (0.84 for two clear frames 2 min
    apart on a dark sky).

    Returns
    -------
    (float, float)
        gain, offset.
    """
    if img.shape != ref.shape or img.shape != use.shape:
        raise ValueError("img, ref and use must have the same shape")
    y, x = img[use].astype(float), ref[use].astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 1000:
        raise ValueError("too few pixels for the fit")
    for _ in range(n_iter):
        gain, offset = np.polyfit(x[ok], y[ok], 1)
        r = y - (gain * x + offset)
        s = 1.4826 * np.median(np.abs(r[ok] - np.median(r[ok])))
        ok = ok & (np.abs(r) < k * max(s, 1.0))
    return float(gain), float(offset)


def reference_diff(img, ref, dt, valid, clear):
    """The frame against a clear reference frame.

    Parameters
    ----------
    img, ref : numpy.ndarray
        The frame and the reference frame.
    dt : float
        img time minus ref time [s] (mtimes).
    valid : numpy.ndarray of bool
        Pixels to use (sky minus the Moon).
    clear : numpy.ndarray of bool
        Pixels clear in both frames (for the gain and offset fit).

    Returns
    -------
    dict
        zs (smoothed difference / its pixel noise on the region grid, signed), d (full resolution, NaN outside valid),
        noise, gain, offset.
    """
    p = derotate(ref, dt, CAMERA_PARAMS)
    valid = valid & np.isfinite(p)
    use = valid & clear
    gain, offset = fit_gain_offset(smoothed(img.astype(float), use), smoothed(np.nan_to_num(p), use), use)
    d = np.where(valid, img - (gain * p + offset), np.nan)
    noise = _pixel_noise(np.nan_to_num(d), valid)
    zs = smoothed(np.nan_to_num(d), valid)[::GRID, ::GRID] / noise
    return dict(zs=np.nan_to_num(zs), d=d, noise=noise, gain=gain, offset=offset)


def clear_frames(table):
    """Boolean Series: frames of a region table (allsky_regions run_night) that are clear references."""
    return (table.star_area < CLEAR_STAR_AREA) & (table.diff_area < CLEAR_DIFF_AREA) & (table.star_seen > CLEAR_SEEN)


def run_reference(day, start, end, step=1):
    """Frames of `day` from HHMM start to end (every step-th) through RegionAnalyzer and against the last clear
    reference frame seen so far in the run (none until the first clear frame).

    Returns
    -------
    list of dict
        Per frame: row (RegionAnalyzer row plus ref_frame, ref_age [min], gain, offset, ref_area, ref_sure: sky share of
        the reference regions), img, det (RegionAnalyzer detections), ref (reference_diff output or None), ref_det
        (dict possible / sure regions of |zs| or None), moon (the Moon mask of the difference).
    """
    ra = R.RegionAnalyzer()
    paths = day_frames(day)
    k0, k1 = frame_index(paths, start), frame_index(paths, end)
    up = lambda m: np.repeat(np.repeat(m, GRID, 0), GRID, 1)[:FRAME_SHAPE[0], :FRAME_SHAPE[1]]
    ref, out = None, []
    for k in range(max(k0, 1), k1 + 1, step):
        o = ra.frame(paths[k], paths[k - 1])
        row = o["row"]
        t = frame_time(paths[k])
        moon_alt, moon_az = body_altaz("moon", t)
        valid = ra.sky & ~moon_mask(FRAME_SHAPE, CAMERA_PARAMS, moon_alt, moon_az, MOON_RADIUS_PIX)
        res, rdet = None, None
        if ref is not None:
            clear = up(~o["det"]["possible"] & ~ref["possible"])
            res = reference_diff(o["img"], ref["img"], frame_mtime(paths[k]).unix - ref["mtime"], valid, clear)
            z = np.abs(res["zs"])
            rdet = dict(possible=regions(z, ra.g), sure=regions(z, ra.g, HI_SURE, LO_SURE))
            row.update(ref_frame=ref["frame"], ref_age=(frame_mtime(paths[k]).unix - ref["mtime"]) / 60, gain=res["gain"], offset=res["offset"],
                       ref_area=float(rdet["possible"][ra.g].mean()), ref_sure=float(rdet["sure"][ra.g].mean()))
        out.append(dict(row=row, img=o["img"], det=o["det"], ref=res, ref_det=rdet, moon=o["moon_diff"]))
        if row["star_area"] < CLEAR_STAR_AREA and row["diff_area"] < CLEAR_DIFF_AREA and row["star_seen"] > CLEAR_SEEN:
            ref = dict(img=o["img"], mtime=frame_mtime(paths[k]).unix, frame=row["frame"], possible=o["det"]["possible"])
        print(row["frame"], row["state"], f"star {row['star_area']:.2f}/{row['star_sure']:.2f} diff {row['diff_area']:.2f}",
              "" if res is None else f"ref {row['ref_frame'][9:13]} ({row['ref_age']:.0f} min) gain {res['gain']:.2f} area {row['ref_area']:.2f}/{row['ref_sure']:.2f}", flush=True)
    return out


def fig_reference(out, picks):
    """Frames `picks` (indices into run_reference output): the frame with the usual detections; the reference
    difference (+-6 noises) with its regions (yellow possible, red sure)."""
    import matplotlib.pyplot as plt
    sky = sky_mask(FRAME_SHAPE, CAMERA_PARAMS, 30.0)
    fig, axs = plt.subplots(2, len(picks), figsize=(4.2 * len(picks), 8.6))
    for j, i in enumerate(picks):
        o, row = out[i], out[i]["row"]
        R.draw(axs[0, j], o["img"], o["det"], sky, f"{row['frame'][:13]} {row['state']}", o["moon"])
        ax = axs[1, j]
        if o["ref"] is None:
            ax.set_axis_off()
            continue
        ax.imshow(np.where(sky, o["ref"]["d"], np.nan), cmap="gray", vmin=-6 * o["ref"]["noise"], vmax=6 * o["ref"]["noise"])
        gx, gy = R._grid_xy(o["ref_det"]["sure"].shape)
        for key, color in (("possible", "yellow"), ("sure", "red")):
            if o["ref_det"][key].any():
                ax.contour(gx, gy, o["ref_det"][key].astype(float), levels=[0.5], colors=color, linewidths=1.3)
        R._moon_area(ax, o["moon"], sky)
        R._frame_axes(ax, f"minus {row['ref_frame'][9:13]} ({row['ref_age']:.0f} min), gain {row['gain']:.2f}")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    if len(sys.argv) < 6 or sys.argv[1] != "reference":
        sys.exit(__doc__)
    import matplotlib
    matplotlib.use("Agg")
    day, start, end, png = sys.argv[2:6]
    res = run_reference(day, start, end)
    fig_reference(res, np.unique(np.linspace(0, len(res) - 1, 6).round().astype(int))).savefig(png, dpi=60)
