"""Build a self-contained HTML report of the allsky cloud detection as it is now: the method (allsky_lib.py), what it
gives on whole nights, the checks against the telescopes, the study under way and what was tried and dropped.

Uses allsky_lib.py, the night tables of allsky_regions.py (~/data/allsky/regions/), the telescope zero points in
~/data/allsky/nats/ (fetch_nats.py), the clear-sky calibration (allsky_calib.py), the reference-frame figures of
allsky_diff.py (~/data/allsky/diff/) and ~20 frames for the illustrations; writes ~/data/allsky/report/allsky_report.html.
The report of the study up to 2026-10-29 (segments, caches) is kept in ~/data/allsky/attic/.

    python allsky_report.py
"""

import base64
import gc
import io
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from allsky_lib import *
from allsky_camera import BRIGHT_STARS
import allsky_calib
import allsky_regions as regions

OUT = os.path.expanduser("~/data/allsky/report/allsky_report.html")
DIFF_DIR = os.path.expanduser("~/data/allsky/diff")
SKY30 = sky_mask(FRAME_SHAPE, CAMERA_PARAMS, 30.0)
REGION_NIGHTS = ["20261001", "20261005", "20261007", "20261011", "20261017", "20261018", "20261025", "20261027", "20261028", "20261029"]


def png(fig):
    """Figure as an <img> tag with the PNG inlined."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=75, bbox_inches="tight")
    plt.close(fig)
    gc.collect()  # a closed figure (with its frames) is freed only by the cycle collector
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def png_file(path):
    """A PNG file as an <img> tag, or a note when it is missing."""
    if not os.path.exists(path):
        return f'<p class="muted"><b>missing: {path}</b></p>'
    with open(path, "rb") as fh:
        return f'<img src="data:image/png;base64,{base64.b64encode(fh.read()).decode()}">'


def table(df, fmt="{:.2f}"):
    """DataFrame as an HTML table."""
    return df.to_html(classes="t", float_format=lambda v: fmt.format(v), border=0)


def crop(ax):
    ax.set_xlim(150, 1080)
    ax.set_ylim(1100, 150)
    ax.axis("off")


# ---------------------------------------------------------------- data

def nights_table(nights=REGION_NIGHTS):
    """The night tables of allsky_regions.py together, with the time, the night and the veil."""
    d = pd.concat([pd.read_parquet(f"{regions.OUT_DIR}/{n}.parquet").assign(night=n) for n in nights], ignore_index=True)
    d["t"] = pd.to_datetime(d.frame.str[:13], format="%Y%m%d_%H%M")
    d["veil"] = regions.night_veil(d)
    return d


def region_areas(nights=REGION_NIGHTS):
    """Mean sky share of each detection per state over the nights (no twilight), and the share of clear frames where
    each covers more than 5 % of the sky."""
    rows = []
    g = SKY30[::GRID, ::GRID]
    for day in nights:
        d, maps, zs = regions.load_night(day)
        veil = regions.night_veil(d)
        areas = pd.DataFrame([det_areas(detections(regions.night_maps(maps, i), zs[i], g, veil=veil[i]), g) for i in range(len(d))])
        rows.append(pd.concat([d[["state", "twilight"]], areas], axis=1))
        del maps, zs
    d = pd.concat(rows)
    names = dict(possible="yellow (incl. red)", sure="red", star_possible="stars possible", star_sure="stars sure", star_cirrus="stars cirrus",
                 diff_possible="difference possible", diff_sure="difference sure")
    d = d[~d.twilight].rename(columns=names)
    cols = list(names.values())
    by_state = d.groupby("state")[cols].mean().reindex([s for s in SKY_STATES if s in set(d.state)])
    by_state.insert(0, "frames", d.groupby("state").size())
    clear = d[d.state == "clear"]
    false = pd.DataFrame({c: [(clear[c] > 0.05).mean()] for c in cols}, index=["clear frames with > 5 % of the sky"])
    return by_state, false


def telescope_marks():
    """The observers' marks at the telescope pointings (allsky_regions.telescope_points) against the telescope's loss:
    counts per loss class with and without the Moon, the levels tried with the Moon up, and a few numbers for the text."""
    path = f"{regions.OUT_DIR}/telescopes.parquet"
    if not os.path.exists(path):
        return None
    t = pd.read_parquet(path)
    loss = -t.zero_value
    t["telescope loss [mag]"] = pd.cut(loss.fillna(99), [-9, 0.1, 0.3, 1.0, 2.0, 98, 100], labels=["< 0.1", "0.1–0.3", "0.3–1", "1–2", "> 2", "failed"])
    t["Moon"] = np.where(t.moon_x > 0.1, "Moon up", "no Moon")
    t["mark"] = regions.marks(t)
    counts = pd.crosstab([t.Moon, t["telescope loss [mag]"]], t.mark).reindex(columns=["none", "yellow", "red", "moon"], fill_value=0)
    m = t[t.moon_x > 0.1]
    l = -m.zero_value
    clear, mid, thick = l < 0.1, (l >= 0.3) & (l <= 2), (l > 1) | m.zero_value.isna()
    rows = []
    for name, kw in (("now", {}), ("stars possible 0.3", dict(star_possible=0.3)), ("stars possible 0.7", dict(star_possible=0.7)),
                     ("stars sure 0.8", dict(star_sure=0.8)), ("difference possible seeds 2", dict(diff_possible=2.0)),
                     ("difference possible seeds 4", dict(diff_possible=4.0)), ("difference sure seeds 4", dict(diff_sure=4.0)),
                     ("difference sure seeds 8", dict(diff_sure=8.0))):
        k = regions.marks(m, **kw)
        ok = k != "moon"
        rows.append({"levels": name, "marked, clear (< 0.1 mag)": f"{(ok & clear & (k != 'none')).sum()} / {(ok & clear).sum()}",
                     "red, loss < 0.3 mag": int(((k == "red") & (l < 0.3)).sum()),
                     "marked, 0.3–2 mag": f"{(ok & mid & (k != 'none')).sum()} / {(ok & mid).sum()}",
                     "red, > 1 mag or failed": f"{(ok & thick & (k == 'red')).sum()} / {(ok & thick).sum()}"})
    n = t[t.moon_x <= 0.1]
    ln = -n.zero_value
    nm = (ln >= 0.3) & (ln <= 2)
    numbers = dict(points=len(t), nights=t.night.nunique(), nomoon_mid=int(nm.sum()), nomoon_mid_marked=int((nm & (n.mark != "none")).sum()),
                   nomoon_clear=int((ln < 0.1).sum()), nomoon_clear_marked=int(((ln < 0.1) & (n.mark != "none")).sum()))
    return counts, pd.DataFrame(rows).set_index("levels"), numbers


# ---------------------------------------------------------------- figures

def fig_camera():
    paths = day_frames("20260713")
    path = paths[frame_index(paths, "0500")]
    img = read_frame(path)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.5))
    for alt_c in (0, 30, 60):
        xc, yc = altaz_to_xy(np.full(361, alt_c), np.arange(361), CAMERA_PARAMS)
        ax.plot(xc, yc, "y-", lw=0.8)
    names = list(BRIGHT_STARS)
    ra, dec = np.array([BRIGHT_STARS[n] for n in names]).T
    alt, az = radec_to_altaz(ra, dec, frame_time(path))
    x, y = altaz_to_xy(alt, az, CAMERA_PARAMS)
    for n, xi, yi, a in zip(names, x, y, alt):
        if a > 3:
            ax.plot(xi, yi, "o", mfc="none", mec="r", ms=10)
            ax.text(xi + 10, yi - 10, n, color="r", fontsize=7)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    ax.set_title(f"{os.path.basename(path)}: alt 0/30/60° and catalogue positions", fontsize=9)
    return png(fig)


def fig_derotation():
    paths = day_frames("20260713")
    k = frame_index(paths, "0500")
    a, b = read_frame(paths[k - 1]), read_frame(paths[k])
    dt = (frame_mtime(paths[k]) - frame_mtime(paths[k - 1])).sec
    raw, der = b - a, b - derotate(a, dt, CAMERA_PARAMS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, d, t in [(axes[0], raw, f"plain difference, std {np.nanstd(raw[SKY30]):.1f}"), (axes[1], der, f"after derotation, std {np.nanstd(der[SKY30]):.1f}")]:
        ax.imshow(np.where(SKY30, d, np.nan), cmap="RdBu_r", vmin=-40, vmax=40)
        crop(ax)
        ax.set_title(t, fontsize=10)
    fig.suptitle(f"clear night 20260713, consecutive frames {dt:.0f} s apart", fontsize=10)
    return png(fig)


def fig_zp_clear(d):
    """Frame zero point against the Moon on the ten nights, the clear frames coloured by date, with zp_clear."""
    d = d[~d.twilight & d.transp.notna()]
    zp = d.transp + d.zp_ref
    c = (d.state == "clear") & (d.star_seen > 0.9)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(d.moon_x, zp, ".", color="lightgray", ms=3, label="all frames")
    days = (d.t - d.t.min()).dt.days
    sc = ax.scatter(d.moon_x[c], zp[c], c=days[c], s=8, cmap="viridis", label="clear frames (colour: date)")
    xx = np.linspace(0, 1, 50)
    ax.plot(xx, ZP_CLEAR[0] + ZP_CLEAR[1] * xx, "k-", label=f"zp_clear = {ZP_CLEAR[0]} {ZP_CLEAR[1]:+} × moon_x")
    plt.colorbar(sc, ax=ax, label=f"days since {d.t.min():%Y-%m-%d}")
    ax.set_xlabel("moon_x = moon_flux(phase) × sin(Moon altitude)^0.25")
    ax.set_ylabel("zp_frame")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return png(fig)


def fig_calibration(cal):
    """The clear-sky calibration: seen share per alt x az cell, against the distance from the Moon, the per-star dimming."""
    fig = plt.figure(figsize=(15, 4.2))
    ax = fig.add_subplot(1, 3, 1, projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    az, alt = np.radians(cal["az_edges"]), 90 - cal["alt_edges"]
    m = ax.pcolormesh(az, alt, cal["altaz_rate"], cmap="RdYlGn", vmin=0.5, vmax=1, shading="flat")
    for i, j in np.argwhere(cal["altaz_rate"] < cal["min_rate"]):
        ax.plot(np.radians((cal["az_edges"][j] + cal["az_edges"][j + 1]) / 2), 90 - (cal["alt_edges"][i] + cal["alt_edges"][i + 1]) / 2, "kx")
    ax.set_ylim(0, 60); ax.set_yticks([30, 60]); ax.set_yticklabels(["60°", "30°"])
    fig.colorbar(m, ax=ax, shrink=0.8, label="share of expected stars seen")
    ax.set_title(f"clear sky, alt × az (x: left out, < {float(cal['min_rate'])})", fontsize=9)
    ax = fig.add_subplot(1, 3, 2)
    ax.step(cal["dist_edges"][:-1], cal["moon_rate"], where="post", color="k")
    ax.axhline(float(cal["min_rate"]), color="gray", ls="--", lw=0.8)
    ax.set_xlim(0, 120); ax.set_ylim(0.5, 1.02); ax.set_xlabel("distance from the Moon [deg]"); ax.set_ylabel("share seen")
    ax.set_title("clear sky against the distance from the Moon", fontsize=9)
    ax = fig.add_subplot(1, 3, 3)
    sd = cal["star_dim"][np.isfinite(cal["star_dim"])]
    ax.hist(sd, bins=np.arange(-1.5, 1.01, 0.05), color="gray")
    ax.set_xlabel("star zero point − frame zero point [mag]"); ax.set_ylabel("stars")
    ax.set_title(f"per-star clear-sky dimming ({len(sd)} stars; {len(cal['unreliable'])} unreliable left out)", fontsize=9)
    fig.tight_layout()
    return png(fig)


# ---------------------------------------------------------------- report

def main():
    nights = nights_table()
    areas, areas_false = region_areas()
    tel = telescope_marks()
    cal = load_calibration()
    dark = nights[~nights.twilight]
    limits = pd.DataFrame([("thin", f"transp < {STATE_LIMITS['thin_transp']} {STATE_LIMITS['thin_transp_moon']:+} × moon_x ({STATE_LIMITS['thin_transp'] + STATE_LIMITS['thin_transp_moon']:+.2f} at full Moon)", f"diff_ls_n > {STATE_LIMITS['thin_diff']}"),
                           ("cloudy", f"transp < {STATE_LIMITS['cloudy_transp']}", f"diff_ls_n > {STATE_LIMITS['cloudy_diff']}"),
                           ("overcast", f"transp < {STATE_LIMITS['overcast_transp']} or no zero point", ""),
                           ("clear", "otherwise", "")], columns=["state", "stars", "or the difference"]).set_index("state")
    tn = tel[2] if tel is not None else {}

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>OCM allsky clouds</title>
<style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1150px;margin:2em auto;padding:0 16px;color:#1d1d1f;background:#fff;line-height:1.5}}
h1{{font-size:1.8em;margin-bottom:.2em}} h2{{margin-top:2em;border-bottom:1px solid #ddd;padding-bottom:.2em}} h3{{margin-top:1.4em}}
img{{max-width:100%;height:auto;display:block;margin:1em 0}} .muted{{color:#666}}
table.t{{border-collapse:collapse;margin:1em 0;font-size:.9em}} table.t td,table.t th{{padding:4px 10px;border-bottom:1px solid #eee;text-align:left}}
table.t th{{background:#f5f5f7}} .box{{background:#f5f5f7;border-radius:8px;padding:.8em 1.2em}} code{{background:#f0f0f0;padding:0 3px;border-radius:3px}}
</style></head><body>
<h1>Cloud detection with the OCM allsky camera</h1>
<p class="muted">The method as it is on {pd.Timestamp.now():%Y-%m-%d}. Code: <code>mglab/ocm/allsky/allsky_lib.py</code> (<code>SkyAnalyzer</code>,
<code>StateTracker</code>, <code>detections</code>); one case at a time: <code>allsky_case.ipynb</code>. The study up to 2026-10-29 (fixed segments and
everything around them) is in <code>~/data/allsky/attic/</code>.</p>

<div class="box"><b>Summary</b><ul>
<li>Every frame (~2.4 min) of the allsky camera alone gives the <b>state of the whole sky</b> — clear / thin / cloudy / overcast — and
<b>where the clouds are</b>, shown to the observers with two marks only: <b>red outline</b> = cloud that hides everything, <b>yellow fill</b> =
cirrus, semi-transparent or uncertain cloud, or a veil over the whole sky; a <b>light cyan area</b> marks the Moon, where nothing is detected.
~0.8 s of computing per frame.</li>
<li>Two methods: the <b>stars</b> (which of the stars a clear sky would show are missing — any cloud, still or moving, but coarse under a bright
Moon) and the <b>difference</b> with the previous frame after the sky rotation is removed (moving cloud, early, also under a full Moon; not a
still cloud).</li>
<li>Against the telescopes' zero points at the place they point ({tn.get('points', 0)} frames, {tn.get('nights', 0)} nights): with the Moon up every loss of
0.3–2 mag is marked and a clear sky is marked at ~2 % of the points; without the Moon {tn.get('nomoon_mid_marked', 0)} of {tn.get('nomoon_mid', 0)} losses of
0.3–2 mag are marked now that a veil turns the whole sky yellow.</li>
<li>Calibration: the camera model once per camera; the clear-sky zero point <code>zp_clear</code> is a constant fitted by hand and drifts (to be made
self-learning); the clear-sky star calibration is optional, by hand after cleaning.</li>
</ul></div>

<h2>1. Data and camera model</h2>
<p>Archive <code>/data/misc/allsky/&lt;YYYYMMDD&gt;/&lt;YYYYMMDD&gt;_HHMM.jpg</code>: JPG 1240×1256, R = G = B, every ~2.4 min at night, names in UT,
automatic exposure not stored. Sky positions use the time in the file name; the interval between two frames the file mtimes (the mtime
itself is off by 15–220 s). Only alt ≥ 30° is used (domes, horizon and overlays fall outside). The camera model pixel ↔ (alt, az) —
zenith at ({CAMERA_PARAMS[0]:.1f}, {CAMERA_PARAMS[1]:.1f}) px, ~6.2 px/deg, nearly linear radial scale, N rotated {CAMERA_PARAMS[2]:.1f}° — is fitted on
bright stars, rms ~1 px (0.2°), once per camera: <code>python allsky_camera.py &lt;clear frames&gt;</code>.</p>
{fig_camera()}

<h2>2. One frame</h2>
<h3>2.1 The stars</h3>
<ol>
<li><b>Where they should be.</b> The Yale Bright Star Catalogue (V &lt; 6) is put on the frame with the camera model for the time in the file
name; stars in the Moon glare ({MOON_GLARE[0]:.0f}° + {MOON_GLARE[1]:.0f}° × Moon flux, 30° at full Moon) are dropped.</li>
<li><b>How bright each one is.</b> <i>peak</i> = the brightest pixel of a 3×3 box at the predicted position, <i>background</i> = the median of
the 15×15 window, <i>σ</i> = its robust scatter; <i>amp</i> = peak − background (no PSF: the JPEG stars are 1–3 px). <b>Detected</b> when amp &gt; 5σ.</li>
<li><b>Two zero points of the frame</b> (V + 2.5 log10 amp, median over detected unsaturated stars): <code>zp_frame</code> from V &lt; 5 (at least 5
stars) for transp, and <code>zp_stars</code> from 2 &lt; V &lt; 4.5 (at least 20) for the expectation — brighter stars flatten (the camera
response), fainter ones near the threshold are biased.</li>
<li><b>Expected</b> (what a clear sky would show in <i>this</i> frame): predicted amp 10<sup>−0.4 (V − zp_stars)</sup> above {STAR_EXP} × the
detection threshold, taken from the <i>clear-sky</i> noise of the frame (the lower quartile of the local σ — under a bright textured cloud a star's
own σ grows and the cloud would hide itself); <b>sure</b> above {STAR_CONF} ×. A star on a background too bright to measure counts as expected
and not seen. Because the expectation follows the frame itself, it does not depend on a model of the Moon or of the exposure.</li>
<li><b>star_seen</b> = seen expected / expected, whole sky and the zenith cap (≥ {ZEN_ALT:.0f}°).</li>
</ol>

<h3>2.2 transp — light lost over the whole sky</h3>
<p><code>transp = zp_frame − zp_clear(Moon)</code> [mag]: 0 = clear, −0.25 ≈ 20 % of the light lost, −0.75 ≈ 50 %, no value = too few stars.
<code>zp_clear = {ZP_CLEAR[0]} {ZP_CLEAR[1]:+} × moon_x</code>, moon_x = Moon flux from its phase × sin(Moon altitude)<sup>0.25</sup>: the auto exposure
shortens as the Moon brightens the sky. It is the only indicator that sees a <b>veil over the whole sky</b> (the stars dim together, so the
expectation of 2.1 follows them). Its weak point is <code>zp_clear</code> itself: a constant fitted by hand, ~0.15–0.2 mag night-to-night scatter at
a bright Moon, and a drift during October (below: the clear frames of the ten nights against the line).</p>
{fig_zp_clear(nights)}

<h3>2.3 Clear-sky star calibration (optional)</h3>
<p><code>python allsky_calib.py calibrate &lt;nights&gt;</code> on nights known to be clear, by hand after cleaning or a change of the camera; read by
<code>SkyAnalyzer(calib=load_calibration())</code>. Without it everything works, with more false yellow on the low northern edge and next to the Moon.
It leaves out the stars and places a clear sky does not show reliably: unreliable stars (seen in fewer than {allsky_calib.MIN_UNRELIABLE:.0%} of at least
{allsky_calib.MIN_CHANCES} chances: close doubles, crowded fields), alt × az cells and distances from the Moon where fewer than {float(cal['min_rate']) if cal is not None else 0.8:.0%}
of the expected stars are seen, and it gives each star its clear-sky dimming for the dimming map.</p>
{(f"<p>Now: {', '.join(str(n) for n in cal['nights'])} — {len(cal['unreliable'])} unreliable stars, {int((cal['altaz_rate'] < cal['min_rate']).sum())} cells left out.</p>" + fig_calibration(cal)) if cal is not None else "<p><b>No calibration file.</b></p>"}

<h3>2.4 The difference and diff_ls_n</h3>
<ol>
<li><b>Derotation.</b> The sky turns ~0.6° between frames; the previous frame is resampled to the time of the current one (pixel → alt/az →
hour angle/declination → + 15°/h × dt → pixel). On a clear sky the difference then drops to the noise (figure).</li>
<li><b>D</b> = current − derotated previous, minus its median over the sky (a change of the auto exposure); {MOON_RADIUS_PIX:.0f}° around the Moon left out.</li>
<li><b>diff_ls_n</b> = std of D box-smoothed over {SMOOTH_PX} px (~2.5°) / the pixel-to-pixel noise of D: the large-scale change in units of the
noise, ~0.3 on a clear sky, above 1 when cloud moves. (Other noise scales were compared on ten nights and are no better.)</li>
</ol>
{fig_derotation()}

<h3>2.5 Star maps</h3>
<p>On a grid of every {GRID}th pixel (~0.65°), around each point, Gaussian-weighted by the distance: the share of the expected stars missing, the share
of the <i>sure</i> stars missing and the mean dimming of the seen stars (2 &lt; V &lt; 4.5, their zero point minus the frame's minus their clear-sky
dimming). The Gaussian width follows the number of expected stars, ~{N_KERNEL:.0f} within it: ~6–9° without Moon, 10–12° with a bright one
({SIGMA_RANGE[0]:.0f}–{SIGMA_RANGE[1]:.0f}°). Unknown where too few stars carry the weight; the glare + {MOON_MARGIN:.0f}° is left out.</p>
<ul><li><b>possible</b>: more than {STAR_POSSIBLE} of the expected stars missing, or the seen ones dimmed by more than {-CIRRUS_DIM} mag;</li>
<li><b>sure</b>: more than {STAR_SURE} of the sure stars missing — an opaque cloud lets no star through;</li>
<li><b>cirrus</b> (a label for plots): possible, not sure, sure stars known — the stars shine through.</li></ul>

<h3>2.6 Difference regions</h3>
<p>D smoothed with a Gaussian of {DIFF_SIGMA:.0f} px over its pixel noise, signed; regions of its absolute value: seeds above {HI} grown down to {LO}
(hysteresis), gaps narrower than {CLOSE_DEG:.0f}° closed, holes filled, regions below {MIN_AREA:.0f} deg² dropped = <b>possible</b>; seeds {HI_SURE} → {LO_SURE} =
<b>sure</b> (a strong change). The sign of one frame does not tell the cloud from the sky it left: over ten nights the + and − parts are equally
starless.</p>

<h3>2.7 The marks for the observers</h3>
<ul>
<li><b>red outline — a cloud that hides everything</b>: the star map's sure, or the difference's sure <i>only where the star map cannot tell</i>
(near the Moon, too few stars). Where the sure stars are known, a strong difference is opaque in only ~36 % of cases and in ~40 % more than
half of them are seen — there the difference gives yellow.</li>
<li><b>yellow fill — cirrus, semi-transparent or uncertain</b>: possible by either method; and the <b>whole sky</b> when a veil covers it
(the smoothed transp below the thin limit while at least {VEIL_SEEN:.0%} of the expected stars are still seen — section 3).</li>
<li><b>light cyan area — the Moon</b>: no detection inside ({MOON_RADIUS_PIX:.0f}° for the difference; the star map leaves out more, its own mask is shown
on its panel).</li>
</ul>
<p>Seven frames — top: the frame with the marks; middle: the raw difference (red brighter, blue darker) with its regions (yellow possible, black sure); bottom: the expected
stars (green seen, red missing, large dots the sure ones) with the star map's regions:</p>
{png(regions.fig_cases())}

<h2>3. In time: the state and the veil</h2>
<p><code>StateTracker</code>: transp smoothed as the median of the last 3 frames (~7 min), diff_ls_n of the last 2 (to stay early).</p>
{table(limits)}
<p>The thin limit of transp moves down with the Moon (the clear-sky zero point is less certain at a bright Moon). Frames with the Sun above −16° are
marked as twilight (zp_clear has no Sun term). <b>Veil</b>: the smoothed transp below the thin limit while the smoothed star_seen is at least
{VEIL_SEEN} — light is lost but the stars are still seen everywhere, so the maps, which compare the stars with each other, cannot show it; then the
whole sky is yellow (to be marked in the alarms too). With star_seen lower the light is lost to clouds the maps already show (on the ten nights
they mark 71 % of the sky at star_seen &lt; 0.5, ~1 % above 0.8), and the veil would only paint their clear parts.</p>
<h3>Ten whole nights</h3>
<p>Sky share of each mark and detection, mean per state (no twilight); {int(dark.veil.sum())} of {len(dark)} frames had a veil:</p>
{table(areas)}
{table(areas_false)}
<p><b>2026-10-07</b> — a thin still veil, no Moon: the telescopes lost up to 1.5 mag, the maps see nothing; the veil makes the sky yellow.</p>
{png(regions.fig_night("20261007"))}
<p><b>2026-10-05</b> — clear, a cloud forming in place in the S around 02:00, no Moon in the second half.</p>
{png(regions.fig_night("20261005"))}
<p><b>2026-10-27</b> — overcast, clearing, clouds again under a bright Moon.</p>
{png(regions.fig_night("20261027"))}
<p><b>2026-10-25</b> — clouds under a full Moon: the difference does the work, the star map hardly sees.</p>
{png(regions.fig_night("20261025"))}

<h2>4. Against the telescopes</h2>
<p><code>python allsky_regions.py telescopes &lt;nights&gt;</code>: for every science frame of a telescope (zero point from NATS, <code>fetch_nats.py</code>;
~0 mag = clear, negative = light lost) the mark at the place it pointed, from the allsky frame within {regions.MAX_DT / 60:.0f} min (grid point nearest
the pointing, above 30°, no twilight). A telescope sees one direction, the maps are smoothed over a few degrees, and the telescopes close under bad
weather, so cloudy points are few.</p>
{(table(tel[0], "{:.0f}") + "<p>With the Moon up, other levels (one changed at a time):</p>" + table(tel[1])) if tel is not None else "<p><b>No telescopes.parquet.</b></p>"}
<ul>
<li>With the Moon up the marks agree with the losses. Two false reds (10-25, jk15 lost 0.1–0.3 mag) are difference-sure regions under a full Moon
where the stars cannot tell; a few thick losses (−2 … −3 mag, 10-27) are yellow, not red (sure-star map unknown); one was missed because the cloud
reached the telescope between two allsky frames.</li>
<li>The levels cannot be tuned on this yet: with the Moon up only ~12 points lost 0.3–2 mag. They stay; fetching NATS regularly (it keeps 30 days)
would build the sample.</li>
<li>Without the Moon the losses are uniform veils (10-07, 10-05 before the cloud formed): {tn.get('nomoon_mid_marked', 0)} of {tn.get('nomoon_mid', 0)} losses of
0.3–2 mag are now yellow through the veil, at the price of {tn.get('nomoon_clear_marked', 0)} of {tn.get('nomoon_clear', 0)} clear points. transp sees such a veil only
partly (−0.16 … −0.25 against −0.4 … −0.7 at the telescopes).</li>
</ul>

<h2>5. Under way: a clear reference frame</h2>
<p><code>python allsky_diff.py reference &lt;night&gt; &lt;HHMM&gt; &lt;HHMM&gt; &lt;png&gt;</code>: each frame minus the last frame that was clear over the whole sky,
derotated to now, with a gain and an offset fitted on the part of the sky both call clear. It sees what the step difference cannot: a cloud that
forms in place or stands still. Top: the frame with the marks; bottom: minus the reference, with its regions.</p>
<p><b>2026-10-05 01:20–02:20</b>, no Moon, a cloud forming in place in the S: the step difference hardly sees it (≤ 7 % of the sky); against the
reference the SW patch is marked from 01:57 and the S field from 02:09, where the star map has it.</p>
{png_file(os.path.join(DIFF_DIR, "reference_20261005.png"))}
<p><b>2026-10-25 02:55–03:55</b>, full Moon: the star map sees nothing; the reference keeps the cloud fields while they stand still (03:37–03:47: 19–56 %
of the sky against 3–31 % for the step difference).</p>
{png_file(os.path.join(DIFF_DIR, "reference_20261025.png"))}
<p>Weak points: no reference before the first clear frame of the night; the reference ages (up to 48 min here); on a mostly cloudy sky the gain fit
has few clear pixels and goes astray. Not yet part of the marks.</p>

<h2>6. Tried and dropped</h2>
<ul>
<li><b>Fixed sky segments</b> (9 for the stars, 28 for the difference) with per-segment indicators — replaced by the regions (2026-10-29).</li>
<li><b>Expected stars from zp_clear</b> (star_frac_ref, the first star map): when zp_clear misses the Moon of the night by 0.3 mag, a third of the
expected stars is "missing" on a clear sky.</li>
<li><b>Cloud memory</b> (signed changes summed with a leak): a departure counts as presence, the threshold breaks the cancelling, exposure changes add
up, a cloud that stays is forgotten. The clear reference frame (5) does what it was meant to.</li>
<li><b>A reference built per pixel</b> (each place against the last frame where it was clear): it takes in exactly the clouds the other detections
miss and becomes a slow step difference.</li>
<li><b>Other scales of diff_ls_n</b> (the noise of the previous frame, predicted from the level, fixed per night): no better.</li>
<li>Earlier: a zero point from a gain + offset fit between frames, many pixel statistics (level, noise, texture: their sign flips with the Moon).</li>
</ul>

<h2>7. Open points</h2>
<ol>
<li><b>A self-learning zp_clear</b> instead of the hand-fitted constant (transp and the veil depend on it; it drifted by 0.3 mag in October).</li>
<li>The veil in the alarms; how the observers see it.</li>
<li>The levels of both methods: the meaning is fixed (red hides everything, yellow cirrus / uncertain), the sample against the telescopes is too
small yet.</li>
<li>The clear reference frame: how long it may age, what to do on a night that starts cloudy, whether it joins the marks.</li>
<li>No Sun term in zp_clear: twilight is only marked.</li>
<li>The service: rebuilt on this library, masks stored on disk (not over NATS).</li>
</ol>
</body></html>"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(html)
    print(OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
