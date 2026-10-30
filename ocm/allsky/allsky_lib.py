"""Allsky camera tools for the OCM archive (/data/misc/allsky/<YYYYMMDD>/<YYYYMMDD>_HHMM.jpg).

Everything the cloud service needs, one frame at a time (SkyAnalyzer) and in time (StateTracker): frame access, the
pixel <-> (alt, az) camera model (fitted once per camera with allsky_camera.py), sky / Moon masks, derotation of a frame
to another time, star measurement against the Yale Bright Star Catalogue and the expectation of a clear sky (optionally
refined by the clear-sky calibration of allsky_calib.py), the whole-sky indicators (transp, star_seen, diff_ls_n) and the
sky state, and the cloud regions: star maps and difference regions on a grid, combined into the two marks shown to the
observers -- red = cloud that hides everything, yellow = cirrus, semi-transparent or uncertain (or a veil over the whole
sky).

Camera model (image as stored: N up, E right, x = column, y = row, y grows downwards):

    z = 90 - alt                                  zenith distance [deg]
    r = k1 * z + k2 * z**3                        distance from the zenith pixel [px]
    x = x0 + r * sin(az + theta)
    y = y0 - r * cos(az + theta)

with az measured from N through E. params = (x0, y0, theta, k1, k2), theta in degrees.
"""

import os
import numpy as np
from PIL import Image
from scipy import ndimage
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_body
import astropy.units as u

ALLSKY_DIR = "/data/misc/allsky"
OCM_LOCATION = EarthLocation(lat=-24.598 * u.deg, lon=-70.196 * u.deg, height=2817 * u.m)
SIDEREAL_RATE = 360.98565 / 86400       # sky rotation [deg per second of time]

# Camera model (x0, y0, theta, k1, k2) fitted to 74 bright-star positions on 8 frames of 20260318 and 20260713
# (starting from a fit to the Moon) in a study notebook (removed; method in allsky_report.py): rms ~1 px (~0.2 deg) above alt 15, ~5 px near the
# horizon. Valid for the frame geometry of 2026 (1240 x 1256); check it again if process_ocm() changes.
CAMERA_PARAMS = (610.88, 624.52, -9.094, 6.21458, -3.7e-05)

# Zero point of a clear sky as a function of the Moon: zp_clear = c0 + c1 * x, x = moon_x(alt, illum) =
# moon_flux(illum) * sin(Moon altitude)**0.25 (0 below the horizon). It moves with the auto exposure, which shortens
# as the Moon brightens the sky; the brightness follows the phase and hardly the altitude. Fitted 2026-10-28 on ~690
# clear frames of 2026-07-28..10-27 plus 57 frames of 10 clear full-Moon nights of 2026 (rms 0.19 mag, 0.15 at
# full Moon, no trend with the Moon altitude). History: illum * sin(alt) (rms 0.47 with the Moon up), then
# flux * sqrt(sin(alt)) = (8.94, -2.52), which was +0.2..+0.5 mag off for a high and -0.2..-0.6 for a low full Moon.
ZP_CLEAR = (8.95, -2.10)

# ---------------------------------------------------------------- frames

def day_frames(day, allsky_dir=ALLSKY_DIR):
    """Sorted list of frame paths of one UT day directory.

    Parameters
    ----------
    day : str
        UT date directory, "YYYYMMDD".
    allsky_dir : str
        Root of the archive.

    Returns
    -------
    list of str
    """
    path = os.path.join(allsky_dir, day)
    if not os.path.isdir(path):
        raise ValueError(f"no such day directory: {path}")
    return [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".jpg")]


def frame_time(path):
    """UT time of a frame from its file name YYYYMMDD_HHMM.jpg.

    Parameters
    ----------
    path : str

    Returns
    -------
    astropy.time.Time
    """
    name = os.path.basename(path)
    if len(name) < 17 or name[8] != "_":
        raise ValueError(f"not a frame name YYYYMMDD_HHMM.jpg: {name}")
    # format given: guessing it raises and catches exceptions inside astropy, whose tracebacks keep the callers' frames
    # (and the images in them) in reference cycles
    return Time(f"{name[0:4]}-{name[4:6]}-{name[6:8]}T{name[9:11]}:{name[11:13]}:00", format="isot", scale="utc")


def frame_mtime(path):
    """File modification time of a frame (seconds resolution, written shortly after the capture).

    Use it only for the interval between neighbouring frames (what derotation needs: the sky turns 0.25 deg per
    minute, and the file name has only minutes). As an absolute time it is unreliable: mtime - name time was seen
    from 15 s to 220 s and changes from night to night; bright stars fit the name time (frame_time) better.

    Parameters
    ----------
    path : str

    Returns
    -------
    astropy.time.Time
    """
    return Time(os.stat(path).st_mtime, format="unix", scale="utc")


def read_frame(path):
    """Frame as a 2-D float array (one channel; the archive is R = G = B).

    Parameters
    ----------
    path : str

    Returns
    -------
    numpy.ndarray
    """
    return np.asarray(Image.open(path))[:, :, 0].astype(float)


# ---------------------------------------------------------------- ephemerides

def body_altaz(body, t, location=OCM_LOCATION):
    """Altitude and azimuth [deg] of a solar-system body ("sun", "moon", ...) at time(s) t.

    Parameters
    ----------
    body : str
    t : astropy.time.Time
        Scalar or array.
    location : astropy.coordinates.EarthLocation

    Returns
    -------
    alt, az : float or numpy.ndarray
    """
    aa = get_body(body, t, location).transform_to(AltAz(obstime=t, location=location))
    return aa.alt.deg, aa.az.deg


def radec_to_altaz(ra, dec, t, location=OCM_LOCATION):
    """Altitude and azimuth [deg] of fixed sky positions (RA, Dec J2000 [deg]) at time t.

    Parameters
    ----------
    ra, dec : float or array_like
    t : astropy.time.Time
        Scalar.
    location : astropy.coordinates.EarthLocation

    Returns
    -------
    alt, az : float or numpy.ndarray
    """
    aa = SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg).transform_to(AltAz(obstime=t, location=location))
    return aa.alt.deg, aa.az.deg


def moon_illumination(t, location=OCM_LOCATION):
    """Illuminated fraction of the Moon at time(s) t (from the Sun-Moon elongation).

    Parameters
    ----------
    t : astropy.time.Time
    location : astropy.coordinates.EarthLocation

    Returns
    -------
    float or numpy.ndarray
    """
    sun = get_body("sun", t, location)
    moon = get_body("moon", t, location)
    return (1 - np.cos(sun.separation(moon).rad)) / 2


def moon_flux(moon_illum):
    """Brightness of the Moon relative to the full Moon, from its illuminated fraction.

    Phase angle a from illum = (1 + cos a) / 2, then the V magnitude law of Krisciunas & Schaefer (1991, PASP 103,
    1033): m = -12.73 + 0.026 |a| + 4e-9 a^4, so flux / flux_full = 10**(-0.4 * (0.026 |a| + 4e-9 a^4)). A half
    Moon gives ~0.1 of the full Moon.

    Parameters
    ----------
    moon_illum : float or array_like
        Illuminated fraction, 0..1.

    Returns
    -------
    float or numpy.ndarray
    """
    illum = np.asarray(moon_illum, dtype=float)
    if np.any((illum < 0) | (illum > 1)):
        raise ValueError("moon_illum must be in [0, 1]")
    a = np.degrees(np.arccos(2 * illum - 1))
    return 10 ** (-0.4 * (0.026 * a + 4e-9 * a**4))


def moon_x(moon_alt, moon_illum):
    """How much the Moon brightens the frame, 0 (below the horizon) .. 1 (full Moon in the zenith):
    moon_flux(moon_illum) * sin(moon_alt)**0.25.

    The weak altitude dependence follows the clear-sky zero point (see ZP_CLEAR): the auto exposure reacts to the
    Moon in the frame, its altitude matters only near the horizon.

    Parameters
    ----------
    moon_alt : float or array_like
        Moon altitude [deg].
    moon_illum : float or array_like
        Illuminated fraction, 0..1.

    Returns
    -------
    float or numpy.ndarray
    """
    alt = np.asarray(moon_alt, dtype=float)
    return np.where(alt > 0, moon_flux(np.clip(moon_illum, 0, 1)) * np.sin(np.radians(np.clip(alt, 0, 90))) ** 0.25, 0.0)


def zp_clear(moon_alt, moon_illum, coeffs=ZP_CLEAR):
    """Expected photometric zero point (V + 2.5 log10 amplitude) of a clear sky for a given Moon.

    zp = c0 + c1 * moon_x(moon_alt, moon_illum).

    Parameters
    ----------
    moon_alt : float or array_like
        Moon altitude [deg].
    moon_illum : float or array_like
        Illuminated fraction, 0..1.
    coeffs : (float, float)
        (c0, c1).

    Returns
    -------
    float or numpy.ndarray
    """
    if len(coeffs) != 2:
        raise ValueError("coeffs must be (c0, c1)")
    return coeffs[0] + coeffs[1] * moon_x(moon_alt, moon_illum)


# Learning zp_clear from the frames themselves (learn_zp_clear): the median zero point of the frames that look clear, over
# the last ZP_LEARN_NIGHTS nights, in bins of moon_x. The median, not an upper envelope: the envelope (75th-90th percentile)
# lies 0.1-0.2 mag above the typical clear frame and turned ordinary clear nights into veils (checked on 10 nights of
# 2026-10: the median gives (9.02-9.03, -2.10...-2.45) against the hand fit (8.95, -2.10)).
ZP_LEARN_NIGHTS = 14       # nights of history
ZP_LEARN_SEEN = 0.9        # a frame looks clear with star_seen at least this ...
ZP_LEARN_DIFF = 1.0        # ... and diff_ls_n below this
ZP_LEARN_BIN = 0.1         # width of the moon_x bins
ZP_LEARN_MIN = 20          # frames a bin needs
ZP_LEARN_SPAN = 0.3        # the slope is fitted only when the bins span this much moon_x, else the old slope is kept


def learn_zp_clear(zp_frame, moon_x, star_seen, diff_ls_n, coeffs=ZP_CLEAR):
    """zp_clear coefficients learned from a history of frames (night frames of the last ZP_LEARN_NIGHTS nights).

    The frames that look clear (star_seen >= ZP_LEARN_SEEN, diff_ls_n < ZP_LEARN_DIFF) are binned in moon_x; each bin with
    at least ZP_LEARN_MIN frames gives the median zp_frame at its centre. A line is fitted through the bins when they span
    at least ZP_LEARN_SPAN in moon_x; otherwise only c0 is fitted with the slope of `coeffs` (e.g. two weeks without the
    Moon). With no bin, `coeffs` is returned unchanged.

    Parameters
    ----------
    zp_frame, moon_x, star_seen, diff_ls_n : array_like
        Per frame, from SkyAnalyzer.analyse (NaN allowed: such frames are skipped).
    coeffs : (float, float)
        The current (c0, c1).

    Returns
    -------
    coeffs : (float, float)
        The learned (c0, c1).
    n_bins : int
        Bins used (0 = nothing learned).
    """
    if len(coeffs) != 2:
        raise ValueError("coeffs must be (c0, c1)")
    zp, x, seen, diff = (np.asarray(a, dtype=float) for a in (zp_frame, moon_x, star_seen, diff_ls_n))
    if not (zp.shape == x.shape == seen.shape == diff.shape):
        raise ValueError("zp_frame, moon_x, star_seen and diff_ls_n must have the same shape")
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(zp) & np.isfinite(x) & (seen >= ZP_LEARN_SEEN) & (diff < ZP_LEARN_DIFF)
    b = np.floor(x[ok] / ZP_LEARN_BIN).astype(int)
    centres, medians = [], []
    for k in np.unique(b):
        if (b == k).sum() >= ZP_LEARN_MIN:
            centres.append((k + 0.5) * ZP_LEARN_BIN)
            medians.append(np.median(zp[ok][b == k]))
    if not centres:
        return (float(coeffs[0]), float(coeffs[1])), 0
    centres, medians = np.array(centres), np.array(medians)
    if centres.max() - centres.min() >= ZP_LEARN_SPAN:
        c1, c0 = np.polyfit(centres, medians, 1)
        return (float(c0), float(c1)), len(centres)
    return (float(np.median(medians - coeffs[1] * centres)), float(coeffs[1])), len(centres)


# ---------------------------------------------------------------- camera model

def altaz_to_xy(alt, az, params):
    """Pixel position (x = column, y = row) of sky direction(s) (alt, az) [deg].

    Parameters
    ----------
    alt, az : float or array_like
        Altitude and azimuth in degrees (az from N through E).
    params : sequence of 5 floats
        (x0, y0, theta, k1, k2), see the module docstring.

    Returns
    -------
    x, y : float or numpy.ndarray
    """
    if len(params) != 5:
        raise ValueError("params must be (x0, y0, theta, k1, k2)")
    x0, y0, theta, k1, k2 = params
    z = 90.0 - np.asarray(alt, dtype=float)
    r = k1 * z + k2 * z**3
    a = np.radians(np.asarray(az, dtype=float) + theta)
    return x0 + r * np.sin(a), y0 - r * np.cos(a)


def xy_to_altaz(x, y, params):
    """Sky direction (alt, az) [deg] of pixel(s) (x = column, y = row); inverse of altaz_to_xy.

    The radial relation r(z) is inverted by interpolation in a table z = 0..120 deg, so r(z) must be increasing
    there (true for any sensible fit).

    Parameters
    ----------
    x, y : float or array_like
    params : sequence of 5 floats
        (x0, y0, theta, k1, k2).

    Returns
    -------
    alt, az : float or numpy.ndarray
        az in [0, 360).
    """
    if len(params) != 5:
        raise ValueError("params must be (x0, y0, theta, k1, k2)")
    x0, y0, theta, k1, k2 = params
    z_table = np.linspace(0, 120, 2401)
    r_table = k1 * z_table + k2 * z_table**3
    if np.any(np.diff(r_table) <= 0):
        raise ValueError("r(z) is not increasing on 0..120 deg; params are not a valid camera model")
    dx = np.asarray(x, dtype=float) - x0
    dy = np.asarray(y, dtype=float) - y0
    z = np.interp(np.hypot(dx, dy), r_table, z_table)
    az = (np.degrees(np.arctan2(dx, -dy)) - theta) % 360
    return 90.0 - z, az


# ---------------------------------------------------------------- stars

def load_bsc(path):
    """Yale Bright Star Catalogue (BSC5, ~9100 stars to V ~ 6.5) from a CSV with columns hr, ra, dec, v.

    The CSV is made once from https://github.com/brettonw/YaleBrightStarCatalog (bsc5-short.json, J2000), see
    fetch_bsc.

    Parameters
    ----------
    path : str

    Returns
    -------
    ra, dec, v : numpy.ndarray
        J2000 RA and Dec [deg], V magnitude.
    """
    if not os.path.exists(path):
        raise ValueError(f"no catalogue file {path}")
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data["ra"], data["dec"], data["v"]


def star_amplitudes(img, x, y, r_bg=7, r_peak=1):
    """Peak height above the local background, local noise and peak value at predicted star positions.

    amplitude = maximum of the (2*r_peak+1)^2 box at the position minus the median of the (2*r_bg+1)^2 window;
    sigma = 1.4826 * MAD of that window (at least 1). Positions too close to the image edge give NaN.

    Parameters
    ----------
    img : numpy.ndarray
        2-D frame.
    x, y : array_like
        Predicted positions [px].
    r_bg, r_peak : int
        Half sizes of the background window and of the peak box [px].

    Returns
    -------
    amplitude, sigma, peak : numpy.ndarray
    """
    if img.ndim != 2:
        raise ValueError("img must be 2-D")
    if not 0 <= r_peak < r_bg:
        raise ValueError("need 0 <= r_peak < r_bg")
    xs, ys = np.round(np.atleast_1d(x)).astype(int), np.round(np.atleast_1d(y)).astype(int)
    amplitude, sigma, peak = (np.full(len(xs), np.nan) for _ in range(3))
    for n, (xi, yi) in enumerate(zip(xs, ys)):
        if not (r_bg <= xi < img.shape[1] - r_bg and r_bg <= yi < img.shape[0] - r_bg):
            continue
        win = img[yi - r_bg:yi + r_bg + 1, xi - r_bg:xi + r_bg + 1]
        med = np.median(win)
        sigma[n] = max(1.4826 * np.median(np.abs(win - med)), 1.0)
        peak[n] = img[yi - r_peak:yi + r_peak + 1, xi - r_peak:xi + r_peak + 1].max()
        amplitude[n] = peak[n] - med
    return amplitude, sigma, peak


# ---------------------------------------------------------------- sky rotation

def altaz_to_hadec(alt, az, lat):
    """Hour angle and declination [deg] of horizontal coordinates (alt, az from N through E) [deg].

    Plain spherical trigonometry, no refraction.

    Parameters
    ----------
    alt, az : float or array_like
    lat : float
        Observer latitude [deg].

    Returns
    -------
    ha, dec : float or numpy.ndarray
        Hour angle in degrees (positive to the west), declination in degrees.
    """
    a, A, f = np.radians(alt), np.radians(az), np.radians(lat)
    dec = np.arcsin(np.sin(f) * np.sin(a) + np.cos(f) * np.cos(a) * np.cos(A))
    ha = np.arctan2(-np.sin(A) * np.cos(a), np.cos(f) * np.sin(a) - np.sin(f) * np.cos(a) * np.cos(A))
    return np.degrees(ha), np.degrees(dec)


def hadec_to_altaz(ha, dec, lat):
    """Horizontal coordinates (alt, az from N through E) [deg] of hour angle and declination [deg]; inverse of
    altaz_to_hadec.

    Parameters
    ----------
    ha, dec : float or array_like
    lat : float
        Observer latitude [deg].

    Returns
    -------
    alt, az : float or numpy.ndarray
        az in [0, 360).
    """
    h, d, f = np.radians(ha), np.radians(dec), np.radians(lat)
    alt = np.arcsin(np.sin(f) * np.sin(d) + np.cos(f) * np.cos(d) * np.cos(h))
    az = np.arctan2(-np.sin(h) * np.cos(d), np.cos(f) * np.sin(d) - np.sin(f) * np.cos(d) * np.cos(h))
    return np.degrees(alt), np.degrees(az) % 360


def derotate(img, dt, params, lat=OCM_LOCATION.lat.deg, alt_min=0.0):
    """Frame resampled so that its sky matches a frame taken dt seconds later (stars land where they are then).

    For every output pixel: (x, y) -> (alt, az) -> (ha, dec) at the later time -> ha - rotation at the time of img
    -> (alt, az) -> (x, y) in img, read with bilinear interpolation. Only the sky moves; anything fixed on the ground
    (domes, overlays) is moved with it and is meaningless in the result.

    Parameters
    ----------
    img : numpy.ndarray
        2-D frame taken at time t.
    dt : float
        Target time minus t [s] (positive = shift the frame forward in time).
    params : sequence of 5 floats
        Camera model.
    lat : float
        Observer latitude [deg].
    alt_min : float
        Output pixels whose source direction is below this altitude [deg] are NaN.

    Returns
    -------
    numpy.ndarray
        Float array of the same shape, NaN where there is no sky to take the value from.
    """
    if img.ndim != 2:
        raise ValueError("img must be 2-D")
    yy, xx = np.indices(img.shape)
    alt, az = xy_to_altaz(xx, yy, params)
    ha, dec = altaz_to_hadec(alt, az, lat)
    alt_src, az_src = hadec_to_altaz(ha - SIDEREAL_RATE * dt, dec, lat)
    x_src, y_src = altaz_to_xy(alt_src, az_src, params)
    out = ndimage.map_coordinates(img.astype(float), [y_src, x_src], order=1, cval=np.nan)
    out[(alt_src < alt_min) | (alt < 0)] = np.nan
    return out


# ---------------------------------------------------------------- masks

def sky_mask(shape, params, alt_min):
    """Boolean mask of the pixels with altitude >= alt_min (a "circle" around the zenith in the camera model).

    Parameters
    ----------
    shape : (int, int)
        Image shape (rows, columns).
    params : sequence of 5 floats
        Camera model.
    alt_min : float
        Lowest altitude kept [deg].

    Returns
    -------
    numpy.ndarray of bool
    """
    if not 0 <= alt_min < 90:
        raise ValueError("alt_min must be in [0, 90)")
    yy, xx = np.indices(shape)
    alt, _ = xy_to_altaz(xx, yy, params)
    return alt >= alt_min


def moon_mask(shape, params, moon_alt, moon_az, radius):
    """Boolean mask of the pixels within an angular radius of the Moon (True = near the Moon); works for any
    direction (moon_alt, moon_az), see also sky_disk.

    Parameters
    ----------
    shape : (int, int)
        Image shape (rows, columns).
    params : sequence of 5 floats
        Camera model.
    moon_alt, moon_az : float
        Moon position [deg].
    radius : float
        Angular radius [deg].

    Returns
    -------
    numpy.ndarray of bool
        All False when the Moon is more than `radius` below the horizon.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    yy, xx = np.indices(shape)
    alt, az = xy_to_altaz(xx, yy, params)
    a1, a2 = np.radians(alt), np.radians(moon_alt)
    cos_sep = np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(np.radians(az - moon_az))
    return cos_sep >= np.cos(np.radians(radius))


# ---------------------------------------------------------------- noise

def _robust_sigma(a):
    """1.4826 * median absolute deviation (NaN for an empty array)."""
    if a.size == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(a - np.median(a)))


def _pixel_noise(img, region):
    """Pixel-to-pixel noise in a region: std of horizontal neighbour differences / sqrt(2), after dropping
    differences more than 5 robust sigmas from the median (stars, edges).

    The robust sigma alone is not used because on integer pixels it comes out in coarse steps. JPEG compression
    correlates neighbouring pixels, so this underestimates the true noise; use it as a relative scale.
    """
    both = region[:, 1:] & region[:, :-1]
    d = (img[:, 1:] - img[:, :-1])[both]
    if d.size == 0:
        return np.nan
    med = np.median(d)
    keep = np.abs(d - med) <= 5 * max(_robust_sigma(d), 1.0)
    return np.std(d[keep]) / np.sqrt(2)


def _smooth_in(img, region, size):
    """Box mean of img over `size` x `size` pixels using only the pixels of `region` (normalised convolution)."""
    w = ndimage.uniform_filter(region.astype(float), size)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ndimage.uniform_filter(np.where(region, img, 0.0), size) / w


# ---------------------------------------------------------------- stars: measurement, zero point, clear-sky expectation

def star_detections(img, t, params, valid, stars, v_max=6.0, zp_ref=None, v_ref=6.0, k=5.0, bg_max=230):
    """Per-star measurement of the catalogue stars that fall on `valid` pixels: position, amplitude, detection and
    (with zp_ref) whether a clear sky would show the star.

    Parameters
    ----------
    img : numpy.ndarray
        2-D frame.
    t : astropy.time.Time
        Frame time (from the file name, frame_time).
    params : sequence of 5 floats
        Camera model.
    valid : numpy.ndarray of bool
        Pixels where stars count (sky mask minus the Moon, for example).
    stars : (ra, dec, v)
        Catalogue arrays (load_bsc).
    v_max : float
        Only stars brighter than this.
    zp_ref : float or None
        Clear-sky zero point (zp_clear); None = no `expected`.
    v_ref, k, bg_max : float
        A star is expected (the first, zp_clear-based expectation, kept for allsky_calib.py) when V < v_ref, its
        expected amplitude 10**(-0.4 * (V - zp_ref)) exceeds k local sigmas and its local background is below bg_max;
        detected when its amplitude exceeds k sigmas.

    Returns
    -------
    dict of numpy.ndarray
        idx (index into the catalogue arrays), x, y, alt, az, v, amp, sigma, peak, detected, and expected (all False
        without zp_ref); one entry per star used.
    """
    if img.shape != valid.shape:
        raise ValueError("img and valid must have the same shape")
    ra, dec, v = stars
    alt, az = radec_to_altaz(ra, dec, t)
    x, y = altaz_to_xy(alt, az, params)
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    on = (alt > 0) & (v < v_max) & (xi >= 0) & (xi < img.shape[1]) & (yi >= 0) & (yi < img.shape[0])
    on[on] = valid[yi[on], xi[on]]
    amp, sig, peak = star_amplitudes(img, x[on], y[on])
    with np.errstate(invalid="ignore", over="ignore"):
        detected = amp > k * sig
        expected = np.zeros(on.sum(), bool) if zp_ref is None else (v[on] < v_ref) & (peak - amp < bg_max) & (10 ** (-0.4 * (v[on] - zp_ref)) > k * sig)
    return dict(idx=np.flatnonzero(on), x=x[on], y=y[on], alt=alt[on], az=az[on], v=v[on], amp=amp, sigma=sig, peak=peak, detected=detected, expected=expected)


STAR_EXP = 1.5     # a star is expected when its predicted amplitude exceeds STAR_EXP x the detection threshold
STAR_CONF = 3.0    # ... and sure to be seen on a clear sky above STAR_CONF x the threshold
CALIB_NPZ = os.path.expanduser("~/data/allsky/calib/allsky_calib.npz")

# The camera's tone curve (JPEG): a star on a brighter background gives a smaller amplitude, as if dimmed by RESPONSE_DIM
# [mag] at the local background RESPONSE_BKG [DN] (0 at 70 DN). Measured 2026-10-30 within frames (a constant per frame,
# airmass term 0.28 mag fixed) on clear twilight frames taken while the telescopes did sky flats (6 sessions) and on clear
# moonlit nights; the two agree within ~0.1 mag up to 160 DN. Kept flat beyond the ends. Once per camera.
RESPONSE_BKG = (20.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0)
RESPONSE_DIM = (-0.27, -0.13, 0.0, 0.16, 0.36, 0.56, 0.91, 1.25)


def response_dim(st, v_lo=2.0, v_hi=4.5, saturation=240):
    """Dimming of each star by the camera's tone curve at its local background, relative to the stars the frame's zero
    point is made of (frame_zero_point's selection), so that the zero point keeps its meaning.

    Parameters
    ----------
    st : dict
        star_detections output.
    v_lo, v_hi, saturation : float
        As in frame_zero_point.

    Returns
    -------
    numpy.ndarray
        [mag] per star, positive = shown fainter than on the zero point's background; zeros without stars.
    """
    bkg = st["peak"] - st["amp"]
    if len(bkg) == 0:
        return np.zeros(0)
    dim = np.interp(bkg, RESPONSE_BKG, RESPONSE_DIM)
    ok = st["detected"] & (st["peak"] < saturation) & (st["v"] > v_lo) & (st["v"] < v_hi) & (st["amp"] > 0)
    return dim - np.median(dim[ok] if ok.any() else dim)


def frame_zero_point(st, v_lo=2.0, v_hi=4.5, saturation=240, min_stars=20):
    """Zero point of the frame from its own stars: median of V + 2.5 log10(amplitude) over the detected stars with
    v_lo < V < v_hi and a peak below saturation.

    Unlike zp_clear it follows what the camera does in this frame (the auto exposure, the Moon, dirt on the dome),
    so a clear sky gives the stars it should; a cloud over part of the sky leaves it to the stars elsewhere, a veil
    over the whole sky lowers it (transp sees that).

    Parameters
    ----------
    st : dict
        star_detections output.
    v_lo, v_hi : float
        Magnitude range (brighter stars flatten, fainter ones are biased by the detection threshold).
    saturation : float
    min_stars : int
        Fewest stars; below that NaN.

    Returns
    -------
    float
    """
    if not v_lo < v_hi:
        raise ValueError("need v_lo < v_hi")
    ok = st["detected"] & (st["peak"] < saturation) & (st["v"] > v_lo) & (st["v"] < v_hi) & (st["amp"] > 0)
    if ok.sum() < min_stars:
        return np.nan
    return float(np.median(st["v"][ok] + 2.5 * np.log10(st["amp"][ok])))


def load_calibration(path=CALIB_NPZ):
    """Clear-sky calibration written by allsky_calib.py (None when the file does not exist).

    Returns
    -------
    dict or None
        unreliable (catalogue indices of stars a clear sky often does not show), alt_edges, az_edges, altaz_rate
        (share of the expected stars seen on a clear sky per alt x az cell, NaN = no data), dist_edges, moon_rate (the
        same against the distance from the Moon), min_rate, star_dim (per catalogue star: its median zero point minus
        the frame's on a clear sky, NaN = unknown; colour, crowding, close doubles), altaz_dim (the rest of it per
        alt x az cell: extinction, vignetting), nights.
    """
    if not os.path.exists(path):
        return None
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def calibrated_out(st, calib, moon_alt, moon_az):
    """Stars that a clear sky does not reliably show, by the calibration: unreliable stars, alt x az cells and
    distances from the Moon where fewer than calib["min_rate"] of the expected stars are seen.

    Parameters
    ----------
    st : dict
        star_detections output.
    calib : dict or None
        load_calibration output; None = nothing left out.
    moon_alt, moon_az : float
        [deg]

    Returns
    -------
    numpy.ndarray of bool
        True = leave the star out.
    """
    out = np.zeros(len(st["v"]), bool)
    if calib is None:
        return out
    out |= np.isin(st["idx"], calib["unreliable"])
    ia = np.clip(np.digitize(st["alt"], calib["alt_edges"]) - 1, 0, len(calib["alt_edges"]) - 2)
    iz = np.clip(np.digitize(st["az"] % 360, calib["az_edges"]) - 1, 0, len(calib["az_edges"]) - 2)
    out |= np.nan_to_num(calib["altaz_rate"][ia, iz], nan=1.0) < calib["min_rate"]
    if moon_alt > 0:
        a1, a2 = np.radians(moon_alt), np.radians(st["alt"])
        dist = np.degrees(np.arccos(np.clip(np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(np.radians(st["az"] - moon_az)), -1, 1)))
        idist = np.clip(np.digitize(dist, calib["dist_edges"]) - 1, 0, len(calib["dist_edges"]) - 2)
        out |= np.nan_to_num(calib["moon_rate"][idist], nan=1.0) < calib["min_rate"]
    return out


def star_expectation(st, zp, out=None, v_ref=6.0, k=5.0, bg_max=230, exp=STAR_EXP, conf=STAR_CONF, dim=None):
    """Which stars a clear sky would show in this frame, against the frame's own zero point and its clear-sky noise.

    The predicted amplitude 10**(-0.4 * (V - zp)) is compared with k times the lower quartile of the local sigmas of
    the frame (the clear-sky noise), not with the star's own local sigma: under a bright textured cloud the local sigma
    grows and the star would otherwise stop being expected, so the cloud would hide itself (the median is not used:
    a cloud over half of the sky raises it). A star whose local background is too
    bright to measure (>= bg_max) counts as expected and not seen for the same reason. With `dim` the prediction is
    dimmed by the camera's tone curve at the star's background (response_dim), so that the Moon's or the twilight's
    bright sky does not make clear-sky stars look missing.

    Parameters
    ----------
    st : dict
        star_detections output.
    zp : float
        Zero point (frame_zero_point, or zp_clear when the frame has too few stars).
    out : numpy.ndarray of bool or None
        Stars to leave out (calibrated_out).
    v_ref, k, bg_max : float
        As in star_detections.
    exp, conf : float
        Expected above exp x the threshold, sure above conf x the threshold.
    dim : numpy.ndarray or None
        Per-star dimming [mag] added to the prediction (response_dim); None = none.

    Returns
    -------
    dict of numpy.ndarray of bool
        expected, sure (subset of expected), seen (detected and measurable).
    """
    if not 0 < exp <= conf:
        raise ValueError("need 0 < exp <= conf")
    n = len(st["v"])
    use = np.ones(n, bool) if out is None else ~out
    if n == 0 or not np.isfinite(zp):
        z = np.zeros(n, bool)
        return dict(expected=z, sure=z.copy(), seen=z.copy())
    with np.errstate(invalid="ignore", over="ignore"):
        noise = np.nanpercentile(st["sigma"][st["v"] < v_ref], 25) if (st["v"] < v_ref).any() else np.nan
        ratio = 10 ** (-0.4 * (st["v"] - zp + (0.0 if dim is None else dim))) / (k * noise)
        hidden = (st["v"] < v_ref) & (st["peak"] - st["amp"] >= bg_max)
        expected = use & (st["v"] < v_ref) & ((ratio > exp) | hidden)
        sure = use & (st["v"] < v_ref) & ((ratio > conf) | (hidden & (ratio > exp)))
    return dict(expected=expected, sure=sure, seen=st["detected"] & ~hidden)


# ---------------------------------------------------------------- sky state

# Limits of the per-frame sky state (version 1, 2026-10-28, from the study caches of then, now in ~/data/allsky/attic). transp in magnitudes
# (median of the last 3 frames), diff_ls_n of the whole sky (median of the last 2 frames).
# thin_transp -0.25: 0.2 % of the frames of two clear whole nights below it, 44 % of the 2026-10-07 thin veil;
# thin_diff 1.0: 95 % of frames with transp > -0.2 have diff_ls_n < 0.67.
# thin_transp_moon -0.40 (was -0.15 until 2026-10-29): the thin limit moves to thin_transp + thin_transp_moon * moon_x
# (-0.65 for a full Moon), because at a bright Moon the clear-sky zero point is off by up to ~0.3 mag from night to
# night at the same Moon (frames clear by the telescopes' zero points, 2026-10: +0.15 on 10-01, -0.05 on 10-25, -0.33
# on 10-27, -0.29 on 10-29); clear moonlit frames called thin fell from 43 to 28 of 255, cloudy ones all still caught.
STATE_LIMITS = dict(thin_transp=-0.25, thin_transp_moon=-0.40, cloudy_transp=-0.8, overcast_transp=-1.5, thin_diff=1.0, cloudy_diff=3.0)
SKY_STATES = ("clear", "thin", "cloudy", "overcast")


def sky_state(transp, diff_ls_n, limits=STATE_LIMITS, moon=0.0):
    """Sky state from the whole-sky transparency and the large-scale change between frames.

    - overcast: transp NaN (too few stars for a zero point) or transp < overcast_transp,
    - cloudy: transp < cloudy_transp or diff_ls_n > cloudy_diff,
    - thin: transp < thin_transp + thin_transp_moon * moon or diff_ls_n > thin_diff (thin / partial / moving cloud),
    - clear: otherwise.

    The stars decide how much light is lost; the difference adds moving cloud, which it sees earlier and which the
    stars may miss (cirrus). Pass smoothed values (see STATE_LIMITS).

    Parameters
    ----------
    transp : float or array_like
        zp_frame - zp_clear [mag], NaN when there is no zero point.
    diff_ls_n : float or array_like
        Whole-sky diff_ls / diff_noise (NaN = unknown, e.g. no previous frame: then only transp decides).
    limits : dict
        Keys as in STATE_LIMITS.
    moon : float or array_like
        moon_x of the frame (0 = no Moon); widens the thin limit.

    Returns
    -------
    str or numpy.ndarray of str
        One of SKY_STATES.
    """
    missing = set(STATE_LIMITS) - set(limits)
    if missing:
        raise ValueError(f"limits lack {sorted(missing)}")
    if not limits["overcast_transp"] < limits["cloudy_transp"] < limits["thin_transp"] + min(limits["thin_transp_moon"], 0):
        raise ValueError("need overcast_transp < cloudy_transp < thin_transp (also with the Moon term)")
    if not limits["thin_diff"] < limits["cloudy_diff"]:
        raise ValueError("need thin_diff < cloudy_diff")
    tr = np.asarray(transp, dtype=float)
    d = np.nan_to_num(np.asarray(diff_ls_n, dtype=float), nan=0.0)
    thin = limits["thin_transp"] + limits["thin_transp_moon"] * np.asarray(moon, dtype=float)
    with np.errstate(invalid="ignore"):
        state = np.select([np.isnan(tr) | (tr < limits["overcast_transp"]),
                           (tr < limits["cloudy_transp"]) | (d > limits["cloudy_diff"]),
                           (tr < thin) | (d > limits["thin_diff"])],
                          ["overcast", "cloudy", "thin"], "clear")
    return state.item() if state.ndim == 0 else state


# ---------------------------------------------------------------- frame selection, catalogue

FRAME_SHAPE = (1256, 1240)
BSC_CSV = os.path.expanduser("~/data/allsky/bsc5.csv")
BSC_URL = "https://raw.githubusercontent.com/brettonw/YaleBrightStarCatalog/master/bsc5-short.json"


def fetch_bsc(path=BSC_CSV, url=BSC_URL):
    """Download the Yale Bright Star Catalogue (JSON copy on GitHub, J2000) and write it as the CSV that load_bsc reads.

    Does nothing when `path` already exists.

    Parameters
    ----------
    path : str
        Output CSV (columns hr, ra, dec, v).
    url : str
        Source JSON.

    Returns
    -------
    str
        path
    """
    import json
    import re
    import urllib.request
    if os.path.exists(path):
        return path
    rows = []
    for s in json.loads(urllib.request.urlopen(url, timeout=60).read()):
        if "V" not in s or "RA" not in s:
            continue
        h, m, sec = map(float, re.findall(r"[\d.]+", s["RA"]))
        d, dm, ds = map(float, re.findall(r"[\d.]+", s["Dec"]))
        sign = -1 if s["Dec"].strip().startswith("-") else 1
        rows.append(f"{int(s['HR'])},{15 * (h + m / 60 + sec / 3600):.5f},{sign * (d + dm / 60 + ds / 3600):.5f},{float(s['V'])}")
    if not rows:
        raise ValueError(f"no stars read from {url}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("hr,ra,dec,v\n" + "\n".join(rows) + "\n")
    return path


def frame_index(paths, hhmm):
    """Index of the first frame at or after UT time "HHMM" in a sorted list of frame paths of one day.

    Parameters
    ----------
    paths : list of str
        From day_frames.
    hhmm : str
        "HHMM".

    Returns
    -------
    int
    """
    if len(hhmm) != 4 or not hhmm.isdigit():
        raise ValueError('hhmm must be "HHMM"')
    for k, p in enumerate(paths):
        if os.path.basename(p)[9:13] >= hhmm:
            return k
    raise ValueError(f"no frame at or after {hhmm}")


def dark_frames(day, sun_max=-18.0, ut_max="1100", allsky_dir=ALLSKY_DIR):
    """Frame paths of the morning part (00 - ut_max UT) of a day directory taken with the Sun below sun_max.

    The evening frames of a UT day belong to the next night, hence ut_max.

    Parameters
    ----------
    day : str
        "YYYYMMDD".
    sun_max : float
        [deg]
    ut_max : str
        "HHMM".
    allsky_dir : str

    Returns
    -------
    list of str
    """
    paths = [p for p in day_frames(day, allsky_dir) if os.path.basename(p)[9:13] < ut_max]
    if not paths:
        return []
    sun_alt, _ = body_altaz("sun", Time([frame_time(p).jd for p in paths], format="jd", scale="utc"))
    return [p for p, a in zip(paths, np.atleast_1d(sun_alt)) if a < sun_max]


# ---------------------------------------------------------------- frame analysis: indicators and cloud regions

MOON_GLARE = (15.0, 15.0)  # deg; the stars leave out MOON_GLARE[0] + MOON_GLARE[1] * moon_flux around the Moon
#                            (30 deg at full Moon: within 25 deg of a full Moon 0-18 % of the expected stars are seen,
#                            at 25-30 deg ~50 %, beyond 30 deg 70-94 %; 10 clear full-Moon nights of 2026)
MOON_RADIUS_PIX = 25.0     # deg; the difference leaves out this much around the Moon (the glare spoils it)
MOON_MARGIN = 10.0         # deg; the star maps leave out this much more than the glare (its rim)
SMOOTH_PX = 15             # px; box of the large-scale difference behind diff_ls_n
ZEN_ALT = 60.0             # deg; the zenith cap of zen_star_seen
VEIL_SEEN = 0.8            # a veil: light lost (transp below the thin limit) while at least this share of the stars is still seen
#                            (frames with transp that low and star_seen < 0.8 are clouds the maps already show: 71 % of the sky
#                            marked at star_seen < 0.5, ~1 % at > 0.8; ten nights of 2026-10)

# Region maps are made on a grid of every GRID-th pixel (the lens is close to equidistant, ~6.2 px/deg above 30 deg).
PX_PER_DEG = 6.2
GRID = 4
N_KERNEL = 12.0            # star maps: expected stars within the Gaussian kernel (2 pi sigma^2) ...
SIGMA_RANGE = (4.0, 12.0)  # ... its width clipped to this [deg]
MIN_W = 4.0                # effective number of expected stars for a known point of the map
MIN_W_SURE = 3.0           # ... of sure stars
STAR_POSSIBLE = 0.5        # possible cloud: share of the expected stars missing above this
STAR_SURE = 0.9            # sure (opaque) cloud: share of the sure stars missing above this (none seen, up to a stray one)
CIRRUS_DIM = -0.4          # mag; possible cloud also where the seen stars are dimmed by more than this
DIFF_SIGMA = 10.0          # px; Gaussian smoothing of the difference for its regions
HI, LO = 3.0, 1.5          # difference regions, possible: seeds above HI grown down to LO (pixel noise units)
HI_SURE, LO_SURE = 6.0, 3.0  # ... sure: (a strong change, not necessarily opaque: used for red only where the stars cannot tell)
CLOSE_DEG = 3.0            # deg; gaps narrower than this are closed
MIN_AREA = 20.0            # deg^2; smaller regions are dropped


def moon_masks(t, params=CAMERA_PARAMS, shape=FRAME_SHAPE):
    """The parts of the frame near the Moon that the two methods leave out at time t.

    Parameters
    ----------
    t : astropy.time.Time
        Frame time.
    params : sequence of 5 floats
        Camera model.
    shape : (int, int)
        Frame shape.

    Returns
    -------
    stars : numpy.ndarray of bool
        Left out of the star maps: the glare (MOON_GLARE, grows with the phase) plus MOON_MARGIN.
    diff : numpy.ndarray of bool
        Left out of the difference: MOON_RADIUS_PIX. Both all False with the Moon well below the horizon.
    """
    moon_alt, moon_az = body_altaz("moon", t)
    glare = MOON_GLARE[0] + MOON_GLARE[1] * float(moon_flux(np.clip(moon_illumination(t), 0, 1)))
    return moon_mask(shape, params, moon_alt, moon_az, glare + MOON_MARGIN), moon_mask(shape, params, moon_alt, moon_az, MOON_RADIUS_PIX)


def _gauss_sum(yi, xi, w, shape, s):
    """Sum of weights w at grid cells (yi, xi), Gaussian-smoothed with sigma s cells, peak weight 1."""
    a = np.zeros(shape)
    np.add.at(a, (yi, xi), w)
    return ndimage.gaussian_filter(a, s) * 2 * np.pi * s ** 2


def star_dim_correction(st, calib):
    """Clear-sky dimming of each star by the calibration (its own median plus its alt x az cell), 0 where unknown or
    without calibration."""
    if calib is None or "star_dim" not in calib:
        return np.zeros(len(st["v"]))
    sd = calib["star_dim"]
    own = np.where(st["idx"] < len(sd), sd[np.clip(st["idx"], 0, len(sd) - 1)], np.nan)
    ia = np.clip(np.digitize(st["alt"], calib["alt_edges"]) - 1, 0, len(calib["alt_edges"]) - 2)
    iz = np.clip(np.digitize(st["az"] % 360, calib["az_edges"]) - 1, 0, len(calib["az_edges"]) - 2)
    return np.nan_to_num(own) + np.nan_to_num(calib["altaz_dim"][ia, iz])


def star_maps(st, valid, zp, calib=None):
    """Star maps on the grid: around each grid point, Gaussian-weighted by the distance, the share of the expected stars
    missing, the share of the sure stars missing and the mean dimming of the seen stars. The kernel width follows the
    number of expected stars, so that about N_KERNEL of them fall within it (clipped to SIGMA_RANGE).

    Parameters
    ----------
    st : dict
        star_detections output with out, expected_f, sure_f and seen (as SkyAnalyzer.analyse makes it).
    valid : numpy.ndarray of bool
        Full-resolution pixels where stars count (sky minus the Moon glare and margin).
    zp : float
        The zero point the expectation used.
    calib : dict or None
        Clear-sky calibration (load_calibration): the dimming is read against each star's clear-sky dimming.

    Returns
    -------
    maps : dict of numpy.ndarray
        missing, missing_sure, dim (mean zero point of the seen stars 2 < V < 4.5 minus zp, less their clear-sky dimming
        and their tone-curve dimming st["resp"] when present, clipped to +-1.5 mag); NaN where unknown (fewer than MIN_W expected / MIN_W_SURE sure or clean stars carry the weight).
    sigma : float
        Kernel width used [deg].
    n : int
        Expected stars.
    """
    for key in ("out", "expected_f", "sure_f", "seen"):
        if key not in st:
            raise ValueError(f"st lacks {key} (made by SkyAnalyzer.analyse)")
    g = valid[::GRID, ::GRID]
    nan = np.full(g.shape, np.nan)
    yi = np.clip(np.round(st["y"] / GRID).astype(int), 0, g.shape[0] - 1)
    xi = np.clip(np.round(st["x"] / GRID).astype(int), 0, g.shape[1] - 1)
    on = g[yi, xi]
    exp, sure, seen = st["expected_f"] & on, st["sure_f"] & on, st["seen"]
    n = int(exp.sum())
    if n == 0 or not np.isfinite(zp):
        return dict(missing=nan, missing_sure=nan.copy(), dim=nan.copy()), np.nan, n
    s = np.clip(np.sqrt(N_KERNEL / (2 * np.pi * n / g.sum())), SIGMA_RANGE[0] * PX_PER_DEG / GRID, SIGMA_RANGE[1] * PX_PER_DEG / GRID)
    gs = lambda sel, w=1.0: _gauss_sum(yi[sel], xi[sel], w if np.isscalar(w) else w[sel], g.shape, s)
    we, wm = gs(exp), gs(exp & ~seen)
    ws, wsm = gs(sure), gs(sure & ~seen)
    with np.errstate(invalid="ignore", divide="ignore"):
        clean = on & seen & ~st["out"] & (st["peak"] < 240) & (st["v"] > 2) & (st["v"] < 4.5) & (st["amp"] > 0)
        resp = st["resp"] if "resp" in st else 0.0      # dimmed by the tone curve: back up by resp
        dim = np.clip(st["v"] + 2.5 * np.log10(np.clip(st["amp"], 1e-3, None)) - zp - star_dim_correction(st, calib) + resp, -1.5, 1.5)
        wa, wd = gs(clean), gs(clean, dim)
        maps = dict(missing=np.where(g & (we >= MIN_W), wm / we, np.nan), missing_sure=np.where(g & (ws >= MIN_W_SURE), wsm / ws, np.nan),
                    dim=np.where(g & (wa >= MIN_W_SURE), wd / wa, np.nan))
    return maps, float(s * GRID / PX_PER_DEG), n


def diff_map(d, valid):
    """The difference on the grid: Gaussian-smoothed (DIFF_SIGMA) over the pixel noise of d, signed.

    Parameters
    ----------
    d : numpy.ndarray
        Full-resolution derotated difference, zero point removed.
    valid : numpy.ndarray of bool
        Pixels to use (sky minus the Moon).

    Returns
    -------
    zs : numpy.ndarray
        Smoothed difference / pixel noise, 0 outside valid.
    noise : float
        Pixel noise of d.
    """
    if d.shape != valid.shape:
        raise ValueError("d and valid must have the same shape")
    valid = valid & np.isfinite(d)
    noise = _pixel_noise(np.nan_to_num(d), valid)
    w = ndimage.gaussian_filter(valid.astype(float), DIFF_SIGMA)
    with np.errstate(invalid="ignore", divide="ignore"):
        ds = ndimage.gaussian_filter(np.where(valid, d, 0.0), DIFF_SIGMA) / w
    g = valid[::GRID, ::GRID]
    return np.where(g, ds[::GRID, ::GRID] / noise, 0.0), noise


def regions(z, g, hi=HI, lo=LO):
    """Regions of a non-negative map on the grid: seeds above hi grown down to lo (hysteresis), closed over CLOSE_DEG,
    holes filled, kept within g, components above MIN_AREA deg^2."""
    lab, n = ndimage.label(z > lo)
    if n == 0:
        return np.zeros(g.shape, bool)
    seeded = np.zeros(n + 1, bool)
    seeded[1:] = ndimage.maximum(z, lab, np.arange(1, n + 1)) > hi
    reg = seeded[lab]
    r = int(round(CLOSE_DEG * PX_PER_DEG / GRID))
    disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    reg = ndimage.binary_fill_holes(ndimage.binary_closing(reg, disk)) & g
    lab, n = ndimage.label(reg)
    area = ndimage.sum(np.ones(g.shape), lab, np.arange(1, n + 1)) * (GRID / PX_PER_DEG) ** 2
    keep = np.zeros(n + 1, bool)
    keep[1:] = area >= MIN_AREA
    return keep[lab]


def detections(maps, zs, g, veil=False):
    """The cloud detections on the grid, per method and as the two marks for the observers.

    Parameters
    ----------
    maps : dict of numpy.ndarray
        Star maps (star_maps), NaN = unknown.
    zs : numpy.ndarray
        Signed difference map (diff_map).
    g : numpy.ndarray of bool
        Grid points of the sky.
    veil : bool
        A veil over the whole sky (StateTracker): the whole sky is possible cloud.

    Returns
    -------
    dict of numpy.ndarray of bool
        star_possible, star_sure, star_cirrus (star_possible, not sure, sure stars known), diff_possible, diff_sure, and
        the marks: possible (yellow: either method, or the veil) and sure (red, a cloud that hides everything: star_sure,
        or diff_sure where the star map cannot tell -- where the sure stars are known, a strong difference is opaque in
        only ~36 % of cases).
    """
    if any(m.shape != g.shape for m in maps.values()) or zs.shape != g.shape:
        raise ValueError("maps, zs and g must have the same shape")
    miss, miss_sure, dim = (np.nan_to_num(maps[k], nan=v) for k, v in (("missing", -1.0), ("missing_sure", -1.0), ("dim", 0.0)))
    sure = (miss_sure > STAR_SURE) & g
    possible = ((miss > STAR_POSSIBLE) | (dim < CIRRUS_DIM) | sure) & g
    cirrus = possible & ~sure & np.isfinite(maps["missing_sure"])
    z = np.abs(zs)
    det = dict(star_possible=possible, star_sure=sure, star_cirrus=cirrus, diff_possible=regions(z, g), diff_sure=regions(z, g, HI_SURE, LO_SURE))
    det["sure"] = det["star_sure"] | (det["diff_sure"] & ~np.isfinite(maps["missing_sure"]))
    det["possible"] = det["star_possible"] | det["diff_possible"] | det["sure"] | (g if veil else np.zeros(g.shape, bool))
    return det


def det_areas(det, g):
    """Sky share of each detection mask."""
    return {k: float(v[g].mean()) for k, v in det.items()}


class SkyAnalyzer:
    """One frame: the whole-sky indicators and the cloud maps.

    - transp: the frame's zero point (stars V < 5) minus zp_clear(Moon) [mag] -- how much light is lost over the whole
      sky, including a veil that dims it all alike,
    - star_seen, zen_star_seen: share of the stars a clear sky would show that are seen (whole sky above 30 deg, the
      zenith cap above ZEN_ALT); the expectation from the frame's own zero point (stars 2 < V < 4.5) against its
      clear-sky noise, without the stars the calibration finds unreliable,
    - diff_ls_n: large-scale change between the frame and the derotated previous one over the pixel noise,
    - the star maps and the difference map with their detections (detections, without the veil: StateTracker decides
      that over several frames).

    Parameters
    ----------
    stars : (ra, dec, v) or None
        Catalogue; None = load_bsc(fetch_bsc()).
    params : sequence of 5 floats
        Camera model.
    shape : (int, int)
        Frame shape.
    calib : dict or None
        Clear-sky calibration (load_calibration); None = all stars used (more false yellow on the low edge and next to
        the Moon).
    zp_coeffs : (float, float)
        zp_clear coefficients (ZP_CLEAR, or learned with learn_zp_clear; may be replaced between frames).
    """

    def __init__(self, stars=None, params=CAMERA_PARAMS, shape=FRAME_SHAPE, calib=None, zp_coeffs=ZP_CLEAR):
        self.stars = stars if stars is not None else load_bsc(fetch_bsc())
        self.params = params
        self.shape = shape
        self.calib = calib
        self.zp_coeffs = zp_coeffs
        self.sky = sky_mask(shape, params, 30.0)
        self.g = self.sky[::GRID, ::GRID]

    def analyse(self, img, t, prev=None, dt=None, details=False):
        """Indicators and cloud maps of one frame.

        Parameters
        ----------
        img : numpy.ndarray
            2-D frame (read_frame).
        t : astropy.time.Time
            Frame time from the file name (frame_time).
        prev : numpy.ndarray or None
            Previous frame, for the difference.
        dt : float or None
            img time minus prev time [s], from the mtimes (frame_mtime).
        details : bool
            Also return the per-star measurements (`stars`) and the derotated difference image (`diff_image`, zero point
            removed, NaN outside the sky; None without prev).

        Returns
        -------
        dict
            moon_alt, moon_illum, moon_x, sun_alt, zp_ref (zp_clear), zp_frame, transp, zp_stars, star_seen,
            zen_star_seen, diff_ls_n (NaN without prev), n_exp (expected stars), sigma (star map kernel [deg]), maps
            (star_maps), zs (diff_map, zeros without prev), diff_noise, det (detections), moon_stars, moon_diff
            (moon_masks), and with details: stars, diff_image.
        """
        if img.shape != self.shape:
            raise ValueError(f"frame shape {img.shape}, analyser built for {self.shape}")
        if (prev is None) != (dt is None):
            raise ValueError("give both prev and dt, or neither")
        moon_alt, moon_az = body_altaz("moon", t)
        moon_alt, moon_az = float(moon_alt), float(moon_az)
        sun_alt = float(body_altaz("sun", t)[0])
        moon_illum = float(moon_illumination(t))
        zp_ref = float(zp_clear(moon_alt, moon_illum, self.zp_coeffs))
        glare = MOON_GLARE[0] + MOON_GLARE[1] * float(moon_flux(np.clip(moon_illum, 0, 1)))
        moon_stars = moon_mask(self.shape, self.params, moon_alt, moon_az, glare + MOON_MARGIN)
        moon_diff = moon_mask(self.shape, self.params, moon_alt, moon_az, MOON_RADIUS_PIX)

        st = star_detections(img, t, self.params, self.sky & ~moon_mask(self.shape, self.params, moon_alt, moon_az, glare), self.stars, zp_ref=zp_ref)
        zp_frame = frame_zero_point(st, v_lo=-np.inf, v_hi=5.0, min_stars=5)
        zp_stars = frame_zero_point(st)
        zp = zp_stars if np.isfinite(zp_stars) else zp_ref
        st["out"] = calibrated_out(st, self.calib, moon_alt, moon_az)
        st["resp"] = response_dim(st)
        e = star_expectation(st, zp, st["out"], dim=st["resp"])
        st["expected_f"], st["sure_f"], st["seen"] = e["expected"], e["sure"], e["seen"]
        seen_share = lambda sel: float(e["seen"][sel & e["expected"]].mean()) if (sel & e["expected"]).sum() >= 5 else np.nan
        maps, sigma, n_exp = star_maps(st, self.sky & ~moon_stars, zp, self.calib)

        diff_ls_n, zs, noise, diff_image = np.nan, np.zeros(self.g.shape), np.nan, None
        if prev is not None:
            d = img - derotate(prev, dt, self.params)
            valid = self.sky & ~moon_diff & np.isfinite(d)
            if valid.sum() >= 100:
                d0 = d - np.median(d[valid])
                ls = np.std(_smooth_in(np.nan_to_num(d0), valid, SMOOTH_PX)[valid])
                pix = _pixel_noise(d, valid)
                diff_ls_n = ls / pix if pix > 0 else np.nan   # zero noise: saturated
            diff_image = np.where(self.sky, d - np.nanmedian(d[self.sky]), np.nan)
            zs, noise = diff_map(diff_image, self.sky & ~moon_diff)

        out = dict(moon_alt=moon_alt, moon_illum=moon_illum, moon_x=float(moon_x(moon_alt, moon_illum)), sun_alt=sun_alt, zp_ref=zp_ref,
                   zp_frame=zp_frame, transp=zp_frame - zp_ref, zp_stars=zp_stars, star_seen=seen_share(np.ones(len(st["v"]), bool)),
                   zen_star_seen=seen_share(st["alt"] >= ZEN_ALT), diff_ls_n=float(diff_ls_n), n_exp=n_exp, sigma=sigma, maps=maps, zs=zs,
                   diff_noise=noise, det=detections(maps, zs, self.g), moon_stars=moon_stars, moon_diff=moon_diff)
        if details:
            out["stars"] = st
            out["diff_image"] = diff_image
        return out


class StateTracker:
    """Sky state in time: smooths transp and star_seen (median of the last n_transp frames, NaN when most of them have
    none) and diff_ls_n (median of the last n_diff), applies sky_state and says whether a veil covers the whole sky: the
    smoothed transp below the thin limit while the smoothed star_seen is at least VEIL_SEEN (the light is lost but the
    stars are still seen everywhere -- a uniform veil, which the star maps cannot show). Starts afresh after a gap.

    Parameters
    ----------
    n_transp, n_diff : int
        Smoothing lengths in frames (STATE_LIMITS were set with 3 and 2).
    limits : dict
        As STATE_LIMITS.
    sun_max : float
        Above this Sun altitude [deg] the result is marked twilight (zp_clear has no Sun term).
    max_gap : float
        A gap longer than this [s] between frames resets the history.
    """

    def __init__(self, n_transp=3, n_diff=2, limits=STATE_LIMITS, sun_max=-16.0, max_gap=600.0):
        if n_transp < 1 or n_diff < 1:
            raise ValueError("smoothing lengths must be >= 1")
        self.n_transp, self.n_diff, self.limits, self.sun_max, self.max_gap = n_transp, n_diff, limits, sun_max, max_gap
        self.transp, self.seen, self.diff, self.last_t = [], [], [], None

    def update(self, t, transp, diff_ls_n, sun_alt=np.nan, moon=0.0, star_seen=np.nan):
        """Add one frame and return the smoothed values, the state and the veil.

        Parameters
        ----------
        t : astropy.time.Time
            Frame time.
        transp, diff_ls_n : float
            From SkyAnalyzer.analyse (NaN allowed).
        sun_alt : float
            [deg], for the twilight mark.
        moon : float
            moon_x of the frame (SkyAnalyzer.analyse gives it), for the thin limit.
        star_seen : float
            From SkyAnalyzer.analyse, for the veil (NaN = no veil).

        Returns
        -------
        dict
            state, transp_s, diff_s, star_seen_s, twilight, veil.
        """
        if self.last_t is not None and (t - self.last_t).sec > self.max_gap:
            self.transp, self.seen, self.diff = [], [], []
        self.last_t = t
        self.transp = (self.transp + [transp])[-self.n_transp:]
        self.seen = (self.seen + [star_seen])[-self.n_transp:]
        self.diff = (self.diff + [diff_ls_n])[-self.n_diff:]
        smooth = lambda a: np.nanmedian(a) if np.isfinite(a).sum() > len(a) / 2 else np.nan
        transp_s, seen_s = smooth(np.array(self.transp, dtype=float)), smooth(np.array(self.seen, dtype=float))
        d = np.array(self.diff, dtype=float)
        diff_s = np.nanmedian(d) if np.isfinite(d).any() else np.nan
        with np.errstate(invalid="ignore"):
            veil = bool(transp_s < self.limits["thin_transp"] + self.limits["thin_transp_moon"] * moon and seen_s >= VEIL_SEEN)
        return dict(state=sky_state(transp_s, diff_s, self.limits, moon), transp_s=transp_s, diff_s=diff_s, star_seen_s=seen_s,
                    twilight=bool(sun_alt > self.sun_max), veil=veil)


# ---------------------------------------------------------------- pictures for display (numpy only, frame geometry)

def _grey(img, sky):
    """The frame as an RGB uint8 picture, stretched on the sky (1 - 99.5 percentile)."""
    lo, hi = np.percentile(img[sky], [1, 99.5])
    g = np.clip((img - lo) / max(hi - lo, 1.0) * 255, 0, 255).astype(np.uint8)
    return np.repeat(g[:, :, None], 3, axis=2)


def _full(mask, shape):
    """A grid mask enlarged GRID x to the frame shape."""
    return np.repeat(np.repeat(mask, GRID, 0), GRID, 1)[:shape[0], :shape[1]]


def _fill(rgb, mask, colour, alpha):
    """Blend `colour` into rgb over a grid mask (enlarged to the frame) with opacity alpha."""
    full = _full(mask, rgb.shape)
    rgb[full] = ((1 - alpha) * rgb[full] + alpha * np.array(colour)).astype(np.uint8)


def _outline(rgb, mask, colour, width=2):
    """Paint the border of a grid mask (enlarged GRID x to the frame) onto rgb, `width` px wide."""
    if not mask.any():
        return
    full = _full(mask, rgb.shape)
    edge = full & ~ndimage.binary_erosion(full, iterations=width)
    rgb[edge] = colour


def render_stars(img, st, det, sky, moon):
    """Picture of the star method on the frame itself, as big as the frame: the expected stars ringed, green seen, red
    missing (larger rings = the sure ones), the star-map regions outlined (yellow possible, red sure) and the Moon mask
    as a light cyan area, drawn as given (e.g. clipped to above the horizon, not to the sky analysed).

    Parameters
    ----------
    img : numpy.ndarray
        The frame.
    st : dict
        Stars of SkyAnalyzer.analyse(details=True).
    det : dict
        detections.
    sky : numpy.ndarray of bool
        Full-resolution sky (SkyAnalyzer.sky).
    moon : numpy.ndarray of bool
        Full-resolution Moon area to draw (the star map's Moon mask, moon_stars, or a part of it).

    Returns
    -------
    numpy.ndarray
        RGB uint8, frame shape.
    """
    rgb = _grey(img, sky)
    _fill(rgb, moon[::GRID, ::GRID], (0, 220, 230), 0.3)
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    for sel, r in ((st["expected_f"] & ~st["sure_f"], 4), (st["sure_f"], 6)):
        for x, y, seen in zip(st["x"][sel], st["y"][sel], st["seen"][sel]):
            xi, yi = int(round(x)), int(round(y))
            box = (slice(max(yi - r - 1, 0), yi + r + 2), slice(max(xi - r - 1, 0), xi + r + 2))
            d2 = (yy[box[0]] - yi) ** 2 + (xx[:, box[1]] - xi) ** 2
            rgb[box][(d2 >= (r - 1) ** 2) & (d2 <= (r + 0.5) ** 2)] = (0, 230, 0) if seen else (255, 40, 40)
    _outline(rgb, det["star_possible"], (255, 230, 0))
    _outline(rgb, det["star_sure"], (230, 0, 0))
    return rgb


def render_diff(img, diff, noise, det, sky, moon):
    """Picture of the difference method on the frame itself, as big as the frame: the derotated difference blended in
    colour, the more opaque the stronger the change (red = brighter than the previous frame, blue = darker; below 1.5
    pixel noises nothing, full colour at 6), the difference regions outlined (yellow possible, black sure) and the Moon
    mask as a light cyan area, drawn as given (e.g. clipped to above the horizon, not to the sky analysed).

    Parameters
    ----------
    img : numpy.ndarray
        The frame.
    diff : numpy.ndarray or None
        SkyAnalyzer.analyse(details=True)["diff_image"].
    noise : float
        Its pixel noise (diff_noise).
    det : dict
        detections.
    sky, moon : numpy.ndarray of bool
        Full-resolution sky and the Moon area to draw (the difference's Moon mask, moon_diff, or a part of it).

    Returns
    -------
    numpy.ndarray
        RGB uint8, frame shape.
    """
    rgb = _grey(img, sky)
    if diff is not None and np.isfinite(noise) and noise > 0:
        on = sky & np.isfinite(diff)
        v = np.clip(diff[on] / noise, -6, 6)
        a = 0.85 * np.clip((np.abs(v) - 1.5) / 4.5, 0, 1)[:, None]
        colour = np.where(v[:, None] > 0, np.array([255, 30, 30]), np.array([40, 90, 255]))
        rgb[on] = ((1 - a) * rgb[on] + a * colour).astype(np.uint8)
    _fill(rgb, moon[::GRID, ::GRID], (0, 220, 230), 0.3)
    _outline(rgb, det["diff_possible"], (255, 200, 0))
    _outline(rgb, det["diff_sure"], (0, 0, 0))
    return rgb
