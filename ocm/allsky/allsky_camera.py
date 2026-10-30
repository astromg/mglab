"""Camera model fit for the allsky camera: needed once per camera (a new camera, lens or mounting), not in the service.

Measures the positions of the brightest catalogue stars on a few clear frames (star_peaks, starting from the current
model) and fits the five parameters of allsky_lib's camera model (fit_camera); prints the parameters and the residuals.
Put the result into allsky_lib.CAMERA_PARAMS. If the current model is far off (a new camera), start from a fit to the
Moon (saturated_blob_centroid on frames with the Moon up) and repeat with the stars.

    python allsky_camera.py /data/misc/allsky/20260713/20260713_0500.jpg ...   # clear, moonless frames
"""

import sys

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from allsky_lib import *

# Bright stars for checking the camera model: name -> (RA, Dec) J2000 [deg]
BRIGHT_STARS = {
    "Sirius": (101.287, -16.716), "Canopus": (95.988, -52.696), "Arcturus": (213.915, 19.182),
    "Vega": (279.235, 38.784), "Capella": (79.172, 45.998), "Rigel": (78.634, -8.202),
    "Procyon": (114.825, 5.225), "Achernar": (24.429, -57.237), "Betelgeuse": (88.793, 7.407),
    "Hadar": (210.956, -60.373), "Rigil Kent": (219.902, -60.834), "Altair": (297.696, 8.868),
    "Acrux": (186.650, -63.099), "Aldebaran": (68.980, 16.509), "Antares": (247.352, -26.432),
    "Spica": (201.298, -11.161), "Fomalhaut": (344.413, -29.622), "Deneb": (310.358, 45.280),
    "Peacock": (306.412, -56.735), "Mimosa": (191.930, -59.689),
}



def fit_camera(alt, az, x, y, params0):
    """Least-squares fit of the camera model to sky directions with measured pixel positions.

    Parameters
    ----------
    alt, az : array_like
        True altitude and azimuth [deg] of the points (e.g. the Moon at the frame times).
    x, y : array_like
        Measured pixel positions of the same points.
    params0 : sequence of 5 floats
        Starting (x0, y0, theta, k1, k2).

    Returns
    -------
    params : numpy.ndarray
        Fitted (x0, y0, theta, k1, k2).
    resid : numpy.ndarray
        Per-point distance [px] between the measured and the model position.
    """
    alt, az, x, y = (np.asarray(v, dtype=float) for v in (alt, az, x, y))
    if not (alt.shape == az.shape == x.shape == y.shape):
        raise ValueError("alt, az, x, y must have the same shape")
    if alt.size < 5:
        raise ValueError("need at least 5 points to fit 5 parameters")

    def residuals(p):
        xm, ym = altaz_to_xy(alt, az, p)
        return np.concatenate([xm - x, ym - y])

    result = least_squares(residuals, np.asarray(params0, dtype=float))
    xm, ym = altaz_to_xy(alt, az, result.x)
    return result.x, np.hypot(xm - x, ym - y)


def saturated_blob_centroid(img, near_xy, search_radius=80, threshold=245):
    """Centroid of the saturated blob nearest to an expected position (e.g. the Moon).

    Pixels >= threshold within search_radius of near_xy are grouped into connected regions; the largest one is
    taken and its centroid returned.

    Parameters
    ----------
    img : numpy.ndarray
        2-D frame.
    near_xy : (float, float)
        Expected position (x, y).
    search_radius : float
        Search radius [px].
    threshold : float
        Saturation level.

    Returns
    -------
    (x, y, n_pix) or None
        Centroid and number of pixels of the blob, None if nothing is saturated within the radius.
    """
    if img.ndim != 2:
        raise ValueError("img must be 2-D")
    if search_radius <= 0:
        raise ValueError("search_radius must be positive")
    yy, xx = np.indices(img.shape)
    near = np.hypot(xx - near_xy[0], yy - near_xy[1]) < search_radius
    labels, n = ndimage.label((img >= threshold) & near)
    if n == 0:
        return None
    sizes = ndimage.sum(np.ones_like(img), labels, index=np.arange(1, n + 1))
    best = np.argmax(sizes) + 1
    cy, cx = ndimage.center_of_mass(labels == best)
    return cx, cy, int(sizes[best - 1])


def star_peaks(img, alt, az, params, radius=12, min_contrast=60):
    """Measured pixel positions of stars of known (alt, az): the brightest pixel near each predicted position.

    A star is accepted when the window maximum stands min_contrast above the window median and does not lie on the
    window edge (an edge maximum means the star was not found inside the window). Smooth the image slightly
    beforehand (e.g. scipy.ndimage.uniform_filter(img, 3)) so that single noisy pixels do not win.

    Parameters
    ----------
    img : numpy.ndarray
        2-D frame.
    alt, az : array_like
        Star positions [deg] at the frame time.
    params : sequence of 5 floats
        Camera model used for the prediction.
    radius : int
        Half size of the search window [px].
    min_contrast : float
        Minimum (maximum - median) in the window.

    Returns
    -------
    numpy.ndarray, shape (n, 4)
        Rows (alt, az, x, y) of the found stars.
    """
    if img.ndim != 2:
        raise ValueError("img must be 2-D")
    if radius < 1:
        raise ValueError("radius must be >= 1")
    x, y = altaz_to_xy(alt, az, params)
    rows = []
    for a, z, xi, yi in zip(np.atleast_1d(alt), np.atleast_1d(az), np.atleast_1d(x), np.atleast_1d(y)):
        if a <= 0:
            continue
        x0, y0 = int(round(xi)), int(round(yi))
        if not (radius <= x0 < img.shape[1] - radius and radius <= y0 < img.shape[0] - radius):
            continue
        win = img[y0 - radius:y0 + radius + 1, x0 - radius:x0 + radius + 1]
        j, i = np.unravel_index(np.argmax(win), win.shape)
        if win.max() - np.median(win) < min_contrast or max(abs(i - radius), abs(j - radius)) == radius:
            continue
        rows.append((a, z, x0 - radius + i, y0 - radius + j))
    return np.array(rows).reshape(-1, 4)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ra, dec, v = load_bsc(BSC_CSV)
    bright = v < 2.0
    pts = []
    for path in sys.argv[1:]:
        alt, az = radec_to_altaz(ra[bright], dec[bright], frame_time(path))
        up = alt > 15
        pts.append(star_peaks(ndimage.uniform_filter(read_frame(path), 3), alt[up], az[up], CAMERA_PARAMS))
    pts = np.vstack(pts)
    params, resid = fit_camera(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], CAMERA_PARAMS)
    print(f"{len(pts)} stars; CAMERA_PARAMS = ({', '.join(f'{p:.6g}' for p in params)})")
    print(f"residuals: rms {np.sqrt(np.mean(resid ** 2)):.2f} px, above alt 30 {np.sqrt(np.mean(resid[pts[:, 0] > 30] ** 2)):.2f} px, max {resid.max():.1f} px")
