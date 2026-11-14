"""Camera model for the allsky raw FITS: grid search for a start (bright BSC stars vs detected peaks), then the
least-squares fit of allsky_camera.fit_camera on stars measured with star_peaks.

    python fits_camera.py FRAME.fits [FRAME.fits ...]     # clear, moonless, dark frames
"""
import sys
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from allsky_lib import load_bsc, radec_to_altaz, altaz_to_xy, BSC_CSV
from allsky_camera import fit_camera, star_peaks
from fits_lib import read_fits


def peaks(img, n=300, size=7):
    sm = ndimage.uniform_filter(img, 3)
    bg = ndimage.median_filter(sm[::4, ::4], 15)
    bg = ndimage.zoom(bg, 4, order=1)[:img.shape[0], :img.shape[1]]
    hp = sm - bg
    mx = ndimage.maximum_filter(hp, size) == hp
    ys, xs = np.nonzero(mx & (hp > 0))
    o = np.argsort(hp[ys, xs])[::-1][:n]
    return xs[o], ys[o]


def centroids(img, pk, r=3):
    """Replace the peak positions (columns 2, 3) by background-subtracted centroids in a (2r+1)^2 box."""
    out = []
    for a, z, x, y in pk:
        x, y = int(x), int(y)
        if not (r + 8 <= x < img.shape[1] - r - 8 and r + 8 <= y < img.shape[0] - r - 8):
            continue
        w = img[y - r:y + r + 1, x - r:x + r + 1]
        b = np.median(img[y - r - 8:y + r + 9, x - r - 8:x + r + 9])
        w = np.clip(w - b, 0, None)
        if w.sum() <= 0:
            continue
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        out.append((a, z, x + (w * xx).sum() / w.sum(), y + (w * yy).sum() / w.sum()))
    return np.array(out).reshape(-1, 4)


def grid_start(img, alt, az, k1=5.49):
    px, py = peaks(img)
    tree = cKDTree(np.c_[px, py])
    best = (0, None)
    for theta in np.r_[np.arange(30, 48.1, 0.5), np.arange(-48, -29.9, 0.5)]:
        for x0 in np.arange(690, 871, 3):
            for y0 in np.arange(410, 591, 3):
                x, y = altaz_to_xy(alt, az, (x0, y0, theta, k1, 0.0))
                d, _ = tree.query(np.c_[x, y], distance_upper_bound=4)
                s = np.isfinite(d).sum()
                if s > best[0]:
                    best = (s, (x0, y0, theta, k1, 0.0))
    return best


if __name__ == "__main__":
    ra, dec, v = load_bsc(BSC_CSV)
    paths = sys.argv[1:]
    img, hdr, t = read_fits(paths[0])
    b = v < 2.5
    alt, az = radec_to_altaz(ra[b], dec[b], t)
    up = alt > 25
    p = np.array((773.8, 500.1, -39.1, 5.448, -1.25e-5))   # result of grid_start + fit, 2026-11-12
    for it, vmax in enumerate((2.5, 3.5, 4.0)):
        pts = []
        for path in paths:
            img, hdr, t = read_fits(path)
            sel = v < vmax
            alt, az = radec_to_altaz(ra[sel], dec[sel], t)
            up = alt > 10
            pk = star_peaks(ndimage.uniform_filter(img, 3), alt[up], az[up], p, radius=6, min_contrast=200)
            pts.append(centroids(img, pk))
        pts = np.vstack(pts)
        p, resid = fit_camera(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], p)
        good = resid < 3 * np.median(resid) + 1
        p, resid = fit_camera(pts[good, 0], pts[good, 1], pts[good, 2], pts[good, 3], p)
        print(f"V<{vmax}: {good.sum()}/{len(pts)} stars; params ({', '.join(f'{q:.6g}' for q in p)}); "
              f"rms {np.sqrt(np.mean(resid**2)):.2f} px, max {resid.max():.1f}")
    a = pts[good, 0]; zz = pts[good, 1]
    for lo, hi in ((10, 20), (20, 30), (30, 45), (45, 60), (60, 90)):
        s = (a >= lo) & (a < hi)
        print(f"  alt {lo}-{hi}: n {s.sum()}, rms {np.sqrt(np.mean(resid[s]**2)):.2f}")
    for q in range(4):
        s = (zz >= 90 * q) & (zz < 90 * (q + 1)) & (a > 30)
        print(f"  az {90*q}-{90*q+90} (alt>30): n {s.sum()}, rms {np.sqrt(np.mean(resid[s]**2)):.2f}")
