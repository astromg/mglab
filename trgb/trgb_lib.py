import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.stats import binom, norm

def random_seed():
    """
    Return a random unsigned 32-bit seed.
    """
    return int(np.random.SeedSequence().generate_state(1)[0])


# ------------------------------------------------------------
# Conversion factor from N_1mag to the total number of RGB stars
# ------------------------------------------------------------

def k1mag(alpha=0.3, m_range=5.0):
    """
    Conversion factor between N_1mag and the total number of RGB stars.

    The RGB luminosity function is defined as:

        dN/dm ∝ 10**[alpha * (m - m_trgb)]

    Parameters
    ----------
    alpha : float
        RGB luminosity-function slope. Must be non-negative.
        For alpha = 0, the luminosity function is flat.
    m_range : float
        Magnitude range below the TRGB.

    Returns
    -------
    float
        Ratio N_total / N_1mag.
    """

    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    if m_range <= 0:
        raise ValueError("m_range must be positive.")

    if alpha == 0:
        return m_range

    lam = alpha * np.log(10.0)

    return (
        (np.exp(lam * m_range) - 1.0)
        / (np.exp(lam) - 1.0)
    )


def rgb_lf(m_trgb,alpha=0.3,m_range=5.0,N_stars=None,N_1mag=None,seed=None):
    """
    Generate an RGB luminosity function below the TRGB.

    The luminosity function is defined as:

        dN/dm ∝ 10**[alpha * (m - m_trgb)]

    over the interval:

        m_trgb <= m <= m_trgb + m_range

    Parameters
    ----------
    m_trgb : float
        TRGB magnitude.
    alpha : float
        Luminosity-function slope. Must be non-negative.
        For alpha = 0, the luminosity function is flat.
    m_range : float
        Magnitude range below the TRGB.
    N_stars : int or None
        Exact total number of RGB stars.
    N_1mag : int or None
        Exact number of RGB stars within the first magnitude
        below the TRGB.
    seed : int or None
        Random seed.

    Returns
    -------
    ndarray
        Simulated RGB magnitudes.
    """

    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    if m_range <= 0:
        raise ValueError("m_range must be positive.")

    if N_stars is None and N_1mag is None:
        raise ValueError("Specify either N_stars or N_1mag.")

    if N_stars is not None and N_1mag is not None:
        raise ValueError("Specify either N_stars or N_1mag, not both.")

    if N_stars is not None and N_stars < 0:
        raise ValueError("N_stars must be non-negative.")

    if N_1mag is not None and N_1mag < 0:
        raise ValueError("N_1mag must be non-negative.")

    if N_1mag is not None and m_range < 1.0:
        raise ValueError(
            "m_range must be at least 1 magnitude "
            "when N_1mag is specified."
        )

    rng = np.random.default_rng(seed)

    # Exact total number of stars
    if N_stars is not None:

        if alpha == 0:
            return rng.uniform(m_trgb,m_trgb + m_range,size=N_stars,)

        lam = alpha * math.log(10.0)
        u = rng.uniform(0.0, 1.0, size=N_stars)

        return m_trgb + (1.0 / lam) * np.log(1.0 + u * (10.0 ** (alpha * m_range) - 1.0))

    # Exact number of stars within the first magnitude
    k = k1mag(alpha=alpha,m_range=m_range,)

    N_tot = round(N_1mag * k)
    N_faint = N_tot - N_1mag

    if alpha == 0:
        m1 = rng.uniform(m_trgb,m_trgb + 1.0,size=N_1mag,)

        m2 = rng.uniform(m_trgb + 1.0,m_trgb + m_range,size=N_faint,)

    else:
        lam = alpha * math.log(10.0)

        u = rng.uniform(0.0, 1.0, size=N_1mag)

        m1 = m_trgb + (1.0 / lam) * np.log(1.0 + u * (10.0 ** alpha - 1.0))

        u = rng.uniform(0.0, 1.0, size=N_faint)

        m2 = m_trgb + 1.0 + (1.0 / lam) * np.log(1.0 + u * (10.0 ** (alpha * (m_range - 1.0)) - 1.0))

    m = np.concatenate((m1, m2))
    rng.shuffle(m)

    return m


# ------------------------------------------------------------
# Conversion factor from N_agb_1mag to total AGB population
# ------------------------------------------------------------

def k1mag_agb(alpha=0.2, m_range=2.0):
    """
    Conversion factor between N_agb_1mag and the total number
    of AGB stars.

    The AGB luminosity function is defined as:

        dN/dm ∝ 10**[alpha * (m - m_trgb)]

    above the TRGB.

    Parameters
    ----------
    alpha : float
        AGB luminosity-function slope. Must be non-negative.
        For alpha = 0, the luminosity function is flat.
    m_range : float
        Magnitude range above the TRGB. Must be at least
        one magnitude.

    Returns
    -------
    float
        Ratio N_total / N_agb_1mag.
    """

    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    if m_range < 1.0:
        raise ValueError("m_range must be at least 1 magnitude.")

    if alpha == 0:
        return m_range

    lam = alpha * np.log(10.0)

    return ((1.0 - np.exp(-lam * m_range))/ (1.0 - np.exp(-lam)))


def agb_powerlaw_lf(m_trgb,N_agb_1mag,alpha=0.2,m_range=2.0,seed=None,):
    """
    Generate a power-law AGB luminosity function above the TRGB.

    The luminosity function is defined as:

        dN/dm ∝ 10**[alpha * (m - m_trgb)]

    over the interval:

        m_trgb - m_range <= m <= m_trgb

    Parameters
    ----------
    m_trgb : float
        TRGB magnitude.
    N_agb_1mag : int
        Exact number of AGB stars within the first magnitude
        above the TRGB.
    alpha : float
        Luminosity-function slope. Must be non-negative.
        For alpha = 0, the luminosity function is flat.
    m_range : float
        Magnitude range above the TRGB.
    seed : int or None
        Random seed.

    Returns
    -------
    ndarray
        Simulated AGB magnitudes.
    """

    if N_agb_1mag < 0:
        raise ValueError("N_agb_1mag must be non-negative.")

    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    if m_range < 1.0:
        raise ValueError("m_range must be at least 1 magnitude.")

    rng = np.random.default_rng(seed)

    k = k1mag_agb(alpha=alpha,m_range=m_range,)

    N_tot = round(N_agb_1mag * k)
    N_bright = N_tot - N_agb_1mag

    if alpha == 0:
        m1 = rng.uniform(m_trgb - 1.0,m_trgb,size=N_agb_1mag,)

        m2 = rng.uniform(m_trgb - m_range,m_trgb - 1.0,size=N_bright,)

    else:
        lam = alpha * math.log(10.0)

        u = rng.uniform(0.0,1.0,size=N_agb_1mag,)

        x1 = -(1.0 / lam) * np.log(1.0 - u * (1.0 - 10.0 ** (-alpha)))

        m1 = m_trgb - x1

        u = rng.uniform(0.0,1.0,size=N_bright,)

        x2 = 1.0 - (1.0 / lam) * np.log(1.0 - u * (1.0 - 10.0 ** (-alpha * (m_range - 1.0))))

        m2 = m_trgb - x2

    m = np.concatenate((m1, m2))
    rng.shuffle(m)

    return m



# ------------------------------------------------------------
# Magnitude ↔ flux
# ------------------------------------------------------------

def mag_to_flux(m):
    """
    Convert magnitude to relative flux.
    """
    return 10.0 ** (-0.4 * np.asarray(m))


def flux_to_mag(f):
    """
    Convert relative flux to magnitude.
    """
    f = np.asarray(f)

    if np.any(f <= 0):
        raise ValueError("Flux must be positive.")

    return -2.5 * np.log10(f)


# ------------------------------------------------------------
# Random blending
# ------------------------------------------------------------

def build_blended_catalog(mags, p_pair=0.1, seed=None):
    """
    Generate a separate catalog of blended objects.

    Parameters
    ----------
    mags : array_like
        Input stellar magnitudes.
    p_pair : float
        Fraction of stars selected as primary components.
    seed : int or None
        Random seed.

    Returns
    -------
    blended_mags : ndarray
        Magnitudes of blended objects.
    parent_idx : ndarray
        Indices of first components.
    companion_idx : ndarray
        Indices of secondary components.
    """

    mags = np.asarray(mags, dtype=float)

    if mags.ndim != 1:
        raise ValueError("mags must be one-dimensional.")

    if not 0.0 <= p_pair <= 1.0:
        raise ValueError("p_pair must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    n_blend = round(p_pair * len(mags))

    parent_idx = rng.choice(len(mags),size=n_blend,replace=False,)

    companion_idx = rng.integers(0,len(mags),size=n_blend,)

    same = companion_idx == parent_idx

    while np.any(same):
        companion_idx[same] = rng.integers(0,len(mags),size=np.sum(same),)
        same = companion_idx == parent_idx

    flux_prim = mag_to_flux(mags[parent_idx])
    flux_sec = mag_to_flux(mags[companion_idx])

    blended_mags = flux_to_mag(flux_prim + flux_sec)

    return blended_mags, parent_idx, companion_idx

def add_photometric_errors(mags, errors, seed=None):
    """
    Perturb magnitudes with Gaussian photometric errors.

    Parameters
    ----------
    mags : array_like
        Input magnitudes.
    errors : float or array_like
        Photometric uncertainties in magnitudes.
        Can be a single value or one value per star.
    seed : int or None
        Random seed.

    Returns
    -------
    ndarray
        Perturbed magnitudes.
    """

    mags = np.asarray(mags, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if mags.ndim != 1:
        raise ValueError("mags must be one-dimensional.")

    if np.any(errors < 0):
        raise ValueError("errors must be non-negative.")

    if errors.ndim > 1:
        raise ValueError("errors must be a scalar or one-dimensional.")

    if errors.ndim == 1 and len(errors) != len(mags):
        raise ValueError("If errors is an array, it must have the same length as mags.")

    rng = np.random.default_rng(seed)

    noise = rng.normal(loc=0.0,scale=errors,size=len(mags),)

    return mags + noise

def completeness_function(mags, m50, width):
    """
    Calculate detection completeness as a function of magnitude.

    The completeness model is:

        C(m) = 1 / (1 + exp((m - m50) / width))

    Parameters
    ----------
    mags : array_like
        Stellar magnitudes.
    m50 : float
        Magnitude at which completeness is 50%.
    width : float
        Width of the completeness transition.
        Smaller values produce a sharper cutoff.

    Returns
    -------
    ndarray
        Detection probabilities between 0 and 1.
    """

    mags = np.asarray(mags, dtype=float)

    if width <= 0:
        raise ValueError("width must be positive.")

    return 1.0 / (
        1.0 + np.exp((mags - m50) / width)
    )


def apply_completeness(mags, m50, width, seed=None):
    """
    Randomly remove stars according to a completeness function.

    Parameters
    ----------
    mags : array_like
        Input magnitudes.
    m50 : float
        Magnitude at which completeness is 50%.
    width : float
        Width of the completeness transition.
    seed : int or None
        Random seed.

    Returns
    -------
    detected_mags : ndarray
        Magnitudes of detected stars.
    detected_idx : ndarray
        Indices of detected stars in the input catalog.
    completeness : ndarray
        Detection probability assigned to each input star.
    """

    mags = np.asarray(mags, dtype=float)

    if mags.ndim != 1:
        raise ValueError("mags must be one-dimensional.")

    completeness = completeness_function(mags,m50=m50,width=width,)

    rng = np.random.default_rng(seed)

    detected = rng.random(len(mags)) < completeness
    detected_idx = np.flatnonzero(detected)

    detected_mags = mags[detected_idx]

    return detected_mags, detected_idx, completeness



def fit_alpha_mle(m, m1, m2):
    """
    MLE dla nachylenia RGB luminosity function:

        phi(m) ~ 10**(alpha * (m - m1))

    w zakresie m1 <= m <= m2.

    Zwraca:
        alpha_mle, alpha_err, result
    """

    m = np.asarray(m, dtype=float)

    # wybór gwiazd w zakresie fitu
    mask = (m >= m1) & (m <= m2)
    mm = m[mask]

    N = len(mm)
    if N < 3:
        raise ValueError("Za mało gwiazd w zakresie fitu.")

    W = m2 - m1
    x = mm - m1
    ln10 = math.log(10.0)

    def loglike(alpha):
        """
        log L(alpha)
        """
        if alpha <= 0:
            return -np.inf

        lam = alpha * ln10

        return (
            N * math.log(lam)
            + lam * np.sum(x)
            - N * math.log(math.exp(lam * W) - 1.0)
        )

    def neg_loglike(alpha):
        return -loglike(alpha)

    bounds = (0.001, 2.0)
    xatol = 1e-10

    res = minimize_scalar(
        neg_loglike,
        bounds=bounds,
        method="bounded",
        options={"xatol": xatol},
    )

    if not res.success:
        raise RuntimeError(
            "alpha MLE optimization did not converge: " + str(res.message)
        )

    alpha_mle = res.x

    lo, hi = bounds
    edge_tol = max(1e-4, 1000.0 * xatol)
    if alpha_mle <= lo + edge_tol or alpha_mle >= hi - edge_tol:
        raise RuntimeError(
            f"alpha MLE ({alpha_mle:.4g}) landed on the edge of the search "
            f"bounds {bounds}; the true maximum likely lies outside this "
            "range rather than having converged."
        )

    # analityczna druga pochodna log-likelihood
    lam = alpha_mle * ln10
    E = math.exp(lam * W)

    # d2 lnL / d alpha2
    d2logL = (
        -N / alpha_mle**2
        + N * (ln10**2) * (W**2) * E / (E - 1.0)**2
    )

    if d2logL >= 0:
        raise RuntimeError(
            "Second derivative of the log-likelihood is non-negative at "
            "the fitted alpha; cannot compute a valid uncertainty (the fit "
            "is not at a true interior maximum)."
        )

    alpha_err = math.sqrt(-1.0 / d2logL)

    return alpha_mle, alpha_err, res

# overlaping moving window diagnostics

def local_alpha_ratio(m, d=1.0, step=0.01, mmin=None, mmax=None):
    """
    Re-estimate the RGB slope alpha locally, from the count ratio in a sliding window.

    At each position z the stars are counted in the two adjacent windows
    a = N(z-d < m < z) and b = N(z < m < z+d). For a power-law luminosity function
    phi(m) proportional to 10**(alpha*m) the ratio b/a is 10**(alpha*d), so

        alpha(z) = log10(b/a) / d.

    Comparing this curve with the global fit_alpha_mle value shows whether a single
    slope is a fair description of the data or only an average over a changing one.

    Parameters
    ----------
    m : array_like
        Magnitudes.
    d : float
        Half-window width in magnitudes.
    step : float
        Spacing of the output grid in magnitudes.
    mmin, mmax : float, optional
        Range of window centres. Default to m.min() + d and m.max() - d, i.e. the
        widest range for which both windows are fully inside the data.

    Returns
    -------
    x : ndarray
        Window centres.
    alpha : ndarray
        Local slope estimate at each centre; NaN where a window is empty.
    sigma_alpha : ndarray
        Poisson uncertainty of alpha, sqrt(1/a + 1/b) / (d * ln 10); NaN where alpha is.

    Raises
    ------
    ValueError
        If d or step is not positive.
    """
    m = np.asarray(m)

    if d <= 0:
        raise ValueError("d must be positive.")

    if step <= 0:
        raise ValueError("step must be positive.")

    if mmin is None:
        mmin = m.min() + d
    if mmax is None:
        mmax = m.max() - d

    x = np.arange(mmin, mmax, step)

    alpha = []
    sigma_alpha = []

    for z in x:

        a = np.sum((m > z-d) & (m < z))
        b = np.sum((m > z) & (m < z+d))

        if a > 0 and b > 0:

            alpha_i = np.log10(b/a) / d

            sigma_i = (
                1.0 / (d * np.log(10))
                * np.sqrt(1.0/a + 1.0/b)
            )

        else:
            alpha_i = np.nan
            sigma_i = np.nan

        alpha.append(alpha_i)
        sigma_alpha.append(sigma_i)

    return x, np.array(alpha), np.array(sigma_alpha)


# ============================================================
# Local Poisson Edge Detector (LPED)
# ============================================================

def local_poisson_filter(mags,d=0.5,alpha=0.3,step=0.01,sigma=0.0,method="score",):
    """
    Compute a local Poisson edge response for a luminosity function.

    Two adjacent magnitude intervals are compared at each position z:

        a = N(z - d <= m < z)
        b = N(z <= m < z + d)

    For a smooth luminosity function

        phi(m) proportional to 10**(alpha * m),

    the expected count ratio is

        R = lambda_b / lambda_a = 10**(alpha * d).

    Available methods
    -----------------
    difference
        Raw count excess:

            D = b - R*a

    score
        Signed score statistic for the null hypothesis
        lambda_b = R*lambda_a:

            Z = (b - R*a) / sqrt(R*(a+b))

    lrt
        Signed square root of the Poisson likelihood-ratio statistic.

    binomial
        Signed normal-equivalent deviate derived from an exact
        one-sided binomial test of b against n = a + b trials with
        success probability p0 = R / (1 + R):

            p = P(X >= b),  X ~ Binomial(n, p0)
            response = norm.isf(p)

        Unlike score/lrt, this does not rely on a large-count
        (asymptotic) approximation, so it is more reliable for small
        a/b.

    Parameters
    ----------
    mags : array_like
        Stellar magnitudes.
    d : float, optional
        Width of each adjacent magnitude interval.
    alpha : float, optional
        Logarithmic luminosity-function slope.
    step : float, optional
        Sampling step of the response function.
    sigma : float, optional
        Gaussian smoothing width applied to the final response.
    method : {'difference', 'score', 'lrt', 'binomial'}, optional
        Statistic returned by the filter.

    Returns
    -------
    x : ndarray
        Magnitudes at which the filter was evaluated.
    response : ndarray
        Filter response.
    counts_bright : ndarray
        Counts in the bright-side intervals.
    counts_faint : ndarray
        Counts in the faint-side intervals.
    """

    mags = np.asarray(mags, dtype=float)
    mags = mags[np.isfinite(mags)]

    if mags.size == 0:
        raise ValueError("mags must contain at least one finite magnitude.")

    if d <= 0:
        raise ValueError("d must be positive.")

    if step <= 0:
        raise ValueError("step must be positive.")

    if sigma < 0:
        raise ValueError("sigma cannot be negative.")

    if method not in {"difference", "score", "lrt", "binomial"}:
        raise ValueError(
            "method must be 'difference', 'score', 'lrt', or 'binomial'."
        )

    mags = np.sort(mags)

    mmin = mags[0]
    mmax = mags[-1]

    x = np.arange(mmin + d, mmax - d + 0.5 * step, step)

    if x.size == 0:
        raise ValueError("Magnitude range is too small for the selected d.")

    # searchsorted avoids repeatedly constructing Boolean masks.
    left_1 = np.searchsorted(mags, x - d, side="left")
    left_2 = np.searchsorted(mags, x, side="left")
    right_2 = np.searchsorted(mags, x + d, side="left")

    a = left_2 - left_1
    b = right_2 - left_2

    a = a.astype(float)
    b = b.astype(float)

    ratio = 10.0**(alpha * d)
    excess = b - ratio * a

    if method == "difference":

        response = excess

    elif method == "score":

        variance = ratio * (a + b)

        response = np.zeros_like(excess)

        mask = variance > 0
        response[mask] = (excess[mask] / np.sqrt(variance[mask]))

    elif method == "lrt":

        n = a + b

        mu_a = n / (1.0 + ratio)
        mu_b = ratio * n / (1.0 + ratio)

        deviance = np.zeros_like(excess)

        mask_a = a > 0
        mask_b = b > 0

        deviance[mask_a] += (a[mask_a] * np.log(a[mask_a] / mu_a[mask_a]))

        deviance[mask_b] += (b[mask_b] * np.log(b[mask_b] / mu_b[mask_b]))

        deviance *= 2.0
        deviance = np.maximum(deviance, 0.0)

        response = np.sign(excess) * np.sqrt(deviance)

    elif method == "binomial":

        n = a + b
        p0 = ratio / (1.0 + ratio)

        p_value = np.ones_like(excess, dtype=float)

        valid = n > 0

        p_value[valid] = binom.sf(b[valid] - 1,n[valid],p0,)
        tiny = np.finfo(float).tiny
        p_safe = np.clip(p_value, tiny, 1.0)

        response = norm.isf(p_safe)

    if sigma > 0:

        gx = np.arange(-5.0 * sigma,5.0 * sigma + 0.5 * step,step,)

        kernel = np.exp(-0.5 * (gx / sigma)**2)

        kernel /= np.sum(kernel)

        response = np.convolve(response,kernel,mode="same",)

    return x, response, a, b


def _mean_excess(x_segment, response_segment, threshold):
    """
    Average excess of the response above threshold over the magnitude range x_segment.

    The segment must be contiguous on the response grid — the integral runs straight
    from its first to its last point. Returns NaN for a segment too short to integrate.
    """
    if x_segment.size < 2:
        return np.nan

    span = x_segment[-1] - x_segment[0]

    if span <= 0:
        return np.nan

    excess = np.maximum(response_segment - threshold, 0.0)

    return np.trapezoid(excess, x_segment) / span


def detect_local_peak(x, response, d, initial=None, search_range=None, min_height=None, min_prominence=None,
                      local_fraction=0.7, competitor_fraction=0.5, min_peak_separation=None, response_threshold=1.0):
    """
    Locate the TRGB as the main peak of an edge-filter response, and grade that peak.

    The position alone is not the answer: a peak is only believable when it is high and
    alone. So besides the magnitude of the strongest maximum this returns diagnostics
    describing its height, its shape, and how much competition it has.

    Parameters
    ----------
    x : array_like
        Magnitude grid the response was computed on, uniformly spaced.
    response : array_like
        Filter response, e.g. from local_poisson_filter.
    d : float
        Half-window the response was computed with, in magnitudes. Here it sets the
        radius of the zone counted as "local" and the exclusion zone around the main
        peak used by area_without_main.
    initial : float, optional
        Expected TRGB position. Together with search_range it limits the candidates.
    search_range : float, optional
        Half-width of the search window around initial, in magnitudes. Both this and
        initial must be given for the restriction to apply.
    min_height, min_prominence : float, optional
        Height / prominence a maximum must reach to be considered at all.
    local_fraction : float
        A maximum closer than d to the main peak and at least this fraction of its
        height counts towards n_local_peaks, i.e. towards local roughness.
    competitor_fraction : float
        A maximum farther away than twice the main peak's half-prominence width and at
        least this fraction of its height counts towards n_competing_peaks, i.e. as an
        independent rival detection.
    min_peak_separation : float, optional
        Minimum spacing between accepted maxima, in magnitudes. Closer ones are merged
        by keeping the higher.
    response_threshold : float
        Response level treated as background when integrating the area diagnostics;
        only the excess above it counts.

    Returns
    -------
    trgb : float
        Magnitude of the main peak.
    diagnostics : dict
        trgb, index
            Position of the main peak, and its index into x.
        peak_height, peak_prominence, peak_width_half_prominence
            Height, prominence, and width at half prominence (in magnitudes).
        widths_90, peak_flatness
            Width at 10% of prominence, and its ratio to the half-prominence width;
            a flat-topped peak scores high.
        sharpness
            peak_height / sqrt(peak_width_half_prominence).
        n_local_peaks, n_competing_peaks
            Rival maxima nearby and far away, as defined by local_fraction and
            competitor_fraction above.
        area_with_main, area_without_main
            Mean response excess above response_threshold within 1 mag of the peak,
            with the peak included and with it cut out. The second one says how much
            structure the response has where there should be none.
        d, step
            The settings the response was computed with.

    Raises
    ------
    ValueError
        If the inputs are inconsistent or out of range, if no maximum satisfies the
        criteria, if none falls inside the search range, or if the strongest candidate
        has a non-positive response — the last meaning no genuine edge was found.
    """
    x = np.asarray(x, dtype=float)
    response = np.asarray(response, dtype=float)

    if x.size != response.size:
        raise ValueError("x and response must have the same length.")

    if x.size < 3:
        raise ValueError("x must contain at least three points.")

    if d <= 0:
        raise ValueError("d must be positive.")

    if not 0.0 < local_fraction <= 1.0:
        raise ValueError("local_fraction must be in (0, 1].")

    if not 0.0 < competitor_fraction <= 1.0:
        raise ValueError("competitor_fraction must be in (0, 1].")

    step = np.median(np.diff(x))

    if min_peak_separation is None:
        distance_samples = None
    elif min_peak_separation <= 0:
        raise ValueError("min_peak_separation must be positive.")
    else:
        distance_samples = max(1, int(np.round(min_peak_separation / step)))

    peak_indices, properties = find_peaks(response, height=min_height, prominence=min_prominence,
                                          distance=distance_samples, )

    if peak_indices.size == 0:
        raise ValueError("No local maximum satisfies the criteria.")

    # Restrict the candidate list to the requested TRGB region.
    candidate_indices = peak_indices.copy()

    if initial is not None and search_range is not None:
        in_search_range = (np.abs(x[candidate_indices] - initial) <= search_range)
        candidate_indices = candidate_indices[in_search_range]

    if candidate_indices.size == 0:
        raise ValueError("No local maximum found in the selected search range.")

    # The main peak is the highest candidate.
    main_index = candidate_indices[np.argmax(response[candidate_indices])]

    main_x = x[main_index]
    main_height = response[main_index]

    if main_height <= 0:
        raise ValueError(
            "No positive local maximum found in the selected region — the "
            "strongest candidate has non-positive response, indicating no "
            "genuine edge was detected."
        )

    prominences, left_bases, right_bases = peak_prominences(response, np.array([main_index]), )

    main_prominence = prominences[0]

    widths, _, left_ips, right_ips = peak_widths(response, np.array([main_index]), rel_height=0.5,
                                                 prominence_data=(prominences, left_bases, right_bases), )
    width_half_prominence = widths[0] * step

    widths_90, _, left_ips_90, right_ips_90 = peak_widths(response, np.array([main_index]), rel_height=0.1,
                                                          prominence_data=(prominences, left_bases, right_bases), )
    width_90 = widths_90[0] * step

    peak_flatness = width_90 / width_half_prominence

    # All detected peaks and their distances from the main peak.
    other_indices = peak_indices[peak_indices != main_index]
    other_distances = np.abs(x[other_indices] - main_x)

    # Local roughness: peaks sitting within d of the main one and nearly as high. These are
    # not rival detections, they mean the top of the response is ragged rather than clean.
    local_mask = ((other_distances <= d) & (response[other_indices] >= local_fraction * main_height))

    n_local_peaks = int(local_mask.sum())

    # Independent competitors: peaks clear of the main peak's own width and nearly as high.
    # competitor_mask = ((other_distances > d) & (response[other_indices] >= competitor_fraction * main_height))
    competitor_mask = ((other_distances > 2 * width_half_prominence) & (
                response[other_indices] >= competitor_fraction * main_height))

    competitor_indices = other_indices[competitor_mask]

    n_competing_peaks = competitor_indices.size

    # A simple sharpness measure. This is diagnostic, not an error.
    if width_half_prominence > 0:
        # sharpness = main_prominence / np.sqrt(width_half_prominence)
        sharpness = main_height / np.sqrt(width_half_prominence)

    else:
        sharpness = np.nan

    # Mean response excess above the background level within 1 mag of the peak, computed
    # twice: over the whole window, and with the main peak cut out. The second value is a
    # measure of how much the response wanders where nothing should be happening.
    one_mag_mask = np.abs(x - main_x) <= 1.0

    area_with_main = _mean_excess(x[one_mag_mask], response[one_mag_mask], response_threshold)

    # Cutting the peak out leaves two separate wings, so each is integrated on its own and
    # the two are averaged, weighted by the magnitude range each of them covers.
    bright_wing_mask = one_mag_mask & ((main_x - x) >= 2.0 * d)
    faint_wing_mask = one_mag_mask & ((x - main_x) >= 2.0 * d)

    wings = []

    for wing_mask in (bright_wing_mask, faint_wing_mask):

        x_wing = x[wing_mask]

        if x_wing.size < 2:
            continue

        wings.append((x_wing[-1] - x_wing[0], _mean_excess(x_wing, response[wing_mask], response_threshold)))

    if wings:
        area_without_main = sum(span * value for span, value in wings) / sum(span for span, _ in wings)
    else:
        area_without_main = np.nan

    diagnostics = {
        "trgb": main_x,
        "index": int(main_index),

        "peak_height": main_height,
        "peak_prominence": main_prominence,
        "peak_width_half_prominence": width_half_prominence,

        "widths_90": widths_90,
        "peak_flatness": peak_flatness,

        "sharpness": sharpness,

        "n_local_peaks": n_local_peaks,
        "n_competing_peaks": int(n_competing_peaks),

        "area_with_main": area_with_main,
        "area_without_main": area_without_main,

        "d": d,
        "step": step,
    }

    return main_x, diagnostics

# =======================================================
# =================== FILTR SOBELA ======================
# =======================================================

def make_sobel(sigma, j, logscale=False):
    numerical_step = 0.0005
    m = np.arange(j.min(), j.max() + numerical_step, numerical_step, dtype=float)
    x = (j - j.min()) / numerical_step
    indeks = x.astype(int)
    a = np.zeros(len(m))
    for i in indeks:
        a[i] = a[i] + 1
    gauss_x = np.arange(-5 * sigma, 5 * sigma, numerical_step, dtype=float)
    gaussian = lambda x, sigma: 1. * np.exp(-x ** 2 / (2. * sigma ** 2))
    gauss_y = gaussian(gauss_x, sigma)
    splot = np.convolve(a, gauss_y, "same")  # to zajmuje mase czasu
    miu = int(0.5 * sigma / numerical_step)
    pomocnicza1 = splot[2 * miu:len(splot)]
    pomocnicza2 = splot[0:len(splot) - 2 * miu]
    maska = pomocnicza1 < 0.01
    pomocnicza1[maska] = 0.01
    maska = pomocnicza2 < 0.01
    pomocnicza2[maska] = 0.01
    sobel_x = m[miu:len(splot) - miu]
    if logscale:
        sobel = np.log10(pomocnicza1) - np.log10(pomocnicza2)
    else:
        sobel = pomocnicza1 - pomocnicza2

    sobel = sobel * splot.max() / sobel.max()
    maska = sobel < 0
    sobel[maska] = 0
    trim = int((len(sobel) - len(sobel_x)) / 2.)
    if trim > 0:
        sobel = sobel[trim:np.alen(sobel) - trim]
    return m, splot, sobel_x, sobel


def sobel_detect(x, sobel, trgb0):
    Nmax = max(x)
    trgb = 9.
    if trgb0 > 90 or trgb0 == 0:
        i = np.argmax(sobel)
        trgb = x[i]
    else:
        x = np.array(x)
        tmp = (x - trgb0)
        tmp = tmp * tmp
        i = np.argmin(tmp)
        k = i
        l = i
        x0 = sobel[i]
        zmiana = 1
        while zmiana == 1 and k + 1 < len(sobel) and l - 1 > 0:
            if sobel[k] > sobel[k + 1] and sobel[k] > sobel[k - 1]:
                zmiana = 0
                trgb = x[k]
            elif sobel[l] > sobel[l + 1] and sobel[l] > sobel[l - 1]:
                zmiana = 0
                trgb = x[l]
            else:
                zmiana = 1
            k = k + 1
            l = l - 1
    return trgb


# use
# x,l,xf,f = make_sobel(0.03,m,False)
# trgb = sobel_detect(x,f,10)
# print(trgb)

# =======================================================
# =================== MLA ===========================================
# =======================================================
# trgb,a,b,c=mla([10,0.3,1,0.3],0.01,rozklad)  tak jak w pracy Makarov et al. 2006
# sigma_mla to jest rozmycie fi w zwiazku z bledami fot.
# ten pierwszy array to wartosci poczatkowe
#
# x,y=fi_mla(x,trgb,a,b,c,sigma_mla)
#

from scipy.optimize import minimize


def mla(p0, sigma, gtol, rozklad, trgb0):
    res = minimize(
        wiarygodnosc,
        p0,
        args=(rozklad, sigma),
        method='SLSQP',
        bounds=[
            (trgb0 - 0.5, trgb0 + 0.5),
            (-1, 1),
            (-3, 3),
            (-1, 1)
        ],
        tol=gtol
    )

    return res.x


def wiarygodnosc(p, r, sigma):
    trgb, a, b, c = p

    _, A = fi_mla(r, trgb, a, b, c, sigma)

    A = np.maximum(A, 1e-300)

    x = np.arange(np.min(r), np.max(r), 0.01)

    _, y = fi_mla(x, trgb, a, b, c, sigma)

    B = np.log(np.sum(y))

    L = -np.sum(np.log(A)) + len(r) * B

    return L


def fi_mla(r, trgb, a, b, c, sigma):
    x = np.arange(np.min(r), np.max(r), 0.01)

    x1 = x[x > trgb]
    y1 = np.exp(np.clip(np.log(10) * (a * (x1 - trgb) + b), -700, 700))

    x2 = x[x < trgb]
    y2 = np.exp(np.clip(np.log(10) * (c * (x2 - trgb)), -700, 700))

    xk = np.concatenate((x1, x2))
    yk = np.concatenate((y1, y2))

    idx = np.argsort(xk)

    xk = xk[idx]
    yk = yk[idx]

    if sigma > 0:
        gauss_x = np.arange(-3 * sigma, 3 * sigma, 0.01)

        gauss_y = np.exp(
            -gauss_x ** 2 / (2 * sigma ** 2)
        )

        gauss_y /= np.sum(gauss_y)

        yk = np.convolve(
            yk,
            gauss_y,
            mode="same"
        )

    y_interp = np.interp(r, xk, yk)

    return r, y_interp


# use
# początkowe zgadywanie
# p = mla(
#     [10.0,0.3,1,0.3],
#     0.05,
#     1e-6,
#     m,
#     trgb0=10.02
# )
#
# trgb, a, b, c = p
#
# print("TRGB =", trgb)
# print("a =", a)
# print("b =", b)
# print("c =", c)
#
#
# # model LF
# x_model = np.linspace(
#     np.min(m),
#     np.max(m),
#     500
# )
#
# _, y_model = fi_mla(
#     x_model,
#     trgb,
#     a,
#     b,
#     c,
#     0.05
# )
#
#
# plt.figure(figsize=(8,5))
#
# plt.hist(
#     m,
#     bins=30,
#     histtype='step',
#     label="LF"
# )
#
# # skalowanie modelu
# hist_max = np.histogram(m,bins=30)[0].max()
#
# y_model = y_model * hist_max / y_model.max()
#
# plt.plot(
#     x_model,
#     y_model,
#     '-r',
#     linewidth=2,
#     label="MLA"
# )
#
# plt.axvline(
#     trgb,
#     color='blue',
#     linestyle='--',
#     label=f"TRGB={trgb:.3f}"
# )
#
# plt.gca().invert_xaxis()
#
# plt.xlabel("Magnitude")
# plt.ylabel("N")
# plt.legend()
# plt.title("TRGB Maximum Likelihood Fit")
#
# plt.show()

# =======================================================
# gloess
# =======================================================

def gloess(x, y, sigma):

    """
    Gaussian-windowed locally weighted scatterplot smoothing

    x     - bin centers / magnitudes
    y     - luminosity function
    sigma - smoothing scale (mag)
    """

    y_smooth = np.zeros_like(y, dtype=float)

    for i, xi in enumerate(x):

        # odległość od aktualnego punktu
        dx = x - xi

        # Gaussian weights
        w = np.exp(
            -0.5*(dx/sigma)**2
        )

        # lokalna średnia ważona
        y_smooth[i] = np.sum(w*y) / np.sum(w)

    return y_smooth

def gloess_linear(x,y,sigma):

    ys = np.zeros_like(y,dtype=float)

    for i,xi in enumerate(x):

        dx = x-xi

        w = np.exp(
            -0.5*(dx/sigma)**2
        )


        X = np.vstack(
            [
                np.ones(len(x)),
                dx
            ]
        ).T


        W = np.diag(w)


        beta = np.linalg.solve(
            X.T @ W @ X,
            X.T @ W @ y
        )


        ys[i] = beta[0]


    return ys

def gloess_edge(x,y,sigma):

    smooth = gloess_linear(
        x,y,sigma
    )

    edge = np.gradient(
        smooth,
        x
    )

    return smooth, edge

# use
# hist, edges = np.histogram(
#     m,
#     bins=100
# )
#
# x = 0.5*(edges[1:]+edges[:-1])
# y = hist
#
# smooth, edge = gloess_edge(
#     x,
#     y,
#     sigma=0.05
# )
#
#
# i = np.argmax(edge)
#
# trgb_gloess = sobel_detect(x,edge,10)
#
# print("TRGB GLOESS =", trgb_gloess)
