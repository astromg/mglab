# trgb

A laboratory for **Tip of the Red Giant Branch (TRGB)** detection methods — the sharp cutoff in the
red-giant luminosity function (LF) that marks the helium flash, used as a standard candle.

The work here is simulation-driven: build a synthetic LF with a *known* tip, run detection on it,
repeat over many realizations, and study how the recovered tip and its uncertainty behave. The aim is
both a good detector and a good **error estimate** for it. `trgb_lib.py` is the deliverable — it will
eventually be lifted out into an application for real data; the notebooks stay here.

## What is to be studied

The plan, in the order it is meant to be worked through. One notebook per case, kept separate;
`d_param_mc.ipynb` is the template — every realization is analysed with **all** values of the swept
parameter, so the comparison is paired rather than between independent runs.

1. **How the diagnostics depend on the simulation parameters.** Sweep one at a time with the rest
   held fixed — mainly `N_rgb_1mag`, `R` (the AGB-to-RGB ratio, `N_agb_1mag / N_rgb_1mag`) and
   `photometric_error`. The question is the same one `d_param_mc.ipynb` asks about `d`: do two
   realizations that differ only in that parameter give measurably different diagnostics?
2. **Combinations of two parameters.** Once the one-dimensional behaviour is known, look for
   interactions — e.g. `R` against `photometric_error` as two-dimensional heatmaps of the bias and
   of the error.
3. **The method parameters last**: `d`, the smoothing `sigma`, and the rest. They are settings we
   choose, so it only makes sense to tune them once it is clear what the data itself does.
4. **Pick the diagnostic that models the error best.** This is the point of the whole exercise: a
   quantity measurable without knowing the truth, from which the uncertainty of a detection can be
   predicted.
5. **Check what bootstrap does and does not capture.** It should reproduce the noise-driven part of
   the error; the expectation to test is that it stays blind to the biases.

## Files

| File | What it is |
| --- | --- |
| `trgb_lib.py` | The library. All reusable logic: LF simulation, slope fit, detection, diagnostics. |
| `lf.ipynb` | Walk-through of the **simulation** side, step by step: RGB → AGB → blending → photometric errors → completeness, with a plot after each stage. |
| `filters.ipynb` | Walk-through of the **detection** side on one simulated LF: MLE slope fit, local slope check, then LPED (score statistic) with the peak diagnostics drawn where they are measured. |
| `d_param_mc.ipynb` | Monte Carlo case study: how the LPED half-window `d` affects the TRGB error. One notebook per MC case; this is the first. |
| `mi0.ipynb` | Unrelated to the above — compiles published `M_TRGB` calibrations into forest plots. |
| `TRGB_calibrations_forest.png`, `TRGB_space_telescopes_forest.png`, `trgb_MIb_updated.png` | Figures produced by `mi0.ipynb`. |

Notebooks import the library with `from trgb_lib import *` and run with the notebook's own directory
as CWD (no package install). Start Jupyter from the repo's `.venv`.

## `trgb_lib.py`

**Simulation**

- `random_seed()` — random 32-bit seed.
- `k1mag(alpha, m_range)`, `k1mag_agb(alpha, m_range)` — factor converting "stars in the first
  magnitude below the tip" into a total star count.
- `rgb_lf(m_trgb, alpha, m_range, N_stars, N_1mag, seed)` — samples the RGB LF *below* the tip.
- `agb_powerlaw_lf(m_trgb, N_agb_1mag, alpha, m_range, seed)` — power-law AGB population *above* the
  tip, i.e. the contamination that sits on top of the edge in real data.
- `mag_to_flux(m)`, `flux_to_mag(f)` — magnitude ↔ relative flux.
- `build_blended_catalog(mags, p_pair, seed)` — unresolved blends: pairs a fraction `p_pair` of stars
  with random companions and sums their fluxes. Returns `(blended_mags, parent_idx, companion_idx)`.
- `add_photometric_errors(mags, errors, seed)` — Gaussian noise, scalar or per-star sigma.
- `completeness_function(mags, m50, width)`, `apply_completeness(mags, m50, width, seed)` — logistic
  completeness `C(m) = 1 / (1 + exp((m - m50) / width))` and the random dropping of stars by it.

**RGB slope**

- `fit_alpha_mle(m, m1, m2)` — maximum-likelihood power-law slope `alpha` over `m1 <= m <= m2`.
  Returns `(alpha, alpha_err, result)`; the error comes from the analytic second derivative of the
  log-likelihood.
- `local_alpha_ratio(m, d, step, mmin, mmax)` — `alpha` re-estimated from the bright/faint count
  ratio in a sliding window, to check whether one global slope is actually a fair assumption.

**Detection — the current method**

- `local_poisson_filter(mags, d, alpha, step, sigma, method)` — the **Local Poisson Edge Detector
  (LPED)**. At each trial position `z` it counts stars in two adjacent windows of width `d`,
  `a = N(z-d <= m < z)` and `b = N(z <= m < z+d)`. A smooth power-law LF predicts the ratio
  `R = 10^(alpha*d)`; departures from it mark an edge. `method` selects the test statistic:
  `"difference"`, `"score"` (default), `"lrt"`, `"binomial"`. Optional Gaussian smoothing via
  `sigma`. Returns `(x, response, counts_bright, counts_faint)`.
- `detect_local_peak(x, response, d, ...)` — picks the main peak of the response and returns
  `(trgb, diagnostics)`.

The chain is: `fit_alpha_mle` → `local_poisson_filter` (using that `alpha`) → `detect_local_peak`.

**Detection — earlier methods, kept for comparison**

Not the current approach; they live in the library but are not part of the MLE+LPED chain.

- `make_sobel`, `sobel_detect` — classic Sobel-derivative edge detection on the binned LF
  (Lee et al. 1993).
- `gloess`, `gloess_linear`, `gloess_edge` — Gaussian-windowed locally weighted smoothing, edge taken
  from the derivative of the smoothed curve.
- `mla`, `fi_mla`, `wiarygodnosc` — Maximum Likelihood Algorithm: fits tip position and RGB/AGB shape
  directly to the unbinned magnitudes (Makarov et al. 2006).

## Peak-quality diagnostics

The dict returned by `detect_local_peak`. The idea being tested is that *a trustworthy peak is high
and alone*, so most entries measure either height or isolation.

| Key | Meaning |
| --- | --- |
| `trgb`, `index` | Position of the main peak (magnitude, and index into `x`). |
| `peak_height` | Response value at the peak. |
| `peak_prominence`, `peak_base` | `scipy` prominence of the peak, and the level it rises from (`peak_height - peak_prominence`). |
| `peak_width_half`, `peak_width_90` | Widths at 0.5 and at 0.9 of the peak height, in magnitudes. |
| `peak_flatness` | `peak_width_90 / peak_width_half` — a flat-topped peak scores high. |
| `sharpness` | `peak_height / sqrt(peak_width_half)`. |
| `n_local_peaks` | Ragged top: rival peaks within `d` of the main one and at least `local_fraction` of its height. Not competing detections — a sign that the summit is noisy. |
| `n_competing_peaks` | Independent rivals: further than `2 * peak_width_half` away and at least `competitor_fraction` of the main height. |
| `area_main` | Mean response excess above `response_threshold` over the peak zone, `|m - trgb| < d`. |
| `area_outside_bright`, `area_outside_faint`, `area_outside` | The same, over the two wings left inside a ±1 mag window once the peak zone is cut out: brighter than the tip, fainter than it, and the two together weighted by the range each covers. They measure how much structure the response has where there should be none. The sides are reported apart on purpose — the AGB sits on the bright one, the RGB on the faint one, so an excess there does not mean the same thing. |
| `outside_area_fraction` | `area_outside / area_main` — how the background compares with the peak itself. Near 0 for a clean single peak, around 1 when the wings are as busy as the peak. |
| `peak_width_half_x1/x2/level`, `peak_width_90_x1/x2/level` | Where the two widths actually sit: endpoints on the magnitude grid and the response level they are measured at. For drawing them, not for grading the peak. |
| `d`, `step` | The settings the response was computed with. |

Three things to keep in mind when reading these.

**Widths are measured from zero**, at a fraction of the peak height — the plain FWHM idea, computed
by walking away from the summit until the response falls to the level and interpolating between the
two samples that straddle it. Prominence is reported but no longer defines any width: it reaches
down to distant minima, so it describes the response as a whole rather than the peak.

**The area diagnostics depend on `d` by construction.** The peak zone is `±d` inside a fixed ±1 mag
window, so as `d` grows the peak zone widens and the wings shrink. Part of any trend against `d` is
that geometry, not a cleaner response.

**`peak_height` is not comparable across `d`.** The window holds `∝ d` stars, so the response of a
given edge grows roughly as `sqrt(d)`. Comparing detections computed with different `d` needs either
a `sqrt(d)` rescaling or `sharpness`, which carries that factor implicitly (`peak_width_half ∝ d`).

## Luminosity function model

The RGB LF is a power law in magnitude below the tip:

```
phi(m) = A * 10^[alpha * (m - m_trgb)],   m_trgb <= m <= m_trgb + m_range
```

with `alpha` around 0.3. Sampling inverts the CDF with `u ~ Uniform(0, 1)`:

```
m = m_trgb + (1 / (alpha * ln 10)) * ln[1 + u * (10^(alpha * m_range) - 1)]
```

Real TRGB samples are usually quoted as a star count in the first magnitude below the tip, so the
number of stars is specified that way and converted with

```
N_total = N_1mag * (10^(alpha * m_range) - 1) / (10^alpha - 1)
```

which is `k1mag` / `k1mag_agb`. `agb_powerlaw_lf` mirrors the same relation for a power law declining
*away* from the tip.
