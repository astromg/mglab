# trgb

A laboratory for **Tip of the Red Giant Branch (TRGB)** detection methods — the sharp cutoff in the
red-giant luminosity function (LF) that marks the helium flash, used as a standard candle.

The work here is simulation-driven: build a synthetic LF with a *known* tip, run detection on it,
repeat over many realizations, and study how the recovered tip and its uncertainty behave. The aim is
both a good detector and a good **error estimate** for it. `trgb_lib.py` is the deliverable — it will
eventually be lifted out into an application for real data; the notebooks stay here.

## Files

| File | What it is |
| --- | --- |
| `trgb_lib.py` | The library. All reusable logic: LF simulation, slope fit, detection, diagnostics. |
| `lf.ipynb` | Walk-through of the **simulation** side, step by step: RGB → AGB → blending → photometric errors → completeness, with a plot after each stage. |
| `filters.ipynb` | Walk-through of the **detection** side on one simulated LF: MLE slope fit, local slope check, then LPED with the peak diagnostics. |
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
| `peak_prominence` | `scipy` prominence of the peak. |
| `peak_width_half_prominence` | Width at half prominence, in magnitudes. |
| `widths_90`, `peak_flatness` | Width at 10% of prominence, and the ratio `width_90 / width_half_prominence` — a flat-topped peak scores high. |
| `sharpness` | `peak_height / sqrt(peak_width_half_prominence)`. |
| `n_competing_peaks` | Rival peaks: further than `2 * peak_width_half_prominence` away and at least `competitor_fraction` of the main height. |
| `area_with_main`, `area_without_main` | Mean response excess above `response_threshold` within ±1 mag of the peak — once including the peak, once excluding it (the region `x - trgb >= 2d`). The second measures how noisy the background of the response is. |
| `d`, `step` | The settings the response was computed with. |

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
