# trgb

Simulation, edge-detection, and calibration work around the **Tip of the Red Giant Branch (TRGB)**
distance indicator: the sharp cutoff in the red-giant-branch luminosity function that marks the
helium flash, used as a standard candle for nearby galaxies.

## `trgb_lib.py`

All reusable logic lives here now — simulation, the RGB slope fit, and the LPED edge detector with
its peak-quality diagnostics. The notebooks below are thin usage examples built on top of it
(`from trgb_lib import *`), not where the logic lives.

**Simulation**

- `random_seed()` — random 32-bit seed for reproducible simulations.
- `k1mag(alpha, m_range)` / `rgb_lf(m_trgb, alpha, m_range, N_stars=None, N_1mag=None, seed=None)` —
  conversion factor and sampler for the RGB luminosity function below the TRGB (see **Luminosity
  function model** below).
- `k1mag_agb(alpha, m_range)` / `agb_powerlaw_lf(m_trgb, N_agb_1mag, alpha, m_range, seed=None)` —
  same idea for a power-law AGB (asymptotic giant branch) population *above* the TRGB, so simulated
  luminosity functions can include the AGB contamination that sits on top of the RGB tip in real data.
- `mag_to_flux(m)` / `flux_to_mag(f)` — magnitude ↔ relative-flux conversion.
- `build_blended_catalog(mags, p_pair, seed=None)` — simulates unresolved blends: picks a fraction
  `p_pair` of stars as primaries, pairs each with a random companion, and sums their fluxes to model
  crowding/blending effects on the luminosity function.
- `add_photometric_errors(mags, errors, seed=None)` — perturbs magnitudes with Gaussian photometric
  noise (scalar or per-star sigma).
- `completeness_function(mags, m50, width)` / `apply_completeness(mags, m50, width, seed=None)` —
  logistic completeness model `C(m) = 1 / (1 + exp((m - m50) / width))` and a function that randomly
  drops stars accordingly, to simulate detection incompleteness near the detection limit.

**RGB slope fit**

- `fit_alpha_mle(m, m1, m2)` — maximum-likelihood fit of the RGB power-law slope `alpha` over
  `m1 <= m <= m2`, returning `(alpha_mle, alpha_err, result)`. The error comes from the analytic
  second derivative of the log-likelihood at the MLE.
- `local_alpha_ratio(m, d=1.0, step=0.01, mmin=None, mmax=None)` — re-estimates `alpha` from the
  bright/faint count ratio in a sliding window of half-width `d`, to check whether the single-slope
  assumption holds locally rather than just globally.

**Local Poisson Edge Detector (LPED) and peak diagnostics**

- `local_poisson_filter(mags, d=0.5, alpha=0.3, step=0.01, sigma=0.0, method="score")` — the TRGB
  edge statistic (see below). Returns `(x, response, counts_bright, counts_faint)`.
- `detect_local_peak(x, response, d, initial=None, search_range=None, min_height=None,
  min_prominence=None, local_fraction=0.7, competitor_fraction=0.5, min_peak_separation=None)` —
  finds the main peak in a filter response and returns `(trgb, diagnostics)`, where `diagnostics` is
  a dict of peak-quality statistics (see **Peak-quality diagnostics** below) — the formalized version
  of the informal S1/S2/R21/D/Q notes from earlier iterations of this work.

## Luminosity function model

The RGB luminosity function is modeled as a power law in magnitude below the tip:

```
phi(m) = A * 10^[alpha * (m - m_trgb)],   m_trgb <= m <= m_trgb + m_range
```

with `alpha` around 0.3. Using `10^alpha = e^lambda` (`lambda = alpha * ln(10)`), sampling is done by
drawing `u ~ Uniform(0, 1)` and inverting the CDF over a magnitude range `m_range`:

```
m = m_trgb + (1 / (alpha * ln(10))) * ln[1 + u * (10^(alpha * m_range) - 1)]
```

To generate an exact number of stars in the first magnitude below the tip (`N_1mag`, matching how
real TRGB samples are usually specified) rather than an exact total, the conversion factor is:

```
N_total = N_1mag * (10^(alpha * m_range) - 1) / (10^alpha - 1)
```

This is `k1mag`/`k1mag_agb` in `trgb_lib.py`; `agb_powerlaw_lf` uses the mirrored version of the same
relation for a power law *declining* away from the tip on the AGB side.

## TRGB edge-detection methods (`filters.ipynb`)

`filters.ipynb` builds a simulated luminosity function (RGB + AGB + photometric errors, chained via
the `trgb_lib.py` simulation functions) and demonstrates the two detection methods that are now part
of the library, plus older experimental methods that are still notebook-only (not yet migrated):

- **MLE slope fit + LPED** (library functions, demonstrated in cells 2–7) — `fit_alpha_mle` estimates
  the RGB power-law slope `alpha`; `local_poisson_filter` then uses that `alpha` to run the **Local
  Poisson Edge Detector**: it compares star counts in two adjacent magnitude bins of width `d`
  straddling a trial position `z` (`a = N(z-d <= m < z)`, `b = N(z <= m < z+d)`). For a smooth
  power-law LF the expected count ratio is `R = 10^(alpha * d)`; deviations from that ratio (via a
  `"difference"`, `"score"`, or `"lrt"` test statistic) signal an edge. `detect_local_peak` then
  scores the resulting response curve for peak quality (see below). This MLE+LPED+peak-diagnostics
  chain is the current, actively developed detection method.
- **Sobel filter** (`make_sobel` / `sobel_detect`, still inline in the notebook, not in
  `trgb_lib.py`) — bins the LF finely, convolves with a Gaussian, and applies a Sobel-style
  derivative kernel to find the steepest rise (the classic Sobel-edge-detector approach to TRGB
  detection, e.g. Lee et al. 1993).
- **GLOESS** (`gloess`, `gloess_linear`, `gloess_edge`, still inline, not in `trgb_lib.py`) —
  Gaussian-windowed locally weighted smoothing of the binned LF, with the edge taken from the
  derivative of the smoothed curve.
- **Maximum Likelihood Algorithm (MLA)** (`mla`, `fi_mla`, `wiarygodnosc`, still inline, not in
  `trgb_lib.py`) — fits the TRGB position and RGB/AGB shape parameters directly to the unbinned
  magnitude distribution by maximum likelihood, following Makarov et al. (2006).

These older methods (Sobel, GLOESS, MLA) haven't been carried over into `trgb_lib.py` yet — treat
them as earlier experiments kept for reference/comparison, not as the current approach.

### Peak-quality diagnostics

`detect_local_peak` in `trgb_lib.py` is the formalized version of statistics originally sketched as
notes in a markdown cell:

  > pik musi być wysoki i samotny *(a good peak must be high and isolated)*

| Original note | `detect_local_peak` diagnostic |
| --- | --- |
| `S1 = max(chi)` | `peak_height` |
| `S2` (second-highest peak), `R21 = S1/S2` (1-mag window) | `local_second_height` / `local_second_ratio` (within `d`), or `second_peak_height` / `second_peak_ratio` (beyond `d`, i.e. an independent competing peak) |
| `dm21 = abs(m1 - m2)` | `local_second_separation` / `second_peak_separation` |
| `N_07max_1mag` (peaks at 0.8/0.7/0.5× max height within 1 mag) | `n_local_peaks` (count above `local_fraction` of the main peak's height within `d`) |
| local dominance `D = 1 - chi1/chi2` | `local_second_ratio` / `second_peak_ratio` (equivalent information, expressed as a ratio rather than `1 - ratio`) |
| local roughness `Q` (maxima within distance `d`) | `n_local_peaks`, `n_competing_peaks` |

It also adds diagnostics that weren't in the original notes: `peak_prominence`,
`peak_width_half_prominence`, `width_asymmetry` (left vs. right steepness), `curvature`/
`normalized_curvature`, and `sharpness` (`prominence / width_half_prominence`).

## TRGB calibration compilation (`mi0.ipynb`)

Compiles published absolute-magnitude TRGB calibrations (`M_TRGB` in the I-band / F814W) from the
literature — grouped by anchor type (e.g. globular clusters like ω Cen, geometric anchors like
NGC 4258) and by instrument (ground-based, HST F814W, JWST) — and renders them as forest plots
(point + error bar per reference). Outputs: `TRGB_calibrations_forest.png` (calibration-by-anchor
comparison) and `TRGB_space_telescopes_forest.png` (ground vs. HST vs. JWST comparison).
`trgb_MIb_updated.png` is a related saved figure.
