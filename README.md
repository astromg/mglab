# mglab

Personal lab of Jupyter notebooks for observational-astronomy work: image-level artifact detection,
aperture/PSF photometry, LMC/SMC reddening mapping, observatory data-quality checks, and TRGB
(Tip of the Red Giant Branch) distance-indicator research. Each notebook is a mostly self-contained
investigation rather than part of a shared pipeline, except for the `trgb/` directory, which has its
own shared library — see [`trgb/README.md`](trgb/README.md).

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
```

or, with `uv` (there's a `pyproject.toml` / `uv.lock`):

```
uv sync
uv run jupyter notebook
```

Notebooks pull in packages beyond what's pinned in `pyproject.toml` (astropy, photutils, scipy,
matplotlib, ephem, tqdm, jsonschema, ...) — install whatever a given notebook's import cell needs,
e.g. `pip install astropy photutils` or `uv add astropy photutils`.

## Notebooks

### Image-frame artifact detection

- **`cosmic_ray_detection.ipynb`** — injects synthetic hot pixels and a satellite trail into a test
  frame, then flags cosmic-ray-like features with a pair of Sobel-style convolution kernels
  (`scipy.ndimage.convolve`) followed by connected-component labeling (`generate_binary_structure`
  + `label`) to group flagged pixels into candidate events.
- **`satelite_detection.ipynb`** — detects satellite trails (straight lines) in FITS frames using a
  custom directional line-detection kernel (`line_detection_kernel`) plus Sobel-filter and
  simple-threshold masks for comparison, contoured over the display image.
- **`hot_piksel_detection.ipynb`** — placeholder; currently empty.

### Photometry

- **`photometry.ipynb`** — aperture and PSF photometry on FITS frames using `photutils`
  (`DAOStarFinder`, `PSFPhotometry`, `EPSFBuilder`, `CircularAperture`/`CircularAnnulus`). Splits the
  image into overlapping segments (`make_segments`) to fit a spatially-varying sky background before
  source detection/photometry.

### LMC/SMC reddening

- **`red_lmc.ipynb`** / **`lmc_cmd.ipynb`** — near-duplicate notebooks computing (V−I) color and
  E(B−V) reddening across LMC/SMC fields from a (RA, Dec)-indexed photometric catalog
  (`load_fields`, `radeg`/`decdeg` for sexagesimal coordinate parsing). Selects Red Clump (RC) stars
  from the CMD with a polygon mask (`colorselectPoly` / `matplotlib.path.Path`), fits the RC color
  distribution as a Gaussian-plus-polynomial-background profile (`rc_fit`/`RC`/`RCGC` with
  `scipy.optimize.curve_fit`), converts the fitted RC peak color to E(B−V) via a fixed calibration
  constant, and compares the result field-by-field against reference reddening values. `red_lmc.ipynb`
  is the more complete/later version (tracks more diagnostics per field); `lmc_cmd.ipynb` looks like
  an earlier pass over the same logic. **Note:** `lmc_cmd.ipynb` references `f_name` and
  `result_fname` without ever defining them — it will raise `NameError` if run top-to-bottom in a
  fresh kernel (`red_lmc.ipynb` does define both).

### Observatory data quality / site conditions

- **`ocm_data_stats.ipynb`** — walks a directory tree of nightly FITS data (`/work/vela/oca/fits/...`)
  and counts science/flat frames per night for an annual observation-campaign (OCM) report.
- **`ocm_dqc.ipynb`** — data-quality control on FFS pipeline results (loaded from pickled
  `ffs_wyniki_*.pkl` files): validates FITS headers against a hand-written `jsonschema`
  (`SCIENCE_SCHEMA`, Draft-07) covering things like image type, mirror/dome status, telescope
  altitude, and exposure time, then plots QC diagnostics (e.g. background gradient vs. telescope
  altitude) per target object.
- **`sky_brightness.ipynb`** — measures sky background level (median count rate per exposure time)
  across a run of science frames and correlates it with Moon position/phase using `ephem`
  (observer fixed at the OCA site: lon −70.1963°, lat −24.598°, elevation 2817 m).
- **`ffs_test.ipynb`** — exercises an external `FFS`/`ffs_make_segments` module (from a sibling
  project, not included in this repo — see note below) for per-frame flat-field/segment statistics,
  computes Moon phase/altitude per frame with `ephem`, and pickles the aggregated results for
  `ocm_dqc.ipynb` to consume.

### Misc / scratch

- **`colors_trans.ipynb`** — a couple of one-off color-transformation snippets (K, J−K → V−K, B−V).
- **`examples.ipynb`** — scratch pad of Markdown/Python syntax reminders (not analysis).
- **`syntax_tpg.ipynb`** — exploring `pyaraucaria`'s `ObsPlanParser` together with `jsonschema`
  validation to parse/convert observation-plan syntax. **Note:** this file currently contains an
  unresolved git merge conflict (`<<<<<<< HEAD` markers), so it won't parse as valid JSON/open in
  Jupyter until that's resolved.

### External dependencies used by some notebooks

A few notebooks (`ffs_test.ipynb`, `syntax_tpg.ipynb`) import from sibling projects on this machine
(`ffs`/`ffs_lib`, `pyaraucaria`) that live outside this repo. They won't import unless those projects
are also installed/on `PYTHONPATH` (see the `pip install -e ../pyaraucaria` note in `examples.ipynb`).

## `trgb/`

TRGB distance-indicator simulation, edge-detection, and calibration work, with its own shared module
(`trgb_lib.py`). See [`trgb/README.md`](trgb/README.md) for the theory and a breakdown of each
notebook.

## `data/`

Gitignored local data directory; its contents are machine-local and not tracked/reproducible from
this repo.
