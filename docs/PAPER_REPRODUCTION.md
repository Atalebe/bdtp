# Paper reproduction

Title: **Finite-Dictionary Tests of Predictive Adequacy for State–History Models**.
This package covers the original main-paper numerical models, campaigns, reference
outputs, six numerical table datasets and four figure datasets. It does not import
the manuscript itself, referee correspondence, or externally reconstructed code.
The manuscript is maintained separately; its title and results are unchanged.

## Quick verification (no Monte Carlo campaign)

From the checkout:

```bash
bash scripts/run_paper_checks_no_exit.sh
```

This verifies imported hashes, runs the core and packaging tests, checks retained
replicate records against summaries, and rebuilds six CSV tables and four figures.
Diagnostics and generated files go under a fresh `.local_runs/paper_checks_*` folder.
It performs no network, Git, package-installation or publication action. No success
record is written after failure; run with `bash`, not by sourcing the launcher.

For a new environment, install the numerical and plotting requirements explicitly:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-numeric.txt -r requirements-figures.txt
```

Historical reference: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, mpmath 1.3.0.
The WSL import uses Python 3.12.3 and the same numerical dependency versions.
Matplotlib 3.10.8 is used for plotting. Full installed versions are recorded in the
local environment report. A random seed alone does not promise cross-platform
byte identity. The replay checker reports differences and never replaces reference
outputs or silently relaxes a tolerance.

## Recompute a frozen campaign

Each invocation requires a new output directory. For example:

```bash
bash scripts/run_paper_campaign_no_exit.sh \
  --campaign fit-pair --output .local_runs/fit_pair_001
```

The runner stages the original source tree into the output directory before
execution. This avoids writing into checked-in references even where historical
scripts have fixed relative paths. Source files, random seeds, scientific models,
error levels, relevance margins and sample allocations are not changed.

| Campaign | What runs | Reference comparison |
|---|---|---|
| `fit-pair` | All 20 fitting allocations, both envelopes, all four rules; 1,900 evaluations, 1,000 fitted prefixes and 400 independent pool pairs | 20 cells, fits and summary: 22 files |
| `class-selection` | Original 700-batch comparison conditional on three fixed libraries | Eight class-extension JSON files |
| `fit-generic` | Original generic-only fitting campaign | Original campaign summary; timing output is not byte-compared |
| `inference-predictive` | Five fitted libraries and 3,000 conditional held-out batches | All 21 predictive JSON files |
| `inference-inverse` | Six inverse-calibration cases, 1,000 datasets each | Seven inverse-calibration JSON files |
| `inference-numerical` | Original finite-model high-precision checks | One numerical JSON file |
| `dependence-paper` | 12 duplication/power/null cells, 2,000 trials each; copied configurations share 8,000 pool pairs | 12 cells plus summary: 13 files |
| `dependence-original` | Earlier 62-case within-realisation dependence campaign | Original all-case and summary JSON files; extra arrays/CSV are regenerated |
| `power` | 90 finite-binomial operating probabilities, exact integer weights and 80-digit thresholds | One JSON file; not Monte Carlo |
| `planning` | Original 512-group library, widths and population-moment curves/crossings | One JSON file; not a power study |

Full fitting/dependence runs are intentionally not launched by the quick check.
They may take several minutes and produce tens of megabytes. The current replay
record states precisely which commands were actually run in WSL. A repeat of a
frozen stream is reproduction, not new independent statistical evidence.

## Table and figure map

`reference/table_figure_map.json` maps each numbered manuscript item to its driver
and reference files. `scripts/paper_outputs.py` writes the six numerical tables as
CSV and all plotted series as JSON. It optionally renders four figures. The frozen
four figure PDFs and four externally input LaTeX tables are preserved byte-for-byte
under `reference/paper_assets/`. Tables 1 and 2 are inline in the manuscript; their
numerical content is rebuilt from the case registry and inference records.

```bash
.venv/bin/python scripts/paper_outputs.py \
  --output .local_runs/paper_assets_001 --figures
```

Regenerated figures preserve numerical series and scientific labels. They are not
claimed to have identical PDF bytes or exact typography to the frozen assets.
No simulation is run by this command.

## Coverage boundary

The appendix's single-pair operating estimates and paired pair-versus-class study
come from separately identified AI-assisted reconstructions. Those scripts and raw
outputs are NOT imported as the author's original FIT code. Their inclusion in a
public deposit needs a separate source/rights decision. The supplementary margin
and betting explorations are likewise outside this numerical import. The main
method comparisons do not silently substitute reconstruction results.

The original code remains in its historical relative-import layout beneath
`reference_impl/`. This is a verified reference implementation, not a new installable
API. Do not rename or flatten those paths without an explicit parity-tested change.
No licence, release tag or archival DOI is assigned by this packaging step.

## Supplementary resource timings

The original timing driver is available with `--stage resources` at
`reference_impl/work/fit_parent/NMCD_PC_FIT_001_v0_1/scripts/run_experiment.py`.
Use a new output directory. Timing values depend on the machine and are not
byte-level reference targets; this packaging stage does not claim a new timing
benchmark. `experiments/original/complete_results.py` is retained as an original
extraction helper, not a portable entrypoint; use `scripts/paper_outputs.py` instead.
