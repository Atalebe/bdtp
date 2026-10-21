# Campaign and reference-output import 002

Date: 21 October 2026. Local checkout: `/home/satal/bdtp_repo`.

## Disposition

Local packaging and verification passed. The new import has NOT been pushed.
GitHub now reports `Atalebe/bdtp` as public, whereas the previous import recorded a
private repository. This task did not change visibility. Publishing these new
files is held pending explicit confirmation of the public destination. The licence,
release tag and Zenodo DOI remain unassigned. The manuscript was not changed.

## Imported material

105 original source/configuration/result/figure/table files were imported with
unchanged bytes (70,968,506 bytes before Git compression). Together with the prior
19-file core, 124 original file identities are checked. The additions comprise
13 Python files, 83 JSON files, four figure PDFs, four generated LaTeX tables and
one original plan-checksum file. A separate clean re-import reproduced all 105
selected identities and the same import manifest.

The source chain verifies 142 outer, 202 v0.04, 385 v0.03, 485 FIT and 208 DEP
manifest entries, plus 12 manuscript-asset source entries. The import reads the
archives but does not execute them wholesale. It excludes the main manuscript,
review correspondence, external reviewer reconstructions, credentials and nested
archive copies.

New adapters stage the original scripts in fresh run directories, enforce pinned
numerical dependencies and compare computed JSON outputs with immutable references.
The original method implementations, seeds, plans, thresholds and sample sizes are
unchanged. Six derived CSV table datasets and a map of all six tables and four
figures are included. Frozen paper figure PDFs retain their bytes. The new plotting
adapter preserves numerical data, not identical typography to the old figures.

## Executed historical replays

| Mode | Byte-identical JSON comparisons | Scope |
|---|---:|---|
| `fit-pair` | 22/22 | Full 20-cell fitting/range design: 1,900 evaluations, 1,000 fitted prefixes, 400 independent maximal pool pairs |
| `dependence-paper` | 13/13 | All 12 main-paper dependence cells, 2,000 repetitions each; 8,000 independent pool pairs shared across duplication factors |
| `inference-predictive` | 21/21 | Five fits and 3,000 conditional evaluation batches |
| `inference-inverse` | 7/7 | Six inverse-calibration cases, 1,000 datasets per case |
| `class-selection` | 8/8 | Original 700-batch finite-class extension conditional on three libraries |
| `dependence-original` | 2/2 | Full original 62-case dependence campaign; all-case and summary JSON compared |
| `inference-numerical` | 1/1 | Original high-precision numerical checks |
| `power` | 1/1 | 90 finite-binomial probabilities; not Monte Carlo |
| `planning` | 1/1 | Original library and population-moment planning |
| **Total** | **76/76** | Nine distinct executed replay modes |

`fit-generic` is supplied but was not separately rerun: the complete paired replay
already runs the original generic calculations and checks the historical summaries.
The hardware-dependent resource-timing driver is supplied with its retained record,
not claimed newly benchmarked. The original dependence campaign also emits arrays
and a CSV; only the all-case and summary JSON are direct comparison targets here.
Their stored raw-array hashes are reproduced within the all-case JSON.

These executions replay exposed seeds and do not add independent scientific evidence.
The numerical environment is WSL Python 3.12.3, NumPy 2.3.5, SciPy 1.17.0 and mpmath
1.3.0, with one BLAS thread. These specific outputs reproduce despite the historical
Python 3.13.5 reference; this is not a general cross-version reproducibility guarantee.
Matplotlib 3.10.8 was installed only in the checkout's virtual environment.

## Packaging checks

340 distinct tests passed: 8 bootstrap, 303 inherited numerical and 29 new packaging
checks. Retained records agree with all 208 fitting/dependence method summaries.
The four generated LaTeX tables' 35 numerical rows were checked, alongside direct
extraction of the five design and risk rows. The six CSV tables, plot-data JSON and
eight rendered figure files reproduce byte-for-byte in a separate source-directory
build: 15/15 generated artifacts. This is not a newly provisioned operating system;
the clean copy uses the same installed virtual environment.

The four regenerated figure images were visually inspected. The combined launcher
passes its injected-failure control: a nonzero status is returned, the caller and
diagnostics survive, downstream artifact generation stops, and no completion or
successful logbook is written. Existing output directories and protected reference
paths are rejected.

## Failures and repairs

During dependency inspection, two unchanged vendored modules were added for the
standalone original-dependence driver. Its first actual launch then stopped before
simulation because the original `PLAN_SHA256.txt` was missing. The original checksum
file was restored and included in runtime staging; the failed run remains preserved.
The repaired run passed at the original settings. No scientific code, expected
number, test tolerance, random seed or numerical reference was changed.

## Coverage limits

The main-paper original numerical campaigns are present. The appendix's separate
AI-assisted single-pair and paired-reconstruction records are not imported as the
author's original implementation. Including those external reconstruction materials
in a public deposit requires a separate source/rights decision. The repository is
not yet a complete licensed archival release of every supplement or the manuscript.

## Commands

Fast combined checks and paper-asset regeneration:

```bash
cd ~/bdtp_repo
bash scripts/run_paper_checks_no_exit.sh
```

Explicit full paired fitting replay (use a new output directory):

```bash
bash scripts/run_paper_campaign_no_exit.sh \
  --campaign fit-pair --output .local_runs/fit_pair_002
```

Neither command installs packages, uses the network, changes Git state or writes a
formal milestone. Refer to `PAPER_REPRODUCTION.md` and `reference/table_figure_map.json`
for other modes and exact input/output mappings. Do not rerun successful science
merely to retry a later remote synchronization.
