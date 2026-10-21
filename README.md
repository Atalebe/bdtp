# Finite-dictionary predictive adequacy

Reference implementation and numerical reproduction assets for
**Finite-Dictionary Tests of Predictive Adequacy for State–History Models**
(Stephen Atalebe).

The methods compare restricted state-predictor classes with a declared finite
history-predictor dictionary under bounded losses. The guarantees require independent
evaluation groups; they do not establish autonomous dynamical closure or physical
non-Markovian cosmology.

## Package scope

The original numerical core, campaign drivers and reference outputs are retained
without changing their scientific source. The package includes the dependence,
inference, fitting/allocation, pair-envelope, finite-binomial and planning
calculations used by the main paper. Six table datasets and four figure datasets
can be regenerated from stored outputs without running simulations.

## Licensing and release

The approved v1.0.0 distribution licenses original software and executable
configurations under **MIT** and original documentation, figures, tables and
synthetic reference data under **CC BY 4.0**. See `LICENSING.md` for scope.
Citation metadata are in `CITATION.cff`; Zenodo metadata are in `.zenodo.json`.
The manuscript is maintained separately and is not part of this software release.

The repository is public with the author's approval. Historical source records
retain their original status; `provenance/current_scope.json` records current
permissions. A public release is identified by its GitHub release and tag. A DOI
is not claimed until the Zenodo record has been observed and verified.

## Quick checks

With the isolated environment already installed:

```bash
bash scripts/run_paper_checks_no_exit.sh
```

The command checks source identities, deterministic tests, reference records and
paper-asset regeneration. It does not run a full simulation campaign, use the
network, install dependencies, write a milestone entry or perform Git operations.
Each invocation retains diagnostics in a new `.local_runs/` directory. Invoke the
launcher with `bash`, not `source`.

Environment installation, when explicitly needed:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-numeric.txt -r requirements-figures.txt
```

## Frozen campaign replay

```bash
bash scripts/run_paper_campaign_no_exit.sh \
  --campaign fit-pair --output .local_runs/fit_pair_001
```

The output directory must be new. A run stages its sources before execution and
compares numerical outputs with fixed references. A mismatch stops completion;
references are never replaced. Replaying a frozen stream is not new independent
statistical evidence. `docs/PAPER_REPRODUCTION.md` documents all ten run modes,
counts, dependency versions and limitations.

## Navigation

- `reference/table_figure_map.json`: each main-paper table/figure and its command.
- `reference_impl/`: byte-preserved original code, plans and main campaign outputs.
- `reference/`: inference records, planning inputs and frozen paper assets.
- `scripts/paper_outputs.py`: CSV/figure regeneration, no simulations.
- `provenance/IMPORT_001.json` and `IMPORT_002.json`: source-to-destination identities.
- `logbook/entries/`: successful local milestones; failed attempts remain diagnostics.

The historical relative-import layout under `reference_impl/work/` is intentional;
it is source, not a temporary run directory. Avoid flattening it without regression
checks. Numerical packages are pinned; source provenance is distinct from a licence.
External reviewer reconstructions, referee correspondence, manuscript drafts,
credentials and nested archive bundles are not part of this import.
