# bdtp

Private numerical-code workspace for **Finite-Dictionary Tests of Predictive Adequacy for State–History Models**.
The manuscript title retains “Finite”; no equations or historical scientific results are renamed.

## Current state

The first selective import contains the original numerical core, seven deterministic test suites,
required test configurations and one population-moment planning record. All 19 imported files
are byte-identical to their recorded source members. No reviewer implementations, reports,
correspondence, manuscripts, credentials, nested archives or failed numerical variants are imported.

This is not yet a complete paper-reproduction release. Full campaign drivers, all selected result
summaries, figure regeneration and a public release remain a separate integration stage. The
historical relative import paths are preserved under `reference_impl/` to avoid modifying tested
science during migration. This layout is an import snapshot, not a promised stable public API.

## Local WSL setup

Working directory: `/home/satal/bdtp_repo`. The isolated `.venv` uses the installed WSL Python
and pinned NumPy 2.3.5, SciPy 1.17.0 and mpmath 1.3.0. No system packages are changed.
To reproduce setup on another machine, explicitly run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-numeric.txt
```

Then run the combined offline checks:

```bash
bash scripts/run_core_checks_no_exit.sh
```

The runner verifies imported hashes and package versions, runs the eight infrastructure tests
and all seven selected numerical suites, and preserves diagnostics in `.local_runs/`. It does not
install dependencies, access the network, launch a Monte Carlo campaign or perform Git actions.
Invoke with `bash`, not `source`. Failure stops downstream execution without closing the caller.
No successful milestone is written by the runner itself.

Historical results used Python 3.13.5. WSL currently uses Python 3.12.3. Passing deterministic
compatibility tests does not establish byte-identical Monte Carlo streams across those runtimes.
`environment/reference.json` remains the unchanged historical runtime record.

## Contents

- `reference_impl/`: selected unmodified method code and numerical tests.
- `reference/planning.json`: original population-moment planning calculation, not a power curve.
- `provenance/IMPORT_001.json`: source-to-destination hashes and import scope.
- `scripts/`: explicit source import and combined verification tools.
- `logbook/`: successful repository milestones; failed attempts remain diagnostic records.

The original `provenance/baseline.json` records bootstrap authority at that historical step; it is
not rewritten retrospectively. `provenance/current_scope.json` records the current import scope.

## Release status

The repository remains private. No software licence, release tag, archival DOI, public deposit
or automatic release workflow is assigned. Public redistribution and publication require separate
author approval. Manuscript working copies and declarations are maintained separately.
