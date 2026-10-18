# bdtp

Private development workspace for predictive-adequacy methods in the NMCD programme.

## Current state

Repository bootstrap only. The reviewed numerical implementation has **not yet been
imported**, and this repository is not a paper release or an independently validated
software distribution. The manuscript remains at v0.03, with its existing title,
filenames and contents unchanged while the next scientific review is pending.

The intended methods compare restricted state-predictor classes against declared
finite history-predictor dictionaries under bounded losses. Finite cardinality,
permitted observations, independent evaluation groups and verified pair envelopes
remain explicit assumptions. No theorem, model, tolerance or historical result is
changed by repository preparation.

## Local checks

Only Python 3.10+ and Bash are required for the bootstrap checks:

```bash
bash scripts/run_checks_no_exit.sh
```

The wrapper runs repository checks and unit tests together. It keeps diagnostic logs
under `.local_runs/`, returns a nonzero status on failure, and never exits a calling
interactive shell when invoked with `bash` as above. Do not source it. It does not
install dependencies, access the network, run simulations, or perform Git actions.

The reviewed source archive can be checked without extracting or executing it:

```bash
python3 scripts/verify_bootstrap.py --archive /absolute/path/to/Finite_Dictionary_Predictive_Adequacy_v0_03_Reviewed_Bundle.zip
```

The archive is deliberately not committed: it contains nested development and
review material that requires selective import, not blanket redistribution.

## Repository map

- `provenance/baseline.json`: exact source identities; no numerical rebranding.
- `environment/reference.json`: historical numerical runtime, not an installed environment.
- `docs/IMPORT_PLAN.md`: separate migration of original code, tests and result summaries.
- `docs/RELEASE_CHECKLIST.md`: privacy, attribution, licence and archival decisions.
- `scripts/` and `tests/`: infrastructure verification only.
- `logbook/`: numbered successful repository milestones; failures stay in diagnostics.

Public release, numerical-code import, new simulations and manuscript edits remain
separate steps. No redistribution licence or archival DOI is assigned here.
