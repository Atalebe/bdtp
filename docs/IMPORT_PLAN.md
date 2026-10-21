# Selective numerical-code import

Status: first original-core import completed; full paper-reproduction packaging remains pending. See NUMERICAL_IMPORT.md. Manuscript revision is separate.

## Keep the reviewed baseline intact

The root archive and standalone manuscript identities are recorded in
`provenance/baseline.json`. Retain them outside Git as immutable source records.
Existing FDT, INF, DEP and FIT identifiers remain unchanged. A later display-title
change must not silently change the finite-cardinality assumptions or historical
source names.

## Import one dependency-complete code snapshot

1. Inspect the archive and verify all 385 root-manifest identities. No extraction
   or code execution is necessary for this integrity step.
2. Inventory original numerical modules, their tests, required parameter tables,
   reference outputs and dependency paths. Follow nested dependencies explicitly.
3. Select only the author's method code, tests and necessary synthetic evidence.
   Exclude reviewer reconstructions, correspondence, manuscript-build scripts,
   unrelated proposals, third-party papers, caches and nested archive copies.
4. Make an import map containing source archive hash, member path, member hash,
   destination path and any explicitly reviewed packaging-only change. Preserve
   upstream notices. Do not claim an omitted reviewer suite was imported.
5. Establish one small installable package, separate experiment drivers and source
   provenance. Do not invent a numerical API before comparing actual imports.
6. Run imported deterministic tests and stored-result checks under the recorded
   environment. Separately authorize any complete Monte Carlo replay; distinguish
   a replay from new statistical evidence. No favourable-outcome gate is allowed.
7. Add automation only after the package is runnable. Keep fast checks separate
   from manual full simulations and paper builds; record all relevant commands.

## Intended layout after the later import

`src/` holds original numerical methods; `tests/` their deterministic tests;
`experiments/` explicit frozen drivers/configurations; `reference/` selected small
verified results; `provenance/` their import maps and hashes. The first byte-preserving numerical snapshot is now in `reference_impl/`; the full public layout and scientific reproduction release are not yet complete.

Before a milestone commit, preserve failed diagnostics, repair actual integrity
problems without changing scientific thresholds, pass every applicable gate and
write the next successful local logbook entry. Verify the resulting remote commit.

## Campaign import 002

Original campaign entrypoints, retained outputs and manuscript-asset regeneration
are now present locally. See `PAPER_REPRODUCTION.md`, `reference/table_figure_map.json`
and the import-002 record. This does not import external reconstruction code or
assign redistribution rights. Remote publication is on HOLD after the repository
was observed public; no visibility change was performed by this import.
