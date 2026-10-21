# Repository operating instructions

Scope: prepare and maintain this private code workspace. Do not infer scientific
execution authority from a repository-maintenance request.

1. Preserve source identities, original experiment IDs and historical results.
   Do not rename mathematical variables, candidate families or frozen files merely
   to match repository branding. The title remains Finite. Manuscript edits are separate from numerical-source import; retain historical versions unchanged.
2. Read `provenance/baseline.json` before any numerical import. Use an explicit
   source-path-to-destination-path map with before/after SHA256 values. Do not
   execute source archives or import them wholesale. Scientific API changes require
   separate review and regression tests.
3. Keep manuscripts under external review, referee reports, editorial correspondence,
   credentials and unapproved third-party material out of this repository. Do not
   commit the complete conversation or nested review bundles.
4. Run checks in combined bundles. Use a terminal-preserving outer Bash launcher;
   a failed inner stage stops later stages but retains diagnostics. Never use
   `exit` in instructions pasted directly into an interactive caller.
5. Write a numbered LaTeX milestone entry only after every applicable local gate
   passes. Record test, method, result, failures/repairs and next steps. Then commit
   and synchronize, and verify the remote commit. If synchronization fails, retain
   the local commit and mark remote sync HOLD; do not rerun scientific work.
6. Do not make the repository public, choose a redistribution licence, publish a
   release, mint a DOI, or alter collaborator access without explicit approval.
7. Check all current branch contents before writes. Never force-push or overwrite
   unrelated work. Bootstrap tests do not certify the numerical method.
