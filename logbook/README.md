# Repository milestone logbook

Entries are repository-scoped, numbered LaTeX records under `entries/`. An entry is
written only after all checks for that milestone pass. Failed attempts retain
separate diagnostic logs under `.local_runs/`; they are not successful milestones.

Record: entry number and scope; what was tested; methods; results; failures and
repairs; and next steps. Commit only after the entry exists. Verify remote state
after synchronization. A sync failure is HOLD, not permission to rerun the science.
Bootstrap entry numbering does not replace any experiment's historical logbook.
