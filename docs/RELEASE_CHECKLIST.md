# Release preparation and publication

The author approved the public repository, the v1.0.0 release and this file split
on 21 October 2026: original code/scripts and executable configurations under MIT;
original documentation, figures, tables and synthetic reference datasets under
CC BY 4.0. The manuscript remains separate. See `LICENSING.md`.

Before creating the tag, run the combined paper checks, validate the citation
metadata, verify the imported source identities and the completed replay outputs,
then write the successful local milestone and commit. A tag must identify that
verified commit; never move an existing published tag or overwrite reference data.

`.github/workflows/publish-v1.yml` runs only for the exact `v1.0.0` tag. It fetches
that tag, checks the release metadata and recorded local qualification, and uses
the repository-scoped GitHub Actions token to publish the approved release. It
does not execute new simulations, use personal credentials, or publish other tags.

After publication, inspect the GitHub release, target commit, Zenodo processing
result, creator metadata and licence scope. Only add a DOI after the archive record
is publicly retrievable and its repository/version/files match. A pushed tag alone
is not a published GitHub release; a published release alone is not proof that
Zenodo has finished. Preserve any error and do not create repeated releases just
to retry an unresolved archival problem.

The numerical guarantees retain their stated assumptions. This software release
is not independent human specialist review, a journal submission, or a cosmological
finding. External reconstruction-code rights, manuscript release and venue-specific
disclosures remain separate. Any later anonymous-review snapshot requires its own
identity check; this canonical repository is now public.
