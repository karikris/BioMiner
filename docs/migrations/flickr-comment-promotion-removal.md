# Flickr comment-promotion removal

Status: required breaking cutover from the legacy bucket workflow.

BioMiner no longer fetches Flickr comments, mines comment-derived query terms,
or uses comment text to move records into Gold or Silver occurrence buckets.
The `dev comments` command, comment-review run stages, comment-review SQLite
state, reviewed-object-evidence artifact, and comment-review output columns have
been removed.

This cutover is intentional. A comment is unverified text associated with a
photo; it is not a signed taxonomic review and cannot establish event date,
location, species identity, or occurrence release. Product-owned human review
must use the current immutable review contracts and retain reviewer/provenance
evidence outside BioMiner's handoff.

Existing comment-review SQLite databases and reviewed Parquet outputs are not
modified. Preserve them with the exact prior BioMiner Git revision if they are
needed for audit. Do not relabel their decisions as current review evidence or
copy their Gold/Silver fields into adaptive evidence-maturity outputs.

Raw text already present in an explicitly supplied input may still be retained
as non-authoritative source context or a review signal. It must not autonomously
create query work, confirm an occurrence, or bypass a manual-review gate.

Rollback requires deploying the prior comment-capable revision against a
separate historical artifact root. Current code deliberately has no
compatibility reader or forwarding command for the removed state.
