# Image-bearing butterfly species and keyword enrichment

Start: main `d997aed6651ce8b7e39f6a44116d8e5c768a9d99`.

The explicit current request selects the newest complete ~17M image DuckDB:
`data/derived/gbif_media_database/v3/gbif_media.duckdb` (16,612,063 rows).
Generic ground-zero prose excludes earlier staging derivatives, but this selected
database's manifest chain reaches the retained raw DWCA archive. Validate that
chain and the actual database checksum; do not reuse old enriched registries or
the newer consolidated Parquet marked outside ground-zero production lineage.
Unrelated dirty scientific work belongs to another session and remains unstaged.

1. Collect all image-bearing species under the seven-family butterfly scope.
   Require a valid species key, scientific name and HTTP(S) image evidence;
   retain infraspecific parent species, every source name variant and conflicts.
   Use bounded read-only DuckDB, numeric species order, immutable Parquet outputs
   and a manifest written last. Calculate the actual denominator from this data.
2. Run the 15 existing source families plus ALA and Wikimedia Commons: 17 logical
   sources. Wikimedia retains its existing Wikipedia/Wikispecies aggregate.
   Replace registry-local GBIF-only lookup with a live, taxon-bound names adapter.
   Batch bulk providers once; report absent NCBI/static snapshots explicitly.
   EOL/BOLD/LibreTranslate labels without callable clients are inventory gaps,
   not additional successful providers. Never invent names or machine translate
   Latin names. MyMemory stays dictionary-only, with unreviewed candidates disabled.
3. Add a resumable explicit local runner, SQLite work ledger, bounded HTTP cache,
   provider budgets/retries, an exclusive run lock, Parquet snapshots, progress
   logs/status and a documented one-line launcher. Every source result is committed
   atomically; unavailable/deferred/error states remain distinct from empty success.
4. Execute fixture tests, independent review, bounded real extraction, and one
   initial live worker. Commit only owned implementation/tests/docs/provenance to
   main and push. Start the long job once and report PID, logs, actual counts,
   operational limits and unavailable sources. Do not wait for an all-species job
   to finish or launch duplicate discovery/model/production jobs.

The qualified app26 interpreter may execute this explicitly requested local job
without registering it, repointing `.venv`, reactivating contained environments,
or changing runtime admission. Source names remain provider assertions, not
verified taxonomy, image labels or occurrence release. Outputs stay local under
an explicit `--local` contract; production S3/PostgreSQL defaults are unchanged.

The ADHD preflight selects a direct extension of existing adapters. Named
Superpowers skills were not installed; use equivalent explicit planning,
behavioral tests and independent implementation review without claiming skill
execution. GitHits supports read-only DuckDB design; Valyu/official source docs
support mutable provider API contracts. Preserve failed checks and exact evidence.
