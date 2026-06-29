# Name Translation Harvester Integration Plan

## Current State

- `src/biominer/registry/name_translations.py` is copied into the registry package.
- `config/name_translation_target_locales.json` is copied as the default broad locale target list.
- The downloaded CLI handoff patch is preserved at `docs/research/biominer_name_translation_cli_patch.diff`.
- The handoff patch is not a valid `git apply` patch, so the CLI integration was applied manually.
- The CLI command is exposed as `uv run biominer registry translate-names`.

## Data Products

The harvester writes audit sidecars first:

- `name_translation_assertions.parquet`
- `name_translation_external_links.parquet`
- `name_translation_source_snapshots.parquet`
- `name_translation_errors.parquet`
- `name_translation_manifest.json`

With `--merge-into-enrichment`, those sidecars append/deduplicate into BioMiner enrichment files:

- `source_name_assertions.parquet`
- `external_taxon_links.parquet`
- `enrichment_source_snapshots.parquet`
- `source_error_records.parquet`
- `name_translation_merge_manifest.json`

The assertion schema matches the existing enrichment assertion shape: `assertion_id`, `accepted_taxon_key`, `display_name`, `language`, `script`, `source`, `source_record_id`, `source_taxon_id`, `trust_tier`, `precision_tier`, `confidence`, `enabled`, `review_state`, `disabled_reason`, `retrieved_at`, and `licence`.

## Integration Workflow

1. Run source-backed harvesting into a sidecar directory or registry directory.
2. Inspect `name_translation_manifest.json`, source counts, language counts, and `name_translation_errors.parquet`.
3. Merge into enrichment only after sidecar QA is acceptable.
4. Run `registry compile-enriched` so enabled reviewed names flow into `names.parquet`, `name_evidence.parquet`, and Flickr query definitions.
5. Keep machine translation rows disabled unless reviewed; they should remain weak search-expansion candidates, not accepted vernacular names.

## Test Plan

- Unit test sidecar writing, deterministic deduplication, and enrichment merge idempotency.
- CLI dry-run style tests for species mode, registry mode, locale JSON parsing, source parsing, LibreTranslate key lookup, merge option, and invalid argument combinations.
- Adapter tests with fake clients for GBIF pagination, Wikidata SPARQL bindings, Wikimedia `llcontinue`, iNaturalist locale probing, CoL result parsing, and LibreTranslate disabled machine candidates.
- Registry integration fixture test: harvest fake translated names, merge, compile enriched registry, and assert enabled source-backed names appear in final `names.parquet` while disabled T5 rows remain in candidates.
- No test should require network access or real API credentials.

## Live Run Guardrails

- Start with a single species and source-backed sources only.
- Use `--limit` for registry-wide dry runs.
- Review errors and language distributions before `--merge-into-enrichment`.
- Do not enable T5 machine translations without explicit review.
