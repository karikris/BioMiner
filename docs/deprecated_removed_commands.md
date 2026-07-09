# Deprecated And Removed Commands

BioMiner now exposes one production workflow: build/audit the registry, run the rank-aware production pipeline, and use vision/evidence subcommands only as debug stage tools.

Removed public commands include:

- `biominer apply-rules`
- `biominer compact-parquet`
- `biominer report-name-evidence`
- `biominer qa-rate-limit`
- `biominer qa-summary`
- `biominer export-bucket-views`
- `biominer gc-cache`
- low-level `biominer registry fetch-taxonomy`
- low-level `biominer registry compile-fixture`
- low-level `biominer registry compile-enriched`
- low-level `biominer registry enrich-sources`
- low-level `biominer registry seed-flickr-queries`
- duplicate `biominer species ...` aliases

Removed source-level entrypoints include:

- root `flicker_miner.py`
- `scripts/run_flickr_text_search.py`
- `scripts/run_papilio_demoleus_ranked_flickr_slices.py`
- `scripts/generate_bioclip_species_visual_report.py`
- `scripts/evaluate_bioclip_species_validation.py`
- `scripts/registry_eda.py`
- `biominer dev vision yoloe26-prototype-run`

Current public command groups are:

```text
biominer registry build
biominer registry audit
biominer run
biominer vision detect
biominer vision score
biominer vision ablate
biominer evidence join
biominer storage doctor
biominer workstore doctor
```

Debug-only registry, Flickr, comment, runtime-check, prefetch, smoke, crop-preview, evaluation, and benchmark utilities live under `biominer dev`.

The removed `report-name-evidence` path depended on ad hoc text-list inputs. Registry name evidence now belongs in the versioned registry outputs, and discovery/query provenance is folded into canonical Flickr source records.

Generated reports and historical audit decks are runtime artifacts, not source documentation. Durable decisions belong under `docs/adr/`.
