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

Debug-only registry, Flickr, comment, runtime-check, prefetch, smoke, crop-preview, evaluation, and prototype utilities live under `biominer dev`.

The removed `report-name-evidence` path depended on ad hoc text-list inputs. Registry name evidence now belongs in the versioned registry outputs, and discovery/query provenance is folded into canonical Flickr source records.
