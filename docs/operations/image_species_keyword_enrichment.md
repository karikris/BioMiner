# Image-bearing butterfly species and multilingual names

`scripts/enrich_butterfly_image_species.py` reads the selected v3 image database
without modifying it, collects every image-bearing species in the seven-family
scope, then enriches its scientific and common names. It does not download images.
The database has 16,612,063 rows. Its complete seven-family image cohort has
10,608 source species IDs and 10,607 distinct scientific names; two source IDs
share one spelling and retain separate provenance. Numeric species/source
counters are stable ordinals, with 10,608 as the species denominator.

The observed download uses opaque taxon keys, for example `SLPT` and `DGC`.
These remain source-owned identifiers, with the existing `gbif:` namespace used
for internal associations. They are never converted from base 36 or submitted
as numeric v1 identifiers. Exact name/family resolution can produce a separate
legacy GBIF link. The [GBIF taxonomy documentation](https://techdocs.gbif.org/en/data-processing/taxonomy-interpretation)
explains the separate Catalogue of Life XR and legacy backbone identifiers.

Start or resume, from the repository root, with the explicitly selected qualified
interpreter (this is one shell line):

```bash
BIOMINER_PYTHON="$PWD/.native-runtime/application-p2a26-20261022-1/env/bin/python" bash scripts/start_butterfly_keyword_enrichment.sh --local --output "$PWD/data/derived/image_species_keywords/v4" --secrets-file "$HOME/.config/agent-env/secrets.env"
```

The launcher uses a transient systemd user service when available, and prints its
unit name, supervisor PID and launcher log. This survives the launching tool's
process cleanup. Otherwise it uses nohup for an ordinary persistent terminal;
`BIOMINER_JOB_SUPERVISOR=systemd` or `nohup` selects the launcher explicitly.
It uses the public
resource supervisor with a 24-hour hard limit and 30-second shutdown grace.
The coordinator stops between source jobs after 23.5 hours by default. Repeat
the same command to resume after that invocation ends; an exclusive output lock
rejects concurrent workers. Do not start a second worker while one is running.
Use `--collect-only` with the Python script for extraction without API calls.
The runtime remains unregistered; this explicit local job does not alter
production S3/PostgreSQL defaults or runtime profiles.

Monitor:

```bash
tail -f data/derived/image_species_keywords/v4/progress.log
```

The format is `234/10608 species | 3/17 sources | source=wikidata`.
This shows the current species and source ordinal, not a claim that every earlier
lookup succeeded. `status.json` records the latest event; `collection.log` records
input hashing and extraction. Source outcomes distinguish complete, unavailable,
deferred, error, input conflict and not applicable. A complete lookup with no names
remains a zero-name success. Missing snapshots and errors are never converted into
biological absence.

| Order | Source family | Interface |
| --- | --- | --- |
| 1 | GBIF | Identity-checked species profile, vernaculars and synonyms; exact legacy bridge for all source IDs |
| 2 | Catalogue of Life | ChecklistBank name usage search |
| 3 | Wikidata | SPARQL names and identifier-bound links |
| 4 | iNaturalist | Taxa names |
| 5 | ITIS | Scientific and common names |
| 6 | Atlas of Living Australia | Exact species search and family-bound profile |
| 7 | Wikimedia Commons | Exact gallery/category VN templates, matching Wikidata ID required |
| 8 | Wikimedia | Existing Wikipedia language links and Wikispecies vernacular names |
| 9 | Open Tree | Batched exact scientific names and synonyms |
| 10 | German TMD | Bulk taxonomy/name projects |
| 11 | TAXREF France | Taxon name search |
| 12 | NCBI | Configured current taxonomy/name Parquet snapshots |
| 13 | Butterflies of India | Configured static English snapshot |
| 14 | Bharat Ki Titliya | Configured static Hindi snapshot |
| 15 | Karnataka Chitte | Configured static Kannada snapshot |
| 16 | Papilio demoleus multilingual table | Configured static T5 candidate snapshot |
| 17 | MyMemory | Dictionary-only translation of one eligible English common name per species |

These are 17 logical source families. Wikipedia and Wikispecies retain the
repository's existing single Wikimedia family. EOL, BOLD and LibreTranslate appear
as labels in existing policies but have no callable adapters in this workflow;
they are reported separately as implementation gaps. At launch, NCBI and all four
static snapshots are missing and are explicitly unavailable. ALA is attempted
through its public interface; authorization failures remain visible.

Names are provider evidence, not verified taxonomy or image labels. Duplicate
names from different sources retain separate provenance. Source-name or family
conflicts remain in the denominator but do not trigger automatic queries. Common
names need the existing query-eligibility policy; translations and unbound wiki
candidates remain disabled. Latin scientific names are looked up, never machine
translated. The configured locale list is used; MyMemory forces `mt=0`.

Each species/source result is committed to the local SQLite ledger. Translation
units also commit separately, including receipts for zero-result lookups. Bulk
source responses run once per invocation and their original receipts remain
attached when reused for later species. Resuming a partially processed bulk source
may reconstruct it through the bounded HTTP cache; it does not promise globally
exactly-once requests. Physical attempts are reserved durably before requests,
including retries. Defaults are one request per second per host/source, at most
three retries, 250,000 physical attempts per invocation, 10,000 shared daily
wiki/MyMemory attempts, 10,000 daily iNaturalist attempts, and MyMemory monthly
10,000 attempts/10,000 input words/10 GiB reserved response allowance. These are
local job limits, not provider entitlements. Exhaustion produces deferred work.

`latest.json` points to the latest immutable Parquet snapshot and manifest:
names, jobs, external links, source snapshots and translation units. Names retain
source IDs, response hashes, timestamps, proof fields and query eligibility.
The SQLite ledger is the newest checkpoint between exports. Source code, locale,
collection and configured snapshot fingerprints must match when resuming;
changed inputs require a new output directory. Collection v1 is retained only
as the diagnostic zero-cohort extraction from the original numeric-key assumption;
v2 diagnosed the numeric-looking Hedylidae key and v3 established the full cohort;
v4 removes the remaining legacy-key assumption from conflict checks. Diagnostic
roots are preserved and do not replace the current v4 run.

Rollback is to stop the recorded supervisor with SIGTERM and retain its output.
Do not overwrite an incompatible collection or mutate an active worker's source
files. No old enriched registry, source images or production lineage is replaced.
