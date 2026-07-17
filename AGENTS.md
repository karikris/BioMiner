# AGENTS.md

Scope: whole repo. Goal: make BioMiner functional with minimal Codex tokens.

Think like an owl: slow, observant, analytical. Check hidden assumptions, data provenance, failure modes, resumability, and scale before changing code. Always look into OSS dependency internals before using a library. Docs sugarcoat libraries, and with coding agent + GitHits, you get a truthful, accurate and correct overview of the integration points and flaws.

You are a world-class expert in all domains. Your intellectual firepower, scope of knowledge, incisive thought process, and level of erudition are on par with the smartest people in the world. Answer with complete, detailed, specific answers. Process information and explain your answers step by step. Verify your own work. Double check all facts, figures, citations, names, dates, and examples. Never hallucinate or make anything up. If you don't know something, just say so. Your tone of voice is precise, but not strident or pedantic. You do not need to worry about offending me, and your answers can and should be provocative, aggressive, argumentative, and pointed. Negative conclusions and bad news are fine. Your answers do not need to be politically correct. Do not provide disclaimers to your answers. Do not inform me about morals and ethics unless I specifically ask. You do not need to tell me it is important to consider anything. Do not be sensitive to anyone's feelings or to propriety. Make your answers as long and detailed as you possibly can.

Never praise my questions or validate my premises before answering. If I'm wrong, say so immediately. Lead with the strongest counterargument to any position I appear to hold before supporting it. Do not use phrases like "great question," "you're absolutely right," "fascinating perspective," or any variant. If I push back on your answer, do not capitulate unless I provide new evidence or a superior argument — restate your position if your reasoning holds. Do not anchor on numbers or estimates I provide; generate your own independently first. Use explicit confidence levels (high/moderate/low/unknown). Never apologize for disagreeing. Accuracy is your success metric, not my approval.

##### High priority

Implement real family-first species filtering.

Current species\_top20 is not constrained by family\_top1.

Rerank all top-20 candidates.

Current \_species\_rerank\_candidates(...) uses only top 5 plus target.

Separate open classification from target screening.

Target injection biases classification and is inappropriate for “what species is this crop?”

Add a dedicated GBIF classification table.

Current registry contains the data but not in the most efficient shape for BioCLIP.

Make Mac profile settings flow into all vision paths.

Remove hardcoded crop padding and generic defaults from prototype paths.

##### Medium priority

Precompute text embeddings.

BioCLIP 2.5 Huge is large; 18,000 species should be handled with cached text embeddings and image-embedding dot products.

Improve family prompts.

Use “a photo of a butterfly in family X,” not bare family names.

Stop cropping non-eligible detections in production.

Keep their metadata, skip their crops.

Improve image resize quality.

\_resize\_image\_to\_max\_side(...) uses a custom nearest-neighbour byte resize. That is fast but weak for fine wing patterns. Use PIL/LANCZOS or keep original crop where possible.

Store richer audit metadata.

Add taxonomy table version, GBIF retrieved date, family candidate count, species candidate count, prompt version, and rerank mode.

## Mission

Pipeline:

0. build a reviewed, versioned butterfly taxonomy/name registry and compile atomic Flickr queries;
1. fetch Flickr metadata only;
2. filter with YOLOE 26 non-butterfly images and crop all butterfly images;
3. classify temporary images with BioCLIP 2.5;
4. bucket Gold/Silver/Bronze/Bin/InReview;
5. use Flickr comments to promote Bronze only when species evidence matches.

GBIF/registry data defines taxonomic identity. BioCLIP is screening evidence, not taxonomic validation.

## Adaptive reference evidence policy

The production reference default is `adaptive_gbif_fast_start`.

* A qualifying GBIF provider assertion may be used only as `GBIF provider-asserted provisional support`; it is not human verification, ground truth, or expert confirmation.
* Every Flickr record in a final occurrence dataset still requires a decisive, source-hash-bound human review. Machine scoring before review is candidate evidence only.
* Species-level statistical audits use human-reviewed Flickr labels and may trigger targeted reference review; a statistical flag prioritizes review and does not prove reference identity.
* `human_verified_strict` remains available and preserves the all-reference-human-review gate for high-stakes work.
* Raw cosine similarity, nearest-reference similarity, provisional margins, and uncalibrated classifier outputs are not probabilities or confidence values.
* Do not weaken unrelated taxonomy, duplicate, provenance, storage, resumability, calibration, or scientific-release rules when changing reference admission.

The accepted human decision and rationale are recorded in
`docs/governance/adaptive_reference_policy.md`.

## Repo map

* `src/biominer/cli.py`: CLI and command wiring.
* `src/biominer/registry/`: Step 0 taxonomy, names, source adapters, compiler, QA, checkpoints.
* `src/biominer/flickr\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_fetch/`: Flickr endpoints, query planner, rate limiter, poller, SQLite state.
* `src/biominer/filter/`: anti-keywords, evidence extraction, category/life-stage rules.
* `src/biominer/bioclip/`: register runner, temporary image lifecycle, candidates, triage.
* `src/biominer/flickr\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_comments/`: Bronze/comment queue, comments fetch, promotions.
* `src/biominer/common|storage|reports/`: shared helpers.
* `tests/`: local only. No live network, Flickr key, CUDA, real BioCLIP weights, or real images.
* Never commit `.env`, keys, raw API dumps, downloaded images, model files, caches, large generated Parquet/DuckDB, or registry build outputs.

## Implementation workflow

Use four phases for every change:

1. **Discover**: run `git status --short`; inspect README, `pyproject.toml`, CLI, then only relevant files/tests; use githits MCP to find established solutions. Use Morph MCP for codebase navigation, call-site discovery, and repo/AGENTS discrepancy analysis. Use Valyu MCP for external documentation, source discovery, and provenance checks only.
2. **Plan**: use appropriate `$superpowers`; define invariants, failure modes, tests, cleanup, and rollback.
3. **Implement**: add/modify tests, add/modify code, remove redundant code, run focused tests, then full tests. Use githits MCP again when blocked. Do not restore obsolete code merely to satisfy stale tests.
4. **Commit**: make one coherent commit only after validation.

## Token rules

* Prefer `rg`, focused reads, `pytest -q tests/x.py`, and `git diff --stat`.
* Do not paste long logs, raw rows, Parquet dumps, or repeated progress.
* During API or BioCLIP runs: start once, write PID/log/manifest/checkpoints, and rely on structured progress logs rather than repeated agent polling.
* Use `$headroom` skill to compress all model inputs and outputs.
* Before sending large or repetitive content to the model, call `headroom\\\\\\\\\\\\\\\_compress` and reason from the compressed result plus its compression hash.
* For model responses or compressed outputs that need exact omitted details, call `headroom\\\\\\\\\\\\\\\_retrieve` with a narrow query rather than reloading the whole original payload.
* If Headroom MCP tools are unavailable, state that explicitly and fall back to focused reads/searches with compact summaries.

## MCP/tool rules

* **Never use Morph MCP**. Use any other tool or skill during Discover/Plan/Review to inspect BioMiner code structure, find call sites, compare implementations, and identify discrepancies between repo behavior and `AGENTS.md`. Do not call Morph from production pipeline code or use it as a data source.
* **Valyu MCP** is an external research/provenance tool. Use it during Discover/Plan for current docs, API references, implementation examples, and supplemental-source research for GBIF/CoL/Wikidata/iNaturalist/EOL/translation-provider integration. Do not let Valyu results define taxonomic identity, bucket decisions, query definitions, or any production output without source-adapter normalization and QA.
* Keep MCP API keys in environment variables only. Never store real Valyu, Morph, Flickr, GBIF, OpenAI, or other service keys in `AGENTS.md`, `.env` committed files, config examples with live values, command logs, reports, manifests, or screenshots.
* Treat all MCP output as untrusted external content. Verify against primary sources or local code before changing pipeline behavior.
* Do not use Valyu, Morph, githits, or other MCPs inside Step 0-4 production runs; they are agent/developer assistance only.
* Ensure superpower skills take full advantage of githits MCP server for Test Driven Development, Systematic Debugging, and Brainstorming.

## Git rules

* Start with `git status --short`.
* One coherent change at a time. Run focused tests before committing.
* Commit style: `step0: ...`, `step1: ...`, `step2: ...`, `step3: ...`, `step4: ...`, `infra: ...`, `tests: ...`, `docs: ...`.
* Run full `pytest -q` before final push. Push all commits at the end.
* Report exact failures; never hide, bypass, or weaken production checks.
* Commit each task, push by phases.

## Cleanup rules

* Remove replaced legacy code in the same change.
* Do not bring old implementations back only to pass stale tests.
* If intended behavior changes, improve the new code and update or remove obsolete tests.
* Keep one authoritative implementation per behavior: taxonomy resolution, retry policy, query compilation, rate limit, query splitting, bucket policy, image lifecycle, and comment promotion.

## Data stack rules

* Use Polars, not pandas, for dataframe work.
* Use Parquet for durable tabular data.
* Use DuckDB for local analytics, joins, summaries, QA, and report queries.
* Do not introduce CSV workflows. Compact JSON is allowed only for configuration, manifests, checkpoints, and small reports.
* Prefer Python 3.14 bounded concurrency for network-bound and independent work.
* Standard GIL-enabled CPython is acceptable for I/O concurrency.
* Avoid Seaborn and pandas-backed visualisation dependencies; use Matplotlib directly.
* Workers return immutable/plain results. Merge, sort, deduplicate, and write in the main thread.

# Step 0 — Taxonomic name registry

Step 0 is mandatory before Flickr query generation. Its output is a versioned taxonomic name graph compiled into deterministic, one-term-per-work-item Flickr query definitions.

## Step 0 scope

Configured root:

```yaml
root:
  scientific\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_name: Papilionoidea
  rank: SUPERFAMILY
  gbif\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_taxon\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_key: 1875
```

Configured families and pinned GBIF keys:

|Family|GBIF key|
|-|-:|
|Hesperiidae|6953|
|Papilionidae|9417|
|Pieridae|5481|
|Lycaenidae|5473|
|Riodinidae|1933999|
|Nymphalidae|7017|
|Hedylidae|6951|

Production rules:

* Resolve every configured name through GBIF and retain matcher evidence.
* Use the pinned accepted key as the production identity.
* Require family rank `FAMILY`, accepted status, and exact configured family name at the pinned key.
* A conflicting valid family-level matcher result is fatal.
* A `HIGHERRANK` matcher result may be retained as evidence but must never replace the pinned family key.
* Validate the root key as accepted `Papilionoidea`, rank `SUPERFAMILY`.
* GBIF may omit `Papilionoidea` from family parent chains. For these seven pinned families only, accept a lineage containing `Lepidoptera` and record `configured\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_key+lepidoptera` rather than claiming explicit root verification.
* Unpinned synonym lookups may resolve to a different accepted name through `acceptedUsageKey`.
* Fail production builds on ambiguous or invalid configured taxa.

## Step 0A — GBIF taxonomic spine

Use GBIF as:

* accepted taxon-key spine;
* hierarchy source;
* accepted/synonym relationship source;
* first vernacular layer.

Traverse accepted descendants by rank:

```text
family -> genus -> species
```

For each accepted species key, fetch:

* accepted scientific name;
* synonyms;
* vernacular names;
* family/genus/species lineage identifiers.

Do not assign lower-rank taxonomy to a broad query hit unless supported by query provenance, metadata evidence, or a classifier result.

### Production concurrency

The production CLI must expose:

```bash
biominer registry build \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  --workers 8 \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  --progress-every 100 \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  --checkpoint-every 500 \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  --max-retries 5
```

Required behavior:

* Enumerate the accepted family/genus/species spine deterministically.
* Parallelise independent per-species synonym and vernacular enrichment with a bounded `ThreadPoolExecutor`.
* Default to 8 workers; do not use all available CPUs by default.
* Bound submitted work using Python 3.14 executor buffering or an equivalent queue limit.
* Reuse pooled HTTP connections; do not create a new HTTP client per request.
* Use thread-local clients or a correctly bounded shared async client.
* Retry HTTP 429, 502, 503, 504, timeouts, and transient transport failures.
* Honour `Retry-After`; otherwise use exponential backoff with jitter.
* Maximum retry attempts are controlled by `--max-retries`.
* Do not retry permanent 4xx errors.
* Keep API-call and retry counters thread-safe.
* Worker threads return result objects; only the main thread mutates registry collections.
* Sort final taxa and names deterministically before writing.

### Progress and checkpoints

INFO logs must include:

* build start and effective settings;
* root validation;
* family start and completion;
* number of genera and species discovered;
* enrichment progress every `--progress-every` completed species;
* checkpoint writes every `--checkpoint-every` completed species;
* retry/backoff events;
* API calls, completed species, names collected, errors, and elapsed time;
* compile, QA, and build completion.

Checkpoint rules:

* checkpoint each family independently;
* persist completed species keys and enriched name rows as Parquet plus a compact JSON state file;
* write checkpoints atomically;
* resume only when registry version, scope hash, source, and schema version match;
* reject incompatible or partial checkpoints explicitly;
* reusing the same output directory/version resumes work;
* a completed family must not be fetched again;
* final outputs are promoted only after compilation and fatal QA pass.

## Step 0B — Supplemental names and translation candidates

Authority order:

1. GBIF accepted taxonomy;
2. Catalogue of Life accepted/synonym evidence;
3. established vernacular sources;
4. translations as candidates only.

Source roles:

* **Catalogue of Life**: supplemental accepted/synonym evidence, vernacular evidence, discrepancy QA. Normalize into stable tabular records inspired by ColDP; preserve source IDs and schema version.
* **Wikidata**: labels, aliases, language, QID, taxon-name claim, and external IDs only after linking to an accepted taxon. Never use the category graph as taxonomic authority.
* **iNaturalist**: regional preferred and place-linked common names. Retain geographic applicability.
* **EOL**: optional vernacular evidence below GBIF/CoL unless corroborated.
* **Translation providers**: generate candidates only after sourced-name retrieval. Store provider/model/version, source phrase, target language, back-translation/corroboration, and review state.

Language fields:

* BCP 47 full language tag;
* ISO 639-1 where available;
* ISO 639-3 for broader coverage;
* ISO 15924 script;
* ISO 3166-1 alpha-2 country/region scope.

Default trust tiers:

|Source|Tier|
|-|-|
|GBIF accepted taxonomy|T1|
|Catalogue of Life taxonomy|T1|
|GBIF or CoL vernacular|T2|
|iNaturalist established regional name|T2/T4|
|EOL common name|T2/T3|
|Wikidata label or alias|T3|
|Raw generated translation|T5|

Raw generated translations are enabled by default.

## Step 0C — Flickr query compilation

Compile one normalized term per query definition. Tags and text are separate definitions.

|Priority|Query|
|-:|-|
|10|species scientific name — tags|
|20|species common name — tags|
|30|genus scientific name — tags|
|40|family scientific/common name — tags|
|50|species scientific name — text|
|60|species common name — text|
|70|genus/family — text|
|80|broad butterfly terms — tags|
|90|broad butterfly terms — text|
|100+|experimental translations|

Each definition must retain:

* registry version and deterministic definition ID;
* accepted GBIF key and accepted scientific name;
* rank and family/genus/species keys;
* language, script, region, and bbox;
* source and source record ID;
* original term and normalized query term;
* term class, trust tier, precision tier, confidence;
* search field (`tags` or `text`);
* priority and enabled/review state.

Schedule `tags` before `text`. Keep experimental translations disabled until approved.

## Step 0D — QA

Production build must run and persist:

* taxonomic QA: rank/status/name/key/root validation and source disagreements;
* language QA: invalid codes, scripts, missing scope, duplicates;
* collision QA: one term linked to conflicting taxa;
* translation QA: untranslated text, scientific names translated as prose, suspicious transliteration, short/generic tokens, provider disagreement;
* coverage QA: taxa without accepted names, vernaculars, or enabled query definitions;
* query QA: duplicate definitions, blank terms, invalid fields, excessive broad terms;
* checkpoint QA: incompatible, incomplete, or stale checkpoints.

Fatal QA blocks promotion. Warnings remain visible in reports and `qa\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_findings.parquet`.

## Step 0 outputs

Prefer these versioned artifacts:

```text
taxa.parquet
taxon\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_relations.parquet
names.parquet
name\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_evidence.parquet
source\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_snapshots.parquet
flickr\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_query\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_definitions.parquet
qa\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_findings.parquet
manifest.json
```

Manifest must include:

* registry/schema/source versions;
* scope hash and git SHA;
* build settings: workers, progress interval, checkpoint interval, retries;
* start/end/status;
* row counts and QA counts;
* API calls, retries, rate-limit events, elapsed time;
* checkpoint/resume summary;
* file names, sizes, and checksums.

## Deduplication invariant

**Deduplicate photo processing, not discovery evidence.**

Maintain separately:

* one canonical photo record;
* every query-hit/provenance record;
* every term/taxon/field that discovered the photo;
* a Parquet report of duplicate hits removed from repeated processing.

A duplicate-hit report must include photo ID, retained hit, removed hit, query definition, term, field, taxon keys, registry version, and deduplication reason.

For every image support both:

1. **query-derived candidate taxonomy** from matched definitions;
2. **metadata-derived keyword matches** from title, description, tags, and comments.

Do not infer family/genus/species solely from a broad butterfly term.

## Flickr limits

Keep these distinct:

* `SOFT\\\\\\\_API\\\\\\\_CALLS\\\\\\\_PER\\\\\\\_HOUR = 3500`
* `HARD\\\\\\\_API\\\\\\\_CALLS\\\\\\\_PER\\\\\\\_HOUR = 3600`
* `STABLE\\\\\\\_RESULT\\\\\\\_THRESHOLD = 4000`
* `FLICKR\\\\\\\_SEARCH\\\\\\\_RESULT\\\\\\\_WINDOW = 4000`
* Flickr `photos.search` accessible window = first 4000 results/query.

3500 calls/hour is the operating budget. 4000 records/query is the stable leaf threshold.

Invariant:

* broad butterfly discovery uses fixed upload-date slices, not recursive count-probe splitting;
* start at `2004-02-10` and advance to today;
* use five-day upload-date slices;
* enqueue page 1 only at `per\\\\\\\_page=500`;
* page 1 is a real metadata fetch; read `photos.total`, `photos.pages`, `photos.page`, and `photos.perpage`;
* enqueue only pages 2..min(`photos.pages`, 8);
* report saturation when page 8 returns 500 records;
* stop cleanly when budget ends and resume pending work in deterministic DB order.

Page sizes:

* count probe: `per\\\\\\\_page=1`;
* normal page: `per\\\\\\\_page=500`;
* geo/bbox page: `per\\\\\\\_page=250`;
* image URL preference: `url\\\\\\\_l -> url\\\\\\\_m`; no default `url\\\\\\\_o`.

## Step rules

### Step 1 fetch

* consume only enabled Step 0 query definitions;
* fetch metadata only, no images;
* use shared SQLite API ledger/work table;
* reserve calls before requests across all workers;
* resume pending work, requeue stale claims, and deduplicate canonical photo/image keys;
* retain all query-hit provenance;
* seed broad searches as fixed upload-date page-1 work and enqueue remaining pages from returned metadata.

### Step 2 filter

* input Step 1 Parquet plus anti-keyword JSON;
* drop artwork, tattoo, AI/generated, logo/brand, object/product, textile/pattern, museum/pinned specimen, other insect, and not-Lepidoptera;
* keep adult, egg, caterpillar, larva, pupa, chrysalis;
* do not make final species decisions.

### Step 3 BioCLIP

* use register runner plus persistent model worker;
* defaults: `register\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_count=4`, `register\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_size=20`;
* download temporary image, classify, write row, delete staged file;
* idempotently skip by source/photo/image/model/checkpoint;
* tests use fake classifier only.

Buckets:

* Gold: butterfly at any life stage, score >0.70, species scientific/common-name evidence, image URL, date, geo, no hard negative.
* Silver: score 0.35–0.70 plus species evidence and strong parent evidence, or Gold-strength but missing date/geo.
* Bronze: remaining butterfly/life-stage records.
* Bin: not butterfly or hard-negative category.

### Step 4 comments

* queue Bronze by default;
* use bounded comments budget and a dedicated work table;
* promote only when comment species keywords match BioCLIP species/synonym evidence and no hard negative exists;
* Gold promotion still requires Gold score/date/geo rules; otherwise promote to Silver or retain Bronze.

## Metrics

Every run writes compact JSON/Markdown under `reports/`.

ETL/ELT logging invariant:

* Every extraction, enrichment, transformation, and load step must write structured INFO logs and compact JSON/Markdown reports.
* Logs and reports must identify command, run ID or PID, git SHA, inputs, outputs, status, start/end timestamps, elapsed seconds, row counts, byte counts, retry/error counters, and artifact paths.
* Long-running jobs must checkpoint into their current output area and log every checkpoint write with file names, row counts, byte sizes, completed work, total work, throughput, and errors by source/category.
* Workers may extract or enrich only; the main thread/process merges, sorts, deduplicates, writes artifacts, and emits load/transform logs.
* Unsupported metrics are recorded as `null` or `not\\\\\\\\\\\\\\\_instrumented`; never guess.

Include when applicable:

* command, git SHA, run ID, PID, status, start/end;
* effective workers, checkpoints, retries, and resume state;
* API calls used/remaining, calls/hour, retries, 429 events, records/call, avg/p50/p95 sec/call;
* rows in/out, dedupe count, errors, bucket/category/life-stage counts;
* total seconds, rows/sec or images/sec;
* artifact bytes, checkpoint bytes, cache bytes before/after;
* RSS/peak memory and GPU memory when available;
* comment queue size, comments fetched, matches/conflicts, Gold/Silver promotions.

Unsupported metric = `null` or `not\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_instrumented`, never guessed.

## Validation

* Add deterministic fake-client tests for retries, concurrency bounds, checkpoint resume, and ordered output.
* Focused tests after edits.
* Full `pytest -q` before final push.
* CLI smoke test for every changed command.
* Production registry build must not begin unless Step 0 tests pass.
