# AGENTS.md

Scope: whole repo. Goal: make BioMiner functional with minimal Codex tokens.

Think like an owl — slow, observant and analytical. Examine all problems from multiple perspectives and identify the hidden factors most people overlook.

## Mission

Pipeline:

1. fetch Flickr metadata only;
2. filter obvious non-biodiversity;
3. classify temporary images with BioCLIP 2.5;
4. bucket Gold/Silver/Bronze/Bin/InReview;
5. use Flickr comments to promote Bronze only when species evidence matches.

BioCLIP = screening evidence, not taxonomic validation. No Darwin Core publishing claims.

## Repo map

* `src/biominer/cli.py`: CLI.
* `src/biominer/flickr\\\\\\\_fetch/`: Flickr endpoints, query planner, rate limiter, poller, SQLite state.
* `src/biominer/filter/`: anti-keywords, evidence extraction, category/life-stage rules.
* `src/biominer/bioclip/`: register runner, temp image cache/delete, candidates, triage.
* `src/biominer/flickr\\\\\\\_comments/`: Bronze/comment queue, comments fetch, promotions.
* `src/biominer/common|storage|reports/`: shared helpers.
* `tests/`: local only. No network, Flickr key, CUDA, real BioCLIP weights, or real images.
* Never commit `.env`, keys, raw API dumps, downloaded images, model files, caches, large parquet/DuckDB, generated data.

## Implementation rules

Use a four-step implementation pattern:

1 - Use githits mcp to explore known solutions

2 - Use $superpowers to plan the work

3 - Implement in 4 phases - add tests, add/modify code, remove any redundant code, test. Do not bring old code back to pass a test. use githits mcp for alternative solutions until problem is solved and only after remove or modify tests.

4 - commit

## Token rules

* Read README, pyproject, CLI, then only touched files/tests.
* Prefer `rg`, focused reads, `pytest -q tests/x.py`, `git diff --stat`.
* Do not paste long logs, raw rows, parquet dumps, or repeated progress.
* During API fetch or BioCLIP runs: start process, write PID/log/manifest, no tail loops, no repeated polling, no status chatter.
* Always use the $headroom skill

## Git rules

* Start: `git status --short`.
* One coherent change at a time. Run focused tests. Commit each change.
* Commit style: `step1: ...`, `step2: ...`, `step3: ...`, `step4: ...`, `infra: ...`, `tests: ...`, `docs: ...`.
* Run full `pytest -q` before final push. Push all changes at end only.
* If tests fail, report exact failures; do not hide or bypass them.

## Cleanup rules

* When replacing code, remove redundant legacy code in the same change.
* Do not restore old code only to satisfy stale tests.
* If tests fail after intended replacement: improve new code, redesign tests, or remove obsolete tests.
* Keep one authoritative implementation per behavior: rate limit, query split, bucket policy, image lifecycle, comment promotion.

## Data stack rules

* Prefer Polars over pandas for dataframe work because BioMiner handles large metadata and classification tables where memory efficiency matters.
* Prefer Parquet for durable tabular storage.
* Prefer DuckDB as the query engine for local analytics, joins, summaries, and report queries over loading large tables into memory.
* Do not add new pandas usage unless a dependency boundary truly requires it; when touching pandas-based legacy code, replace it with Polars/DuckDB where practical and remove redundant pandas tests/imports in the same change.

## Flickr limits

Separate these:

* `SOFT\\\\\\\_API\\\\\\\_CALLS\\\\\\\_PER\\\\\\\_HOUR = 3500`
* `HARD\\\\\\\_API\\\\\\\_CALLS\\\\\\\_PER\\\\\\\_HOUR = 3600`
* `STABLE\\\\\\\_RESULT\\\\\\\_THRESHOLD = 4000`
* `FLICKR\\\\\\\_SEARCH\\\\\\\_RESULT\\\\\\\_WINDOW = 4000`
* Flickr `photos.search` accessible window = first 4000 results/query.

3500 API calls/hour is budget. 4000 records/query is the maximum stable BioMiner leaf threshold because Flickr documents only the first 4000 results/query as accessible.

Invariant:

* broad butterfly discovery uses fixed upload-date slices, not recursive count-probe splitting;
* start at `2004-02-10` and advance to today;
* use 5-day upload-date slices for the full range;
* enqueue page 1 only for each slice at `per\\\\\\\_page=500`;
* page 1 is a real metadata fetch; read `photos.total`, `photos.pages`, `photos.page`, and `photos.perpage`;
* after page 1 returns, enqueue only pages 2..min(`photos.pages`, 8);
* report a slice as saturated when page 8 returns 500 records;
* if budget ends, stop cleanly; next run resumes pending work in deterministic DB order.

Page sizes:

* count probe `per\\\\\\\_page=1`;
* normal page `per\\\\\\\_page=500`;
* geo/bbox page `per\\\\\\\_page=250`;
* image URL preference `url\\\\\\\_l -> url\\\\\\\_m`; no default `url\\\\\\\_o`.

Example: `text=butterfly` starts with upload-date slice `2004-02-10..2004-02-14`, page 1, then enqueues only the remaining pages Flickr reports for that slice.

## Step rules

Step 1 fetch:

* metadata only, no images;
* use shared SQLite API ledger/work table;
* reserve call before request across all workers;
* resume pending, requeue stale claims, dedupe photo/image keys;
* seed broad searches as fixed upload-date page-1 work and dynamically enqueue remaining pages from page metadata.

Step 2 filter:

* input Step 1 parquet + anti-keyword JSON;
* drop artwork, tattoo, AI/generated, logo/brand, object/product, textile/pattern, museum/pinned specimen, other insect, not Lepidoptera;
* keep adult, egg, caterpillar, larva, pupa, chrysalis;
* do not make final species decisions.

Step 3 BioCLIP:

* use register runner + persistent model worker;
* defaults `register\\\\\\\_count=4`, `register\\\\\\\_size=20`;
* download temporary image, classify, write row, delete staged file;
* idempotent skip by source/photo/image/model/checkpoint;
* tests use fake classifier only.
Buckets:
* Gold: butterfly (any life stage), score >0.70, species (scientific, common name any language) text or tags match, image URL, date, geo, no hard negative.
* Silver: score 0.35-0.70 + species (scientific or common name in any language) + min 1 family, genus etc parent classifications >0.90 in text match, or Gold-strength but missing date/geo.
* Bronze: remaining butterfly/life-stage records.
* Bin: not butterfly or hard-negative category.

Step 4 comments:

* queue Bronze by default;
* use bounded comments budget + own work table;
* promote only if comment species keywords matches BioCLIP species/synonym and no hard-negative;
* Gold promotion still needs Gold score/date/geo rules; otherwise Silver or remain Bronze.

## Metrics

Every run writes compact JSON/MD under `reports/`.
Always include when applicable:

* command, git SHA, run\_id, PID, status, start/end;
* API calls used/remaining, calls/hour, records/call, avg/p50/p95 sec/call;
* rows in/out, dedupe count, errors, bucket/category/life-stage counts;
* total sec, rows/sec or images/sec;
* artifact bytes, cache bytes before/after;
* RSS/peak memory, GPU memory if available;
* comment queue size, comments fetched, matches/conflicts, Gold/Silver promotions.
Unsupported metric = `null` or `not\\\\\\\_instrumented`, never guessed.

## Validation

* Focused tests after edits.
* Full `pytest -q` before final push.
* CLI smoke for changed commands.
