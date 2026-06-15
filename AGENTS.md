# AGENTS.md

Scope: whole repo. Goal: make BioMiner functional with minimal Codex tokens.

## Mission
Pipeline:
1. fetch Flickr metadata only;
2. filter obvious non-biodiversity;
3. classify temporary images with BioCLIP 2.5;
4. bucket Gold/Silver/Bronze/Bin/InReview;
5. use Flickr comments to promote Bronze only when species evidence matches.

BioCLIP = screening evidence, not taxonomic validation. No Darwin Core publishing claims.

## Repo map
- `src/biominer/cli.py`: CLI.
- `src/biominer/flickr_fetch/`: Flickr endpoints, query planner, rate limiter, poller, SQLite state.
- `src/biominer/filter/`: anti-keywords, evidence extraction, category/life-stage rules.
- `src/biominer/bioclip/`: register runner, temp image cache/delete, candidates, triage.
- `src/biominer/flickr_comments/`: Bronze/comment queue, comments fetch, promotions.
- `src/biominer/common|storage|reports/`: shared helpers.
- `tests/`: local only. No network, Flickr key, CUDA, real BioCLIP weights, or real images.
- Never commit `.env`, keys, raw API dumps, downloaded images, model files, caches, large parquet/DuckDB, generated data.

## Token rules
- Read README, pyproject, CLI, then only touched files/tests.
- Prefer `rg`, focused reads, `pytest -q tests/x.py`, `git diff --stat`.
- Do not paste long logs, raw rows, parquet dumps, or repeated progress.
- During API fetch or BioCLIP runs: start process, write PID/log/manifest, then stop. No tail loops, no repeated polling, no status chatter.

## Git rules
- Start: `git status --short`.
- One coherent change at a time. Run focused tests. Commit each change.
- Commit style: `step1: ...`, `step2: ...`, `step3: ...`, `step4: ...`, `infra: ...`, `tests: ...`, `docs: ...`.
- Run full `pytest -q` before final push. Push all changes at end only.
- If tests fail, report exact failures; do not hide or bypass them.

## Cleanup rules
- When replacing code, remove redundant legacy code in the same change.
- Do not restore old code only to satisfy stale tests.
- If tests fail after intended replacement: improve new code, redesign tests, or remove obsolete tests.
- Keep one authoritative implementation per behavior: rate limit, query split, bucket policy, image lifecycle, comment promotion.

## Flickr limits
Separate these:
- `SOFT_API_CALLS_PER_HOUR = 3500`
- `HARD_API_CALLS_PER_HOUR = 3600`
- `STABLE_RESULT_THRESHOLD = 4000`
- `FLICKR_SEARCH_RESULT_WINDOW = 4000`
- Flickr `photos.search` accessible window = first 4000 results/query.

3500 API calls/hour is budget. 4000 records/query is the maximum stable BioMiner leaf threshold because Flickr documents only the first 4000 results/query as accessible.

Invariant:
- broad butterfly discovery uses fixed upload-date slices, not recursive count-probe splitting;
- start at `2004-02-10` and advance to today;
- use 5-day upload-date slices for the full range;
- enqueue pages 1..8 for each slice at `per_page=500`;
- report a slice as saturated when page 8 returns 500 records;
- if budget ends, stop cleanly; next run resumes pending work in deterministic DB order.

Page sizes:
- count probe `per_page=1`;
- normal page `per_page=500`;
- geo/bbox page `per_page=250`;
- image URL preference `url_l -> url_m`; no default `url_o`.

Example: `text=butterfly` starts with upload-date slice `2004-02-10..2004-02-14`, pages 1..8, then resumes with the next pending deterministic date slice.

## Step rules
Step 1 fetch:
- metadata only, no images;
- use shared SQLite API ledger/work table;
- reserve call before request across all workers;
- resume pending, requeue stale claims, dedupe photo/image keys;
- seed broad searches as fixed upload-date page work.

Step 2 filter:
- input Step 1 parquet + anti-keyword JSON;
- drop artwork, tattoo, AI/generated, logo/brand, object/product, textile/pattern, museum/pinned specimen, other insect, not Lepidoptera;
- keep adult, egg, caterpillar, larva, pupa, chrysalis;
- do not make final species decisions.

Step 3 BioCLIP:
- use register runner + persistent model worker;
- defaults `register_count=4`, `register_size=20`;
- download temporary image, classify, write row, delete staged file;
- idempotent skip by source/photo/image/model/checkpoint;
- tests use fake classifier only.
Buckets:
- Gold: adult butterfly, score >0.70, species text match, image URL, date, geo, no hard negative.
- Silver: score 0.35-0.70 + species text match, or Gold-strength but missing date/geo.
- Bronze: remaining butterfly/life-stage records.
- Bin: no butterfly any life stage or hard-negative category.

Step 4 comments:
- queue Bronze by default;
- use bounded comments budget + own work table;
- promote only if comment species matches BioCLIP species/synonym and no hard-negative;
- Gold promotion still needs Gold adult/date/geo rules; otherwise Silver or remain Bronze.

## Metrics
Every run writes compact JSON/MD under `reports/`.
Always include when applicable:
- command, git SHA, run_id, PID, status, start/end;
- API calls used/remaining, calls/hour, records/call, avg/p50/p95 sec/call;
- rows in/out, dedupe count, errors, bucket/category/life-stage counts;
- total sec, rows/sec or images/sec;
- artifact bytes, cache bytes before/after;
- RSS/peak memory, GPU memory if available;
- comment queue size, comments fetched, matches/conflicts, Gold/Silver promotions.
Unsupported metric = `null` or `not_instrumented`, never guessed.

## Validation
- Focused tests after edits.
- Full `pytest -q` before final push.
- CLI smoke for changed commands.
