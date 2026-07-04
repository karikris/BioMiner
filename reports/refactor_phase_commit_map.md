# BioMiner Refactor Phase Commit Map

Recorded: 2026-07-05

Current branch: `main`

Implementation base used for this phase map: `e6a498b93ba9`

The original plan requested branch `cleanup/production-workflow-postgres-s3`,
but later operator instructions made `main` the source of truth and required
cleanup commits to be pushed there. All commits listed below are present on
`origin/main` at the implementation base above.

Several later hardening commits intentionally revisit earlier phase areas.
They are grouped by the requirement they most directly prove, not strictly by
timestamp.

## Phase Status

| Phase | Status | Pushed branch |
| --- | --- | --- |
| Phase 0 - Repository and environment audit | Complete | `main` |
| Phase 1 - Production architecture skeleton | Complete | `main` |
| Phase 2 - S3/Postgres defaults | Complete | `main` |
| Phase 3 - Registry workflow cleanup and enrichment policy | Complete with operator-approved T5 override | `main` |
| Phase 4 - Rank-aware production run orchestration | Complete | `main` |
| Phase 5 - YOLOE/YOLO26 object-first vision cleanup | Complete | `main` |
| Phase 6 - BioCLIP object scoring as visual classifier | Complete | `main` |
| Phase 7 - Metadata flags and object bucket rules | Complete | `main` |
| Phase 8 - CLI and docs cleanup | Complete | `main` |
| Phase 9 - Test cleanup and fake-backed workflow coverage | Complete | `main` |
| Phase 10 - Final audits and PR readiness evidence | Complete | `main` |

## Phase 0 Commits

- `f28aa84` chore: audit refactor environment and workflow surface
- `beb922c` chore: add refactor guardrails and artifact ignore rules
- `e745e74` Merge branch 'cleanup/production-workflow-postgres-s3'

## Phase 1 Commits

- `e39e974` feat(run): add production run architecture skeleton
- `9303fef` refactor(evidence): introduce object evidence package
- `e2ec9f4` feat(registry): add central T1-T5 trust policy
- `e8bc299` test: cover production run skeleton and trust policy

## Phase 2 Commits

- `b7c1754` refactor(config): default production storage to S3 and Postgres
- `bf0c894` fix(storage): validate S3 and Postgres production configuration
- `d11f568` docs: document S3 and Postgres production storage defaults
- `e2f39e8` test: update storage and workstore default expectations
- `593fc84` fix(config): require production worker id
- `1fa50e3` fix(config): fully redact production credentials

## Phase 3 Commits

- `ff1eaf3` refactor(cli): remove low-level registry internals from public workflow
- `af2b384` feat(registry): add Wikidata enrichment scaffold
- `87d6b8d` feat(registry): add disabled T5 translation candidate layer
- `f1d4deb` refactor(registry): enforce trust policy during registry compilation
- `2ffb3c5` refactor(flickr): require explicit registry query seeding
- `9e50d87` test(registry): cover trust-gated enrichment queries
- `4ccdfff` feat(registry): apply trust policy to translation candidates
- `a703e98` feat(registry): include wikidata in default enrichment
- `ba18ad5` feat(registry): route T5 terms to Flickr retrieval
- `a03d495` refactor(registry): enable T5 translation names
- `b563e08` feat(registry): report enabled T5 metrics
- `e6a7685` docs(registry): align T3 trust policy

## Phase 4 Commits

- `cc74101` feat(run): resolve family genus and species scopes from registry
- `5f6ae83` feat(run): add production run manifest and S3-safe paths
- `865951b` feat(run): add production workflow orchestrator skeleton
- `20f3396` feat(cli): add rank-aware production run command
- `df77a63` test: cover rank-aware production run workflow
- `0764de7` feat(run): enqueue registry Flickr work through workstore
- `5a783ab` feat(run): join evidence and summarize local artifacts
- `03711c9` feat(run): execute local detect and score stages
- `79c9e43` feat(run): execute local Flickr poll stage
- `0f84e98` feat(run): validate registry stage artifacts
- `95437b1` refactor(run): absorb species registry resolver
- `94fc45b` feat(run): write review queue summaries
- `437bc47` feat(run): write compiled queries through storage backend
- `56473fc` feat(run): write summaries through storage backend
- `cf297f2` feat(run): join evidence through storage backend
- `e638e64` feat(run): persist cloud run manifests
- `5313bc5` feat(run): detect objects through storage backend
- `57bab82` feat(run): score BioCLIP objects through storage backend
- `5c91e3a` refactor(run): resolve registry scopes from frames
- `ce4c276` feat(run): read registry artifacts from storage
- `18a9575` feat(run): poll Flickr work through workstore
- `efcbe38` test(run): cover fake cloud workflow end to end
- `3f31b53` chore(run): remove stale cloud registry diagnostic
- `9fd2b9e` fix(run): propagate validated worker id
- `67fb888` feat(run): score all object visual modes
- `15ec7dd` fix(run): require registry query definitions
- `59d026a` fix(run): scope Flickr queries to taxon run
- `6a4a838` fix(run): apply species limit to taxon scope

## Phase 5 Commits

- `5b6edba` refactor(detection): enforce coarse object detector contract
- `fd67c14` feat(detection): add YOLOE-26 coarse object backend
- `11a940c` feat(detection): add YOLO26 inference compatibility scaffold
- `29564a8` refactor(detection): remove legacy YOLOv8 production backend
- `92e7500` feat(vision): support detector crop segmentation without enhancement
- `c2d2e03` refactor(cli): simplify vision command surface
- `87a2bf5` test: update object-first vision workflow tests
- `1f7c05d` refactor(detection): remove legacy yolov8 backend
- `3dc729a` refactor(detection): reject taxonomic detector labels
- `7b4b508` feat(detection): expose yolo26 inference backend
- `7429da7` test(detection): guard against legacy yolo backend
- `3790e69` test(detection): guard against reviewed box training artifacts

## Phase 6 Commits

- `46a5184` refactor(bioclip): promote object scoring as primary classifier
- `456d69b` refactor(bioclip): remove whole-image register workflow from public CLI
- `31bed0e` feat(bioclip): build candidate sets from taxon scope registry data
- `25a7355` refactor(bioclip): require expanded species candidates
- `c936482` docs: align BioCLIP workflow wording with object scoring
- `36ec19b` refactor(bioclip): keep metadata flags soft in triage
- `8beac4f` test(bioclip): guard segmentation visual mode contract

## Phase 7 Commits

- `a622001` refactor(filter): convert keyword filtering to metadata flags
- `7fac6f5` refactor(evidence): make object buckets the rule engine
- `991b5dd` refactor(cli): remove hard-drop filter and legacy apply-rules commands
- `0ad140c` test: update metadata flag and evidence bucket behaviour
- `c68aed9` refactor(evidence): route metadata object hints to review
- `ad423e1` refactor(evidence): centralize object bucket policy
- `7756d06` refactor(evidence): keep metadata hints soft for object buckets
- `c1c6d5a` refactor(flickr): require explicit initial query work
- `09dbd55` refactor(filter): emit metadata flags only
- `f2250b5` refactor(evidence): soften metadata negatives

## Phase 8 Commits

- `ae5cdac` refactor(cli): add evidence join command surface
- `1214173` refactor(cli): route evidence joins through evidence command
- `5aaa414` refactor(cli): remove bioclip object workflow aliases
- `0bc62c8` refactor(cli): remove legacy local maintenance commands
- `a4c4548` refactor(cli): remove ad hoc report commands
- `ad66150` test: remove obsolete report command coverage
- `9d25e5f` feat(cli): add storage and workstore doctor commands
- `adb2c67` refactor(cli): replace cloud command with storage and workstore doctors
- `7e9c347` docs: remove anti keyword config path
- `f2b60f0` refactor(cli): remove species command namespace
- `5f42144` refactor(cli): demote fetch and comment commands to dev
- `da84b40` refactor(cli): consolidate model helpers under vision
- `8fdf3a9` docs: refresh refactor command audit
- `7081977` docs: add production workflow guides
- `bcf3925` docs: add production run examples
- `bfce37a` docs: remove broad probe workflow from readme
- `2c67c6d` docs: remove yolo training dataset recommendation
- `910e177` docs: remove anti keyword path wording
- `29987d5` docs: refresh production workflow audit reports
- `db4c8df` docs: refresh final refactor audits
- `ab0d354` docs: remove stale anti keyword path references
- `3371ec3` docs: refresh removed workflow audit
- `0c1c1f2` docs: align examples with production run workflow
- `874fdd0` docs: refresh refactor audit base
- `41f53a6` docs: add refactor migration notes

## Phase 9 Commits

- `0f37ae7` refactor(flickr): require explicit discovery seed terms
- `f4f5ee6` refactor(reports): remove keyword name evidence report path
- `daf6c02` refactor(species): remove legacy species workflow compiler
- `a33ef65` refactor(evidence): remove legacy filter rule wrappers
- `d141d5f` refactor(filter): remove metadata keyword path helpers
- `09603a8` refactor(flickr): remove built-in multilingual seed terms
- `a6f7bcb` refactor(flickr): remove broad discovery seed planner
- `93cdf5c` refactor(config): remove legacy keyword config loader
- `05fda2b` test(flickr): cover T5 retrieval queries
- `704e5e6` test(flickr): guard against broad seed planner regressions

## Phase 10 Commits

- `77c077e` docs: add final command surface audit
- `080e5d4` docs: record removed legacy workflow paths
- `92e613b` docs: add production refactor completion audit
- `353042c` docs: refresh production refactor completion audit
- `e6a498b` docs: clarify verified refactor audit base

## Verification Snapshot

Latest full-suite verification while adding this phase map:

```text
uv run --extra test pytest -q
530 passed
```

Latest source-of-truth alignment at the implementation base:

```text
git status --short --branch
## main...origin/main

git rev-parse HEAD origin/main
e6a498b93ba9f8fe874cb65a36d522048d01274d
e6a498b93ba9f8fe874cb65a36d522048d01274d
```

The report-only commit that adds this phase map is itself a Phase 10 audit
commit and is expected to appear after the implementation base in `git log`.

## Policy Deviations From Original Plan

- Branching: cleanup was completed and pushed on `main` after later operator
  instructions superseded the original cleanup branch requirement.
- T5 translations: the original plan disabled T5 by default, but later
  operator instructions explicitly enabled T5 as accepted names. Current code
  keeps `trust_tier = T5` provenance, compiles enabled T5 names into normal
  Flickr query definitions, and records enabled T5 metrics.
