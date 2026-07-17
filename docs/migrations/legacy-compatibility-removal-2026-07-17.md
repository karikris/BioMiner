# Legacy compatibility removal — 2026-07-17

Status: required breaking cutover for current BioMiner producers and durable
artifacts.

This release removes compatibility-only readers, inferred defaults, forwarding
modules, and unused implementations. It does not rewrite historical artifacts.
Keep the prior capable release and its artifact roots if historical data must be
read. The current release fails closed rather than silently reinterpreting it.

## What production now requires

| Surface | Current requirement | Removed compatibility behaviour |
|---|---|---|
| Flickr query definitions | Every non-empty definition declares `query_eligible`. | Eligibility inferred from generated-query metadata. |
| Detected object rows | A structured `detector_prompt` supplies route evidence. | Routing from a coarse historical detector label alone. |
| BioCLIP score gating | One fixed, structured route-aware gate; detected objects only. | Label-only gate modes, a runtime gate selector, and whole-image scoring when no object was detected. |
| Reference media inventory | The exact current reference-media-object schema. | In-place legacy inventory upgrades. |
| Reference download checkpoint | The exact current checkpoint schema and its complete binding/evidence/commit/object checks. | V1 checkpoint upgrades, field backfill, and synthesized fingerprints. |
| Reviewed labels | The complete `reviewed-labels-v2` schema, including target-aware fields. | V1 label reading and runtime inference of `target_present`. |
| Python imports | Owning modules and `biominer.config` constructors. | Storage/workstore forwarding modules, reference package facade, test-only evaluation production facade, unused resume planner, and unused YOLO26 adapter. |

## Required cutover procedure

1. Stop producers that write any removed schema or checkpoint shape.
2. Retain existing artifact roots unchanged for audit and rollback.
3. Regenerate query definitions, reviewed labels, reference inventories, and
   checkpoints with the current producer. Do not cast or relabel old rows.
4. Deploy this release only with the regenerated artifacts.
5. If rollback is needed, deploy the prior capable Git revision and point it at
   the retained prior artifact roots. Do not use this release to read or alter
   those roots.

## Behaviour deliberately retained

Current route-aware scoring, reference checkpoint resumability for the current
schema, taxonomy/geographic policy, and durable history are retained. The
change is limited to compatibility-only paths and unreachable legacy code; it
does not weaken validation or silently replace current production behaviour.

## Verification

Focused coverage is in query-planning, detection routing and masks, vision
gating, reference downloader, storage/configuration, workstore, and evaluation
tests. The release acceptance command is:

```bash
uv run pytest -q
```

The source audit and unavailable GitHits searches are recorded in
`provenance/githits.jsonl` under `legacy-cleanup-*`; no external search result
is claimed where the service was unavailable.

On 2026-07-17, the complete collected suite passed: 1,953 tests. The execution
environment limits a single test process to roughly one minute, so acceptance
ran the same collection in disjoint alphabetical and reference/regional chunks;
the focused vision-plumbing benchmark was rerun after its current-prompt fixture
repair.
