# ButterflyLens `1ca6d9e` compatibility review

## Decision

BioMiner advances its audited ButterflyLens consumer pin from
`3d6486da87f32136c35e29aeed6cb6291da66a17` to
`1ca6d9e15b03147df26a15deb309d32aed7ea9f7`.

The 16-commit delta is compatible with BioMiner's producer wire format. Its
contract, Flickr-policy, database-migration, and database-fixture trees are
byte-for-byte unchanged. No producer artifact or downstream database-adapter
change is required. The compatibility classification is
`compatible_contract_scope_unchanged_after_runtime_removal`.

The pin movement remains explicit because the consumer commit participates in
the handoff's content identity. BioMiner therefore regenerates the deterministic
manifest fingerprint and handoff ID even though the payload schemas are
unchanged.

## Exact committed-object evidence

The review compared committed Git objects, not working-tree files. At both
commits, the audited subtrees have these identical Git tree IDs:

| Audited surface | Git tree ID |
|---|---|
| `packages/contracts` | `323bc41a205bf3d81add72a4013d8536dcf750bc` |
| `packages/flickr` | `79b4d3f9ceeee5f89e31c4ddb500c5942b2067dc` |
| `supabase/migrations` | `f436e66f6d2f94b9ee0ff7937f6055d66d91e08a` |
| `supabase/tests/database` | `2acf49d5ce4276b42dc3e72da2e6b4cbfe78ccf8` |

The overall delta spans 80 files with 3,402 insertions and 3,969 deletions.
It removes the separate OpenAI analyst route, its browser and Edge Function
runtime, and its live-evaluation runner; it also adds UI redesign evidence and
agent-governance documentation. None of those files is part of BioMiner's
handoff contract or database-authority boundary.

## Preserved contract and authority boundary

The prior `3d6486d` review remains the semantic baseline for fingerprint v1.1,
adjudication, reviewer reliability, quality, rights, sensitive-location,
occurrence-release, Darwin Core, and ALA controls. In particular:

- the retired v1.0 fingerprint remains rejected and BioMiner continues to emit
  `butterflylens-evidence-fingerprint:v1.1.0`;
- `repeated-independent-v1`, blind, append-only review remains
  ButterflyLens-owned;
- service-role writes and row-level security remain downstream concerns;
- BioMiner exports no reviewer identity, database primary key, raw public
  coordinate, release decision, publication authority, or ALA submission
  authority; and
- missing or unfinished evidence remains unavailable rather than false or zero.

Removing ButterflyLens's runtime analyst route does not move any of these
responsibilities into BioMiner. The producer continues to publish immutable
candidate and model evidence through the same ten artifact roles and the same
contract versions.

## Compatibility classification

- Consumer wire breaking change: **no**.
- BioMiner producer wire compatible: **yes**.
- BioMiner artifact-shape change required: **no**.
- Downstream adapter work required: **no**.
- Deterministic manifest identity update required: **yes**, because the exact
  consumer commit is identity-bearing.
- Import of ButterflyLens implementation code into BioMiner: **no**.
- Database, review, location, release, export, or submission authority added to
  BioMiner: **no**.

Future pin movements still require exact-object review, fixture regeneration,
and an explicit compatibility decision; unchanged schema names alone are not
sufficient evidence of compatibility.

The unchanged downstream security boundary continues to follow Supabase's
[Row Level Security guidance](https://supabase.com/docs/guides/database/postgres/row-level-security).
