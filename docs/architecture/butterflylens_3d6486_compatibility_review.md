# ButterflyLens `3d6486d` compatibility review

## Decision

BioMiner advances its audited ButterflyLens consumer pin from
`1cea643623f2f20a2bea72afc754c7b194db3278` to
`3d6486da87f32136c35e29aeed6cb6291da66a17`.

The 62-commit delta is compatible with BioMiner's current producer wire format.
It is not wholly additive: ButterflyLens retired its evidence-fingerprint v1.0
schema and reader. BioMiner already targets
`butterflylens-evidence-fingerprint:v1.1.0`, and the pinned consumer contains a
negative parity case for retired v1.0 input, so no BioMiner artifact conversion
is required. The pin movement remains explicit in the producer manifest and
both compatibility fixtures.

The compatibility classification is
`compatible_with_v1_1_fingerprint_and_consumer_owned_release_controls`.

## Evidence reviewed

The audit read exact objects at both commits rather than the dirty sibling
worktree. Across `packages/contracts` and `supabase/migrations`, the delta
changes 32 files with 9,472 insertions and 291 deletions. The handoff-relevant
surface adds:

- independent conflict adjudication, control-item-calibrated reviewer
  reliability, layered consensus, and representative quality estimates;
- fail-closed Flickr public-display and media-takedown controls;
- sensitive-location policy and coordinate-free public-location decisions;
- a nine-gate occurrence-release decision;
- deterministic Darwin Core evidence-package preparation; and
- deterministic ALA contribution preparation that still requires human
  submission.

The existing project, run, classification-maturity, geographic-impact,
verification campaign, assignment, event, and consensus schema versions do not
move. Fingerprint v1.1 remains the only accepted fingerprint version.
The earlier `repeated-independent-v1`, blind-review, and append-only correction
controls also remain mandatory and are strengthened by the new layers.

## Producer/consumer boundary

BioMiner may continue to emit immutable project/run projections, source and
media identities, provisional raw model scores, candidate-only geographic
impact, pre-assignment review inputs, maturity evidence, and a blocked release
state. It does not gain authority to:

- display Flickr media, approve rights, or complete a takedown;
- expose a source coordinate or issue a sensitive-location receipt;
- appoint an adjudicator, estimate reviewer reliability, or issue a product
  quality snapshot;
- decide that a candidate is release-ready or publish an occurrence;
- prepare or release a Darwin Core archive; or
- submit data to ALA.

These decisions depend on ButterflyLens-owned database identities, append-only
events, service-role operations, grants, and row-level security. BioMiner
therefore carries no database primary key, reviewer identity, service-role
credential, public-location receipt, release receipt, or publication approval.

## Supabase safety review

No Supabase schema or live database is changed by this BioMiner pin update.
The exact pinned migrations retain RLS on exposed tables, identity-bound review
writes, `security_invoker` views, private reviewer-control material, and
service-owned append paths. The current Supabase changelog was checked on
2026-07-19. Changes to automatic Data API exposure reinforce the existing
boundary: table exposure, grants, and RLS belong to ButterflyLens, not the
artifact producer. See the
[Row Level Security guide](https://supabase.com/docs/guides/database/postgres/row-level-security),
[Data API security guide](https://supabase.com/docs/guides/api/securing-your-api),
and [Supabase changelog](https://supabase.com/changelog).

## Compatibility classification

- Consumer wire breaking change: **yes**, for retired fingerprint v1.0 only.
- BioMiner producer wire compatible: **yes**, because it already emits v1.1.
- BioMiner artifact-shape change required: **no**.
- Downstream adapter work required: **yes**, for the new product-owned gates.
- Import of ButterflyLens TypeScript, Python, or SQL into BioMiner: **no**.
- Database, review, location, release, export, or submission authority added to
  BioMiner: **no**.

Missing, unreviewed, withheld, or unavailable evidence continues to fail closed;
it is never converted to false, zero, absence, a probability, or an occurrence.
