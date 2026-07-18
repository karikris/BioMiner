# ButterflyLens `1cea643` compatibility review

## Decision

BioMiner Task 1.2 advances its audited ButterflyLens consumer pin from
`fcee1a76886e37cb2f0d9badbe91b70a18a0e7c3` to
`1cea643623f2f20a2bea72afc754c7b194db3278`.

The ten-commit delta is compatible and additive at the wire-contract boundary,
but it materially tightens the database adapter. The pin is therefore recorded
in every ButterflyLens handoff and compatibility fixture; it is not a silent
movement.

## Evidence reviewed

The review used committed objects only. The delta adds reviewer accounts,
verification landing surfaces, repeated independent assignments, blind review
disclosure, and append-only review submission. Existing project/run,
fingerprint, classification-maturity, geographic-impact, verification campaign,
assignment, event, and consensus wire versions remain unchanged.

The downstream database now additionally requires:

- at least two independent reviewers under `repeated-independent-v1`, with
  stronger qualified/expert requirements for higher-impact campaigns;
- model labels, scores, comments, queries, and peer decisions to remain hidden
  until the reviewer submits a decision;
- authenticated, append-only review-event submission with explicit supersession
  lineage for corrections; and
- database reviewer identities, primary keys, grants, service-role operations,
  and row-level security to remain ButterflyLens-owned.

## Anti-corruption consequence

BioMiner may emit immutable project/run projections, Flickr and media identity,
model evidence, geographic evidence, campaign and assignment *inputs*, maturity,
and a fail-closed release state. It must not create reviewers, allocate database
primary keys, perform service-role writes, claim an RLS bypass, or convert raw
scores or completed reviews into release authority.

This matches current Supabase guidance: table grants determine object reach and
RLS determines row reach, exposed tables require RLS, and newly created tables
may require explicit API exposure configuration. Those controls belong to the
application that owns the database, not to an artifact producer. See the
[Row Level Security guide](https://supabase.com/docs/guides/database/postgres/row-level-security),
[Data API security guide](https://supabase.com/docs/guides/api/securing-your-api),
and [Supabase changelog](https://supabase.com/changelog).

## Compatibility classification

- Wire-schema breaking change: **no**.
- BioMiner evidence-shape breaking change: **no**.
- Downstream adapter work required: **yes**.
- Import of ButterflyLens TypeScript, Python, or SQL into BioMiner: **no**.
- Release-authority change for BioMiner: **no**.

The resulting classification is
`compatible_additive_with_stricter_review_controls`.
