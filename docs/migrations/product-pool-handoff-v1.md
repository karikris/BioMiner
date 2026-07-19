# Product pool handoff v1 migration

> Historical scope: this document records the Task 1.2 manifest migration.
> Task 14.1 later advances the TaxaLens pin and adds the compatible deterministic
> archive transport described in the architecture handoff audit.
> The current ButterflyLens pin and its v1.1-only fingerprint boundary are
> recorded in
> [`butterflylens_1ca6d9_compatibility_review.md`](../architecture/butterflylens_1ca6d9_compatibility_review.md).

## Scope

Task 1.2 adds two immutable BioMiner producer manifests:

- `taxalens_dynamic_pool_handoff.json` for analytical review, sampling,
  quality, and geographic-impact projections; and
- `butterflylens_dynamic_pool_handoff.json` for project/run, Flickr/media,
  model, geographic, review-input, maturity, and release-state projections.

No sibling source tree or database is changed by this migration.

## Pins and compatibility

TaxaLens remains pinned to
`c5e87ead4fdb26d5c5624bbb8d8d67e46d8eddbc`.

ButterflyLens advances from
`fcee1a76886e37cb2f0d9badbe91b70a18a0e7c3` to
`1cea643623f2f20a2bea72afc754c7b194db3278`. The wire contracts are
compatible; its database adapter is stricter because it now owns repeated
independent assignment, blind disclosure, reviewer identity, and append-only
review submission. The detailed decision is in
[`butterflylens_1cea643_compatibility_review.md`](../architecture/butterflylens_1cea643_compatibility_review.md).

## Consumer migration

Consumers must:

1. Verify the manifest fingerprint, producer commit, consumer commit, required
   role list, and each available artifact's semantic fingerprint, physical
   SHA-256, byte count, and row count.
2. Treat non-available artifacts as typed states with reasons. Do not replace
   them with false, zero, an empty quality estimate, or release-ready state.
3. Apply only the field projections named by the target manifest. Application
   IDs and database primary keys are assigned by the consumer adapter.
4. For ButterflyLens, perform writes under its service-role path and enforce its
   grants, RLS, blind review, repeated assignment, and append-only event rules.
   BioMiner does not carry credentials or reviewer identities.
5. Revalidate a consumer pin before changing it. A regenerated fixture identity
   is a contract change and must not be accepted silently.

## Rollback and failure behavior

The manifests are additive artifacts; rollback means ceasing publication or
consumption of the new filenames. Existing BioMiner artifacts are not rewritten.
Pin mismatch, missing required role, checksum mismatch, unknown maturity,
noncanonical path, or authority escalation must block ingestion. A failed import
must not partially create downstream product records.
