# Migrate to the adaptive GBIF reference default

BioMiner now defaults to `adaptive_gbif_fast_start`. Earlier production
workflows treated `human_verified_strict` as the effective default and blocked
reference embedding and scoring until every support image completed human
identity review.

## Choose the mode explicitly

New runs should record the default mode and its policy fingerprint in every
support, readiness and downstream artifact. To retain the old behavior, select
`human_verified_strict`; provider assertions alone are ineligible and the
existing human reference-review workflow remains mandatory. Do not edit an
adaptive artifact to say it is strict, or vice versa.

The adaptive default admits only GBIF provider-asserted records that pass taxon,
licence, decode, canonical-duplicate, independence, YOLOE route and subject-area
gates. That support is provisional, not human verified. It may authorize
reference embedding and provisional scoring, never calibration labels,
final-test labels or scientific release.

## Artifact and schema boundary

Current contracts are:

| Artifact | Schema |
|---|---|
| Admission policy | `reference-admission-policy-v1.0.0` |
| Support manifest | `reference-support-manifest-v3.0.0` |
| Bank summary | `reference-bank-summary-v2.0.0` |
| Readiness permit | `reference-bank-readiness-v3.0.0` |
| Bank revision | `adaptive-reference-bank-revision-v1.0.0` |
| Feature reuse plan | `incremental-feature-reuse-plan-v1.0.0` |
| Flickr rescore plan | `flickr-rescore-plan-v1.0.0` |

V3 support rows add admission mode, admission policy identity, evidence basis,
provider assertions, human-review state, provisional status and statistical-
audit requirements. The explicit v2-to-v3 support-manifest migration preserves
old rows as `human_verified_strict`; it never invents provider evidence and does
not make an old readiness permit reusable.

## Invalidation and rerun

A changed mode, policy version or policy fingerprint invalidates readiness and
every derived identity that binds it. Revalidate source and support manifests,
publish a new immutable readiness directory, and rebuild the affected embedding,
prototype/model, calibration and score artifacts. Use the revision impact,
feature reuse and Flickr rescore plans to retain exact content-addressed
embeddings and unrelated scores. Do not globally rerun YOLOE or BioCLIP when a
validated plan says the inputs and producer fingerprints are unchanged.

Strict artifacts remain readable only when their declared schema is supported
and their fingerprints validate. Never reuse a strict readiness checksum for an
adaptive bank. Never relabel a provisional raw score as a probability.

## Rollback

1. Stop new adaptive publications; preserve their immutable directories.
2. Set the next run to `human_verified_strict`.
3. Select the last independently validated strict support manifest and its exact
   policy fingerprint, or migrate a supported v2 manifest through the explicit
   v2-to-v3 migration.
4. Publish a new strict v3 readiness directory and pin its SHA-256.
5. Rebuild or reuse downstream artifacts only through validated dependency and
   content-identity plans.
6. Run strict readiness, final Flickr export and full regression tests before
   resuming release.

Rollback changes future publications; it does not delete or rewrite adaptive
history. Flickr final release continues to require source-bound human review in
both modes.
