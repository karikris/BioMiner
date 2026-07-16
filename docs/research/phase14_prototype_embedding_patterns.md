# Phase 14 prototype embedding implementation notes

Date: 2026-07-16

Task 14.4.2 required a prototype-only embedding path because the frozen pilot
bank deliberately retains `provider_supported` evidence and zero attributable
human taxonomic reviews. The scientific `reference_support_manifest` builder
correctly rejects those rows, so the prototype command consumes the distinct
`prototype-support-manifest-v1.0.0` contract and never rewrites its verification
status to `verified`.

## Open-source patterns checked with GitHits

GitHits solution `09b5adde-5bed-408b-ba4a-93b3d4e64207` was used for the
implementation review. The useful patterns were deterministic record ordering,
per-record failure quarantine, durable success checkpoints, L2-normalized
vectors, and lifecycle-specific centroid grouping. BioMiner reimplements these
patterns with Polars, its persistent BioCLIP sidecar, frozen prototype
provenance, and atomic local Parquet replacement; no external code was copied
wholesale.

The GitHits result cited these permissively licensed sources:

- [elara-labs/code-context-engine pipeline](https://github.com/elara-labs/code-context-engine/blob/da47caca487096a18b94a4fbb3ff0ba540b9f4e0/src/context_engine/indexer/pipeline.py) (MIT)
- [Eventual-Inc/Daft dataframe implementation](https://github.com/Eventual-Inc/Daft/blob/0fdec65f445c73eb8ef35068aa71cd6477174a53/daft/dataframe/dataframe.py) (Apache-2.0)
- [shibing624/similarities FAISS/BERT implementation](https://github.com/shibing624/similarities/blob/80d5ef405edb49a0d34264489434267a76027a58/similarities/faiss_bert_similarity.py) (Apache-2.0)

## BioMiner decisions

- Successful records continue when a batch contains one unreadable or failed
  record. The batch is retried one record at a time and failures remain
  retryable operational evidence.
- Operators may explicitly skip a record only with a persisted reason. Skipped
  records enter the same retryable ledger and are not biological negatives.
- A completed embedding Parquet acts as a resumable checkpoint. Its bank,
  support-manifest, model, revision, and OpenCLIP identities must match before
  reuse.
- Prototype fitting consumes `support_train` only and first collapses media by
  independent observation. Adult, larval, and specimen routes are never mixed.
- The visual-neighbour graph compares global species prototypes only within the
  same route. Similarity remains raw screening evidence, not probability.
- Generated Parquet, reports, source images, and model caches remain ignored
  runtime artifacts. The tracked configuration uses local storage; S3 is
  deferred.

## Local validation boundary

The command was validated with five real support images on MPS and a 1,024-wide
BioCLIP embedding. The remaining frozen records were explicit temporary skips
for that validation run. The full 81-record execution remains subject to the
configured different-computer requirement and must not be inferred from the
five-image validation.
