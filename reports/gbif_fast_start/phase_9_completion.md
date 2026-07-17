# Phase 9 completion — selective rebuild and rerun

Phase 9 is complete. All five task commits were pushed and the final full
repository run passed 2,475 tests in 94.12 seconds.

| Task | Commit | Result |
|---|---|---|
| 9.1 | `622e97f80c181e098d78970bf56f534f36ec2ef2` | Validated dependency propagation identifies affected work and persists unaffected artifacts as reusable |
| 9.2 | `b9a798a3e8e38eacf5e11ceaccccd242a7e6c780` | Exact content-addressed identities reuse unchanged Flickr and reference embeddings and compute only cache misses |
| 9.3 | `0208c0fbc2e9e92374ac0cadc49643113d5d0c8b` | Only affected prototypes, indexes and classifier work rebuild; calibrators require changed training evidence |
| 9.4 | `efd8d1ce3b5d3a81932e16fbdee246caa715dbf9` | Five impact triggers plus missing safety evidence select scores; unrelated records retain prior scores |
| 9.5 | `0fd1bce06788e1adfef7b1d11bdcae2e7284ca7e` | Paired audited outcomes, metric deltas, compute/review evidence and identity-component uncertainty are reported separately |

The downstream impact graph starts from the adaptive reference-bank change
ledger and propagates reference IDs, species, routes and regions through only
declared dependencies. The persisted plan names both affected artifacts and
unaffected artifacts that remain reusable. Cycles, stale edge fingerprints and
undeclared impact evidence fail closed.

Feature reuse is content addressed. Flickr detection and embedding reuse
requires the same visual content, model and preprocessing identity. Unchanged
reference images retain their vectors, excluded references are filtered from
the new support frame, and only newly admitted or otherwise missing identities
enter the encoder batch. Conflicting or tampered cache entries are rejected.

Model work is scoped from the impact manifest. Affected species prototypes,
regional candidate indexes and classifier rows are selected independently.
Calibrators are not refreshed merely because an upstream artifact was touched:
the old and new training-data fingerprints must differ.

Flickr records are rescored when the changed bank affects their target, best
competitor, candidate union or referenced support, or when the previous margin
falls inside the configured safety band. Missing dependency or margin evidence
also fails closed to rescoring. Unrelated records remain explicitly reusable.

The comparison report keeps production-plan selection counts separate from the
human-reviewed paired subset. Reused pairs must retain the exact evaluation
row. Corrected errors and new errors use the frozen human target-presence label,
metric deltas are direction-neutral after-minus-before values, and paired
accuracy change uses whole identity-component bootstrap intervals. Unlike
compute units are never summed, and missing review time remains unavailable.

These are fixture-tested workflow, invalidation and reporting contracts. No
live production remediation, accuracy gain, corrected-error total, reviewer
throughput, cost reduction or time-saving benefit is claimed.
