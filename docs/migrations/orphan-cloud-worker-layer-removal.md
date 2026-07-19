# Orphan cloud and model-worker layer removal

Date: 2026-07-19

The post-uplift audit removed four test-only infrastructure layers:

- `flickr_fetch.cloud_poller`, which duplicated the current metadata poller,
  imported its private helpers, and had no CLI or orchestrator caller;
- `storage.compaction` and its compacted-path builder, whose public command had
  already been removed and whose only consumer was its own test module;
- `storage.shard_paths`, a second pair of URI builders duplicating
  `storage.paths.build_evidence_shard_uri`; and
- `vision.model_state`, a four-component cache/checkpoint wrapper never wired
  into a worker, stage handler, or command.

The current metadata poller already accepts `CloudStorage` and `WorkStore`,
registers immutable shards, and supports local or S3 output. The current
storage backends retain atomic Parquet writes, deterministic shard discovery,
and content-addressed handoffs. The generic workstore retains leases, claim
fencing, and stale-claim recovery. Full-frame reference and Flickr embedding
modules own their own content-addressed reuse and resumable checkpoints.

No compatibility fallback is provided. Historical compacted shards or model
worker experiments remain recoverable from their committed artifacts and Git
history; they are not reinterpreted as outputs of the adaptive workflow.
