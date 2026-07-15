# Operation-efficient storage handoffs

Use the handoff workflow for transferring a completed local artifact tree to
another computer through S3-compatible storage. Do not upload ignored run or
checkpoint directories one file at a time.

The handoff model is deliberately narrow:

1. Build one deterministic `tar.gz` archive locally.
2. Embed a sorted inventory containing every relative path, byte count, and
   SHA-256.
3. Verify the complete archive and every inventory entry locally.
4. Name the archive with its complete SHA-256.
5. Open one output stream directly on that immutable final key.
6. On the receiving computer, open one sequential input stream, cache the
   archive, verify it locally, and extract each file atomically.
7. Reuse the verified local cache on subsequent receive attempts, with no
   additional remote read.

This is separate from normal `CloudStorage.write_file` behavior. Ordinary
production objects retain staging, promotion, existence checks, and remote
readback. Those safeguards are appropriate for mutable independent objects but
are wasteful for a bulk computer-to-computer handoff.

## Producer workflow

Build and verify the bundle without making any remote request:

```bash
uv run biominer storage handoff-build \
  --root . \
  --source runs/example_run \
  --source reports/example_report \
  --source config/example.json \
  --output-dir /tmp/biominer-handoffs \
  --name example-handoff \
  --source-git-sha "$(git rev-parse HEAD)"
```

The command prints the archive path and a digest formatted as
`sha256:<64 hex characters>`. Preserve both values. The output directory must
be outside every source directory.

After the local build succeeds, upload that exact archive once:

```bash
uv run biominer storage handoff-upload \
  --archive /tmp/biominer-handoffs/example-handoff.sha256-<digest>.tar.gz \
  --sha256 sha256:<digest> \
  --destination-prefix s3://<bucket>/<prefix>/handoffs \
  --receipt /tmp/biominer-handoffs/example-upload-receipt.json \
  --config config/production.toml
```

The upload path performs no explicit `HEAD`, bucket/object listing, server-side
copy, completion-marker write, or remote readback. It opens one object output
stream. The S3 library may implement that stream as `PutObject` or as multipart
create/upload-part/complete calls depending on object size; those are transport
details, not additional handoff objects.

Configure the exact S3 region (for example, the region segment in a Backblaze
B2 endpoint). The handoff path rejects an empty region and `region = "auto"` so
the S3 client cannot perform automatic bucket-region discovery.

The local upload receipt records `remote_write_acknowledged`, not remote
integrity. It is written only after the output stream closes successfully. A
successful producer-side return proves that the service acknowledged the write;
end-to-end integrity is established only when the receiver verifies the archive
SHA-256 and embedded file inventory.

Do not retry an ambiguous failed upload blindly. The service may have committed
the content-addressed object before the client lost its response, and Backblaze
B2 keeps object versions. Let the receiver attempt the known URI once after the
operation budget permits; re-upload only after that explicit decision.

## Receiver workflow

Copy the URI and digest from the producer output or local receipt. The receiver
does not discover handoffs by listing a prefix:

```bash
uv run biominer storage handoff-receive \
  --uri s3://<bucket>/<prefix>/handoffs/example-handoff.sha256-<digest>.tar.gz \
  --sha256 sha256:<digest> \
  --cache-dir /tmp/biominer-handoff-cache \
  --destination . \
  --receipt /tmp/biominer-handoff-cache/example-receive-receipt.json \
  --config config/production.toml
```

The first successful run uses one sequential object read stream, verifies the
cached archive, verifies every embedded file, and then extracts. Existing files
are accepted only when their byte count and SHA-256 match the inventory. A
subsequent run with the same verified cache uses zero remote reads and only
verifies/extracts locally.

## Invariants and failure behavior

- A handoff key must contain the complete archive SHA-256 in its final filename.
- Sources outside the declared root, symlinks, non-regular files, duplicate
  paths, and unsafe archive paths are rejected.
- Archive metadata and ordering are normalized, so repeated builds from
  unchanged inputs, source Git SHA, and the same supported runtime are
  deterministic.
- Downloads are written to a temporary file and promoted only after local
  SHA-256 verification. Extraction uses temporary files and atomic hard links.
- A mismatching cache or destination file stops the receive; it is never
  overwritten silently.
- Receipts stay local. Publishing a receipt or a separate `COMPLETE` object
  would add remote operations and create split-brain completion state.
- Cleanup of abandoned legacy prefixes or old object versions is a separate,
  explicitly authorized maintenance operation. Upload and receive never list
  or delete remote objects automatically.

Backblaze currently classifies S3 `PutObject` and multipart upload operations as
Class A, `GetObject` and `HeadObject` as Class B, and listing/copy operations as
Class C. See the official [transaction pricing and operation classification](https://www.backblaze.com/cloud-storage/transaction-pricing)
and [S3-compatible operation list](https://www.backblaze.com/docs/cloud-storage-api-operations).
