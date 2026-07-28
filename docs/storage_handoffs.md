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

## Direct GBIF handoff from WSL2 to macOS

Transfer only the terminal resolver-integrated publication, its sealed audit,
and the slim locator index. Do not transfer `.current.staging`, DuckDB temp
spill, old intermediate Parquets, mutable resolver work state, or the 38 GB
superseded set. Parquet is already compressed; rebuilding or recompressing it
only adds I/O.

### 1. Seal a zero-copy transfer directory in WSL2

Do this only after the terminal resolver, resolver-integrated publication,
publication audit, and locator index have all completed. These commands fail
before creating the handoff when any validator fails. `cp -al` creates
same-filesystem hard links, so the staged 25–35 GB handoff consumes almost no
additional data blocks. Run this strict staging sequence before the guarded
superseded-artifact cleanup removes checksum-bound build dependencies.

```bash
cd /home/toffe/github/karikris/BioMiner

BASE="$PWD/data/derived/gbif_media_final/base-v1"
PUBLICATION="$PWD/data/derived/gbif_media_final/current"
RESOLUTION="$PWD/data/state/gbif-media-url-resolution/full-v1/finalized-v1"
AUDIT="$PWD/data/derived/gbif_media_final/audit-v1"
LOCATOR="$PWD/data/derived/gbif_media_final/locator-v1"
HANDOFF="$PWD/data/transfer/gbif-media-final-20260729"

test ! -e "$HANDOFF"

.venv/bin/python scripts/validate_gbif_final_resolution_enrichment.py \
  --output-directory "$PUBLICATION" \
  --base-publication-directory "$BASE" \
  --resolution-directory "$RESOLUTION" \
  --repository-root "$PWD"

.venv/bin/python scripts/validate_gbif_final_publication.py \
  --audit-directory "$AUDIT" \
  --primary-publication-directory "$PUBLICATION" \
  --repository-root "$PWD"

.venv/bin/python scripts/validate_gbif_final_locator_index.py \
  --index-directory "$LOCATOR" \
  --publication-audit-directory "$AUDIT" \
  --publication-directory "$PUBLICATION" \
  --repository-root "$PWD"

mkdir -p "$HANDOFF"
cp -al "$PUBLICATION" "$HANDOFF/publication"
cp -al "$AUDIT" "$HANDOFF/audit"
cp -al "$LOCATOR" "$HANDOFF/locator"
git rev-parse HEAD > "$HANDOFF/SOURCE_GIT_COMMIT"

(
  cd "$HANDOFF"
  find . -type f ! -name SHA256SUMS -print0 |
    sort -z |
    xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)

find "$HANDOFF" -type d -exec chmod a-w {} +
find "$HANDOFF" -type f -links +1 -printf '%n %p\n'
du -sh --apparent-size "$HANDOFF"
```

The apparent size is the number of bytes transferred. `du` cannot report the
incremental allocation of hard links accurately because it attributes the
linked inode's blocks to whichever tree it scans. Link counts greater than one
prove that the large staged files share their original inodes. Only directory
permissions are changed: changing a hard-linked file's mode would also change
the original publication. Do not remove the original publication or the
handoff until the Mac verification succeeds.

Start the SSH server inside WSL2:

```bash
sudo service ssh start
sudo ss -ltnp | grep ':22'
whoami
```

With WSL2 mirrored networking, the Mac can normally use the Windows LAN IP and
port 22. With default NAT networking, run the following once in an
Administrator PowerShell window on Windows. Re-run it after a WSL restart
because the internal WSL address may change:

```powershell
$WslIp = (wsl.exe -d Ubuntu hostname -I).Trim().Split()[0]
$ListenPort = 2222

netsh interface portproxy delete v4tov4 `
  listenaddress=0.0.0.0 listenport=$ListenPort
netsh interface portproxy add v4tov4 `
  listenaddress=0.0.0.0 listenport=$ListenPort `
  connectaddress=$WslIp connectport=22

if (-not (Get-NetFirewallRule -DisplayName "WSL2 GBIF transfer" `
  -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -DisplayName "WSL2 GBIF transfer" `
    -Direction Inbound -Action Allow -Protocol TCP `
    -LocalPort $ListenPort -Profile Private
}
```

Use `ipconfig` or `Get-NetIPAddress -AddressFamily IPv4` on Windows to find
the Windows LAN address. Do not expose port 2222 through the internet router.

### 2. Start and verify the pull from macOS

Run these commands in Terminal on the Mac. The Mac initiates the transfer. Use
port 22 for mirrored networking or port 2222 for the NAT port-forward above.

```bash
WSL_USER="toffe"
WINDOWS_LAN_IP="<for-example-192.168.1.50>"
SSH_PORT="2222"
REMOTE_ROOT="/home/toffe/github/karikris/BioMiner/data/transfer/gbif-media-final-20260729"
DEST="$HOME/BioMiner-data/gbif-media-final-20260729"

ssh -p "$SSH_PORT" "${WSL_USER}@${WINDOWS_LAN_IP}" \
  "test -r '$REMOTE_ROOT/SHA256SUMS'"

mkdir -p "$DEST"
rsync -avhP --partial \
  -e "ssh -p $SSH_PORT" \
  "${WSL_USER}@${WINDOWS_LAN_IP}:${REMOTE_ROOT}/" \
  "$DEST/"

(
  cd "$DEST"
  shasum -a 256 -c SHA256SUMS
)
```

Rerun the same `rsync` command after an interruption. It retains partial files
and completes them without deleting the WSL copy. Do not add `--inplace`,
`--delete`, or `--remove-source-files`.

Use a clone of BioMiner on the Mac and check out the exact transferred
commit. Then independently validate the relocated artifacts. Allowing cleaned
dependencies does not waive final Parquet, primary-manifest, audit-artifact,
schema, row-group, identity, Git-commit, locator, or checksum checks.

```bash
REPO="$HOME/github/karikris/BioMiner"
DEST="$HOME/BioMiner-data/gbif-media-final-20260729"
SOURCE_COMMIT="$(tr -d '[:space:]' < "$DEST/SOURCE_GIT_COMMIT")"

git -C "$REPO" fetch origin main
git -C "$REPO" checkout "$SOURCE_COMMIT"

cd "$REPO"
uv run python scripts/validate_gbif_final_resolution_enrichment.py \
  --output-directory "$DEST/publication" \
  --repository-root "$REPO"

uv run python scripts/validate_gbif_final_publication.py \
  --audit-directory "$DEST/audit" \
  --primary-publication-directory "$DEST/publication" \
  --repository-root "$REPO" \
  --allow-cleaned-dependencies

uv run python scripts/validate_gbif_final_locator_index.py \
  --index-directory "$DEST/locator" \
  --publication-audit-directory "$DEST/audit" \
  --publication-directory "$DEST/publication" \
  --repository-root "$REPO" \
  --allow-cleaned-dependencies
```

After all checksum and semantic validators pass on the Mac, the WSL staging
hard links can be removed without touching the original publication:

```bash
find \
  /home/toffe/github/karikris/BioMiner/data/transfer/gbif-media-final-20260729 \
  -type d -exec chmod u+w {} +
rm -r \
  /home/toffe/github/karikris/BioMiner/data/transfer/gbif-media-final-20260729
```

If both machines are not online together, use the content-addressed object
handoff below. Its source set should still contain only the resolver-integrated
publication, audit, and locator; never package the live staging or spill tree.

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
