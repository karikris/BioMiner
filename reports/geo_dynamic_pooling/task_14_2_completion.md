# Task 14.2 completion — ButterflyLens evidence handoff

Status: completed and pushed to `origin/main` through
`0bde282d9e6604f4a606f7b52a5e836d6ee45b2a`.

## Delivered

BioMiner now publishes one complete, create-only ButterflyLens evidence handoff
with ten canonical roles: project, run, Flickr source records, media-object
identities, model evidence, geographic-impact inputs, review-campaign inputs,
pre-assignment review work items, classification maturity, and release state.
The publisher stages and validates every layer, writes the manifest last, then
builds and verifies a deterministic content-addressed archive before atomically
publishing the directory.

The model layer preserves raw scores as raw scores. Probability availability,
statistical support, human verification, evidence maturity, and release
maturity remain independent fields. Media identities contain no image bytes or
location payloads.

The geographic layer counts only the available Flickr/BioCLIP candidate
evidence. ButterflyLens-owned ALA baselines, human support, expert review, and
release support remain unavailable. Records without coordinates retain an
explicit no-geography state; unavailable evidence is neither zero nor evidence
of biological absence.

The review layer exports a draft blind campaign and representative work items
before consumer-side assignment. It exports no reviewer identifier and grants
no assignment authority. Inclusion probabilities and inverse-probability
weights remain attached to the representative sample, which stays separate
from targeted review. The release projection is blocked and every release gate
is false.

## Pinned consumer verification

ButterflyLens advances from the historical `fcee1a7…` pin to exact committed
object `1cea643…`, classified as additive compatibility with stricter review
controls. Compatibility checks use committed Git objects, never sibling
working-tree source.

Four generated target documents validate against six exact pinned JSON Schema
resources, and the ten frozen artifact descriptors match fresh exports. The
database-boundary fixture checks exact committed pgTAP and migration text for
unavailable-versus-zero semantics, all-gates release eligibility,
repeated-independent assignments, immutable assignment policy, blind review,
append-only review events, and RLS/service-role boundaries. This is contract
and fixture validation, not execution against a live Supabase database.

The consumer's complete parity runner also passed from a clean Git archive:
24 schemas, 20 valid cases, 20 invalid cases, 20 version checks, and 15
vocabulary checks agreed across JSON Schema, Python, and TypeScript 7.0.2. The
compiler came from an isolated temporary offline-cache install; neither
repository worktree was modified.

## Verification and authority boundary

- Focused ButterflyLens handoff gate: 46 passed in 3.70 seconds.
- Full BioMiner regression: 3,169 passed in 118.21 seconds.
- Changed-file Ruff, formatting, JSON parsing, and `git diff --check`: passed.
- Remote `origin/main` resolved to `0bde282…` after the implementation push.
- GitHits calls: zero, per the user's directive.

This handoff does not perform a live model run, production import, database
write, reviewer assignment, calibration, human review, or occurrence release.
It creates portable evidence inputs while leaving consumer database authority
and scientific release authority fail closed.
