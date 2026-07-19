# Cloud And Local Run Boundary

Status: accepted

BioMiner supports local filesystem plus SQLite for development and deterministic tests. Production-scale runs use object storage and a workstore backend, with immutable stage artifacts and resumable work claims.

The orchestration contract is stage-oriented: registry, Flickr metadata,
metadata flags, object detection, BioCLIP scoring, evidence join, and explicit
review queues. Cloud support must be explicit per stage. A cloud path that is
not implemented must fail clearly and must not be documented as complete.

The former optional Flickr-comment review and occurrence-promotion stages were
removed on 2026-07-19. They are not a cloud/local compatibility requirement;
see `../migrations/flickr-comment-promotion-removal.md`.

Generated run reports are operational evidence. They should be written by each run, retained in the configured artifact store, and excluded from source control.
