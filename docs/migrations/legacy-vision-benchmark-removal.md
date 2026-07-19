# Legacy vision benchmark removal

Date: 2026-07-19

The following developer commands were removed during the adaptive production
cutover:

- `build-text-embedding-cache`;
- `benchmark-plumbing`;
- `benchmark-rolling-matrix`; and
- `benchmark-live-m5pro`.

They built or measured classification-v3 staged-rank embeddings, detector
crops, hierarchical cascade scoring, and the old rolling object-evidence join.
None produced an input consumed by the geography-conditioned dynamic-pool
workflow. Keeping them callable made the diagnostic crop/cascade architecture
look like a supported production alternative.

BioCLIP and YOLOE runtime checks and the explicit full-frame prototype smoke
remain available. Historical benchmark artifacts, reports, and Git revisions
remain immutable and may be used only as historical plumbing evidence—not
biological accuracy, current throughput, or production readiness.
