# Vision evidence scope

No production image bank or model output currently exists. Visual work begins
only after the GBIF-grounded Flickr discovery and media-ingestion stages have
produced immutable, rights-aware candidate artifacts.

YOLOE is to be optimized and evaluated as a visual-domain router for the photo
bank. Its routes distinguish butterflies, moths, other insects, appropriate
life stages, specimens, artifacts, no-organism images, and ambiguous material.
YOLOE does not establish taxonomy, and a detector score is not a probability.

Eligible images are evaluated as full frames by BioCLIP. The evidence path
retains candidates at order, superfamily, family, genus, and species, plus
source-bound common-name associations. Higher-rank evidence must not silently
eliminate a species candidate. Scores, margins, and ranking remain model
evidence, not verified identities or occurrence-release decisions.

Human review, calibration, quality estimation, rights, and occurrence release
are independent downstream gates. The complete production ordering is defined
in [GBIF ground-zero pipeline](PIPELINE_GROUND_ZERO.md).
