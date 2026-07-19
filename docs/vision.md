# Vision and classification

BioMiner uses YOLOE for coarse quality and life-stage routing and BioCLIP for
full-frame embedding and comparison. YOLOE does not decide species identity.
The durable detector contract is `object-detection-v2`; every row preserves
the normalized detector prompt, actual class ID, ordered prompt-set
fingerprint, route decision, routing-policy fingerprint, box geometry, and an
optional normalized mask polygon.

Production comparison routes are:

- `adult_butterfly_field` → `adult_field`;
- `caterpillar_field` → `larval`;
- `pinned_specimen` → `pinned_specimen`;
- ambiguous evidence → review when the active routing policy allows it; and
- pupa, moth/other insect, artwork, no-organism, and failure states → retained
  but unscored evidence.

The canonical BioCLIP input is the full decoded image. Adult, larval, and
specimen scoring units from one photo share one content-addressed raw image
embedding while retaining separate detector evidence. Focused, masked, and
multi-object variants preserve the canvas; they do not spatially crop or
manufacture detail. A small or unsuitable subject lowers evidence, abstains,
or enters review.

The detection schema retains nullable historical `crop_*` columns so existing
Parquet readers do not break. Current producers always leave them null with
`crop_storage_policy=not_created`. No crop generator, crop batching policy,
debug-crop writer, crop scorer, or crop runtime profile remains callable.

Adaptive scoring uses the complete target-preserving candidate union and
versioned global/local reference pools. Family and geography accelerate
retrieval; neither is an identity gate. Raw similarities, margins, detector
scores, component scores, and provisional fusion scores are not probabilities.
Calibration, human verification, representative statistical support, and
release authority remain separate contracts.

The classification-v3 staged-rank cache, hierarchical cascade, 0.90 genus
shortcut, bucket rules, object-evidence join, cloud rolling worker, and their
production dispatch have been removed. Frozen historical evaluation fixtures
may still describe old row shapes, but no adaptive run can select or fall back
to that implementation. See
[cascade and crop runtime removal](migrations/cascade-crop-runtime-removal.md).

Developer-only model installation checks remain under `biominer dev vision`:

```bash
uv run biominer dev vision bioclip-runtime-check --device mps
uv run biominer dev vision yoloe26-runtime-check --device mps
```

These checks prove only that a pinned runtime can load. Accuracy and release
claims require reviewed evidence and the corresponding scientific gates.
