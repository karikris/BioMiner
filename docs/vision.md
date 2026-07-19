# Vision and classification

YOLOE identifies butterfly and life-stage objects and emits
`object-detection-v2` rows. Each detected object preserves its normalized raw
prompt, actual class ID, ordered prompt-set fingerprint, route decision, and
routing-policy fingerprint. Segmentation-capable detectors also retain one
nullable normalized mask polygon aligned with the detection; raw raster masks
are not stored. Production routes are:

- `adult_butterfly_field` to the `adult_field` comparison;
- `caterpillar_field` to the separate `larval` comparison;
- `pinned_specimen` to the separate `pinned_specimen` comparison;
- `pupa_or_chrysalis`, `possible_moth_or_other_insect`,
  `artwork_logo_tattoo_or_other_artifact`, and `no_relevant_organism` to
  retained, unscored evidence;
- `ambiguous_visual_domain` to low-priority review only when enabled and above
  its configured threshold.

The current production BioCLIP scorer declares only `adult_field` support.
Larval and specimen rows retain their own comparison route and enter review
until compatible scorers are configured; they never enter adult comparison.
No-organism and no-detection rows never enter BioCLIP, and load/inference
failures remain distinct from biological absence. The legacy
`butterfly_like_only` and `exclude_hard_negative` gates are diagnostic modes,
not production defaults.

Production preserves metadata for ineligible detections but does not create
their crops. Eligible crops use profile-controlled padding, resize, batching,
and memory limits; fine wing patterns use high-quality image resizing.

The opt-in target-aware visual mode is
`whole_image_reference_ensemble`. It disables detection crop computation,
requires no `crop_hash`, builds one scoring unit per photo and comparison
route, and content-addresses the decoded full image independently of detector
metadata. Adult, larval, and specimen units for the same photo reuse one raw
image embedding while retaining their own detection IDs, detector and
full-frame boxes, normalized mask polygons, and routing-policy identity.
Focused full-frame inputs are represented as requests here; their deterministic
transformations are a separate versioned stage. This mode does not write or
reinterpret legacy `object_bioclip_scores` artifacts and is not yet the
production default.

The classification-v3 text-cache builder was retired with the cascade
diagnostic surface. Adaptive scoring persists immutable full-frame image and
reference embeddings through the owning embedding contracts; it does not build
or consume the old staged-rank prompt cache.

## Butterfly family/genus/species funnel

The registry stores BioCLIP's supported identity hierarchy:

```text
KINGDOM → PHYLUM → CLASS → ORDER → FAMILY → GENUS → SPECIES
```

Visual routing first scores all seven butterfly families and keeps family top
1. Genera belonging to that family are ranked to top 20, then narrowed to top
3. Species beneath those genera are ranked to top 20, exactly those species
are reranked with distinct prompts to top 5, and top 1 is selected from that
top 5. If genus top-1 confidence is strictly above 0.90, only that genus is
used for the species universe; otherwise the top-20 then top-3 genus route is
used. Target injection is not used for open classification.

Each crop records:

- family candidate count, top 1 identity, score, and margin;
- genus top 20 and top 3 identities, node IDs, scores, confidence threshold, and routing mode;
- final winning-path `selected_<rank>`, `selected_<rank>_node_id`, and `selected_<rank>_score` fields;
- `species_top20` with first-pass scores, reranked `species_top5`, and reported `species_top3` with accepted GBIF keys;
- candidate, retained, and active-path counts, plus the versioned routing trace;
- `classification_version`, `prompt_version`, `classification_fingerprint`, `hierarchy_fingerprint`, and `embedding_cache_fingerprint`.

Every shortlist remains available for audit even when the >0.90 genus shortcut
is used.

Classifier output remains screening evidence. Open classifications enter `in_review`; GBIF and reviewed registry evidence define identity.

## Runtime checks

Production visual settings flow from the selected runtime profile into detection, crop materialization, and BioCLIP. The Mac profile is `config/vision_profiles/mac_m5pro_64gb.json`.

Developer-only runtime checks live under `biominer dev vision`:

```bash
uv run biominer dev vision bioclip-runtime-check --device mps
uv run biominer dev vision yoloe26-runtime-check --device mps
```

The retired plumbing, rolling, and M5Pro benchmark harnesses measured the old
crop/cascade runtime and were not adaptive scientific evidence. Historical
artifacts remain available in Git. Accuracy evaluation requires reviewed
labels; runtime smoke evidence is not biological accuracy.
