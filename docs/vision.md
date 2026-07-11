# Vision and classification

YOLOE identifies butterfly and life-stage objects. Production preserves metadata for ineligible detections but does not create their crops. Eligible crops use profile-controlled padding, resize, batching, and memory limits; fine wing patterns use high-quality image resizing.

BioCLIP 2.5 runs as a persistent worker. Build text embeddings from reviewed
rank prompts with `biominer dev vision build-text-embedding-cache`, then pass
the Parquet artifact to production with `--taxonomy-text-embedding-cache`.
The cache is validated against model checkpoint, prompt version, and taxonomy
fingerprint. Image embeddings are batched and compared with normalized text
embeddings. Omitting the cache is an explicit slower diagnostic fallback.

## Five-stage cascade

The classifier scores family, reviewed child subfamilies, reviewed child tribes, reviewed child genera, then species belonging only to surviving genera. Configurable path beams allow a lower-ranked score to recover a family path that was not family top-1. The first-pass species top 20 is rescored in full; target injection is not used for open classification.

Each crop records:

- candidates, scores, and margins at every rank;
- candidate counts and beam widths;
- the selected five-rank path;
- every pruning decision and skipped-level reason;
- taxonomy version, source release, prompt version, taxonomy fingerprint, and embedding-cache fingerprint;
- all first-pass and reranked species candidates.

Classifier output remains screening evidence. Open classifications enter `in_review`; GBIF and reviewed registry evidence define identity.

## Runtime profiles and benchmarks

Production visual settings flow from the selected runtime profile into detection, crop materialization, and BioCLIP. The Mac profile is `config/vision_profiles/mac_m5pro_64gb.json`.

Developer-only checks and benchmarks live under `biominer dev vision`:

```bash
uv run biominer dev vision bioclip-runtime-check --device mps
uv run biominer dev vision yoloe26-runtime-check --device mps
uv run biominer dev vision benchmark-plumbing --records 1000 --output-dir reports/vision_benchmarks/plumbing
uv run biominer dev vision benchmark-rolling-matrix --records 1000 --output-dir reports/vision_benchmarks/rolling
```

Benchmarks measure plumbing and throughput, not biological accuracy. Accuracy evaluation uses reviewed labels through `biominer evaluation classify`.
