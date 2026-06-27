# BioCLIP Geo Funnel Plan

Date: 2026-06-28

## Sources Checked

- Morph MCP: local BioCLIP screening flow, `Papilio demoleus` hardcoding, species candidate loading, and bucket assignment.
- GitHits MCP: Polars deterministic Parquet/schema pattern, used as a reminder to keep report and embedding outputs schema-first and typed.
- Hugging Face model card for `imageomics/bioclip-2.5-vith14`: confirmed OpenCLIP loading, ViT-H/14 backbone, zero-shot use with taxonomic or vernacular names, and standard CLIP preprocessing.
- PyTorch `torch.mps` docs: confirmed `current_allocated_memory`, `driver_allocated_memory`, and `recommended_max_memory` are byte-returning MPS metrics.
- GBIF technical docs: occurrence API/download docs reviewed for future ingestion boundaries.
- Darwin Core terms: used to keep occurrence-field names compatible with `basisOfRecord` and `coordinateUncertaintyInMeters`.

## Decisions

- BioCLIP predictions remain screening evidence. The new candidate modes only control labels sent to the model; they do not validate taxonomic identity.
- `--target-species` is now explicit. Global candidate loads no longer pin `Papilio demoleus` unless a focused run asks for it.
- Candidate sets are represented as typed dataclasses with deterministic signatures derived from exact label-set tuples. This makes the existing sidecar text-feature cache effective because images with identical labels are batched together.
- The initial CLI default remains `hybrid` plus `all` candidates for compatibility with existing screen runs, while adding `triage`, `family`, `genus`, `species`, and `rescue_full_species` modes.
- Benchmarking is schema-first in this patch. The new command writes the requested configuration matrix and reports unavailable runtime metrics as `null` or `not_instrumented`; model-running throughput measurement can fill these fields later.
- Image embeddings are optional and stored in a separate Parquet file keyed by `image_hash`, model metadata, checkpoint, and preprocessing version. Raw Flickr image bytes are still deleted after classification.
- The geo-candidate implementation starts with deterministic global grid cells and local Parquet builders. GBIF network ingestion is intentionally left as the next patch so the schema and fallback behavior can be tested independently.

## Geo Schema Notes

The initial geo species index writes:

- `geo_version`
- `geocell_id`
- `grid_level`
- `species_key`
- `scientific_name`
- `family`
- `genus`
- `occurrence_count`
- `record_count_weighted`
- `first_year`
- `last_year`
- `basis_of_record_counts`
- `coordinate_uncertainty_summary`
- `source_dataset_count`
- `candidate_rank_prior`
- `provenance_json`

Fallback order is from the requested grid level toward coarser levels, ending at `G0_world`. Neighbour expansion is supported at lookup time but is not yet wired to Flickr coordinate uncertainty policy.

## Next Patch Sequence

1. Add GBIF occurrence/download ingestion with resumable state and dataset provenance.
2. Wire `geo_candidate_sets.parquet` into `CandidateStrategy.GEO` and `CandidateStrategy.HIERARCHICAL`.
3. Add real benchmark execution that runs bounded samples and fills throughput, distribution, RSS, and MPS metrics.
4. Add Pass 4/Pass 5 species reranking columns and evidence summaries.
5. Add DuckDB QA reports for geo coverage and candidate-set cardinality.
