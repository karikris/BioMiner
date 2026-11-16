# Science and pipeline rules

## Current scope

The active production pipeline begins with the GBIF three-file DWCA and no
derived evidence. Follow [GBIF ground-zero pipeline](../PIPELINE_GROUND_ZERO.md)
in order: fingerprint raw members; enrich source-bound taxonomy and names;
discover through Flickr; propose media domains with EmbeddingGemma 2; rank taxonomic evidence with
BioCLIP; then enter separate review and release gates.

## Non-negotiable evidence rules

- A GBIF row, provider name, Flickr result, query term, detector route, and
  BioCLIP candidate are evidence, not verified biological truth.
- Preserve source identity, checksum, logical-query association, physical API
  request identity, and duplicate provenance; deduplicate work, never discovery
  provenance.
- Common names from iNaturalist, Wikimedia, Catalogue of Life, and other
  providers must retain source, language/region where known, retrieval date,
  and trust/query eligibility. Never invent or machine-translate names.
- EmbeddingGemma 2 proposes butterfly, moth/other-insect, life-stage, specimen,
  artifact, and ambiguous domains from full-frame embeddings. Every proposal
  requires review; signed similarities and margins are not probabilities.
  Object counts, localization, subject presence and subject size remain
  unavailable. Historical YOLOE evidence is never relabelled as Gemma evidence.
- BioCLIP preserves order, superfamily, family, genus, and species evidence;
  raw scores and margins are not probabilities and must not hard-prune targets.
- Human review, quality estimation, rights, release readiness, and publication
  are explicit later gates. Missing evidence fails closed.
