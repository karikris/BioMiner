# Registry and taxonomy

Step 0 is mandatory. It produces an accepted GBIF taxonomic/name registry and a separate reviewed classification overlay.

## GBIF spine

The configured root is Papilionoidea, GBIF key 1875. Production scope pins Hesperiidae, Papilionidae, Pieridae, Lycaenidae, Riodinidae, Nymphalidae, and Hedylidae. Accepted GBIF species keys are the durable identity. Synonyms, vernaculars, translations, and external identifiers are evidence attached to that identity.

The base build writes `taxa.parquet`, `taxon_relations.parquet`, `names.parquet`, `name_evidence.parquet`, `source_snapshots.parquet`, `flickr_query_definitions.parquet`, `qa_findings.parquet`, and `manifest.json`. Checkpoints are family-scoped, atomic, versioned, and rejected when schema, scope, source, or registry version differs.

## Reviewed classification overlay

BioCLIP classification uses the ordered rank contract:

```text
FAMILY → SUBFAMILY → TRIBE → SUBTRIBE → GENUS → SPECIES
```

GBIF does not reliably expose subfamily, tribe, and subtribe in the butterfly backbone. Those ranks therefore come from reviewed source records, never suffix inference or name guessing. SUBTRIBE is optional only through an explicit, sourced, reviewed `TRIBE → GENUS` rank-skip edge; it is never invented. The production source catalog is `config/taxonomy/papilionoidea_classification_v3.json`.

An enabled path requires:

- allowed adjacent rank transitions, or the single reviewed optional-SUBTRIBE skip;
- one enabled parent per child;
- no cycles;
- complete authority, release, citation, retrieval date, and evidence;
- explicit reviewer identity and review date;
- one accepted GBIF mapping per species node;
- every mandatory rank (`FAMILY`, `SUBFAMILY`, `TRIBE`, `GENUS`, `SPECIES`) and either a sourced SUBTRIBE or a reviewed skip.

Fatal findings block artifact promotion. Accepted species without a reviewed path remain in the GBIF registry but receive `unmapped_accepted_species` warnings and are unavailable to the hierarchical classifier.

The classification manifest carries two deterministic identities:

- `classification_fingerprint` covers the versioned sources, nodes, edges, GBIF mappings, leaf paths, and staged prompt labels;
- `hierarchy_fingerprint` covers the rank order, nodes, edges, GBIF mappings, and leaf paths independently of prompt wording.

The separately built text-embedding artifact carries
`embedding_cache_fingerprint`. Its exact classification version, prompt
version, hierarchy fingerprint, prompt-stage label set, model ID, model
checkpoint, dimensions, normalized vectors, and self-fingerprint are validated
before production scoring. A cache from another taxonomy, prompt set, model, or
checkpoint is not accepted.

## Names and Flickr queries

Source authority order is GBIF/CoL taxonomy, established vernacular sources, then generated translation candidates. Raw translations are T5 candidates and do not become enabled queries without review.

Flickr definitions contain one normalized term and one field (`tags` or `text`). Scientific species terms run before common names, genus/family terms, broad butterfly terms, and experimental translations. Discovery evidence is retained even when repeated photo processing is deduplicated.

## Commands

```bash
uv run biominer registry build --output-dir data/registry/butterflies-v2 --registry-version butterflies-v2
uv run biominer registry build-classification --registry-dir data/registry/butterflies-v2
uv run biominer registry audit --registry-dir data/registry/butterflies-v2 --report-dir reports
```
