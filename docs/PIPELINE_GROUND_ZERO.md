# GBIF ground-zero pipeline

## Active state

BioMiner starts at **ground zero** with one GBIF Darwin Core Archive (DWCA) and
its bounded physical Parquet representation. The source members are
`occurrence.txt`, `multimedia.txt`, and `verbatim.txt`; the corresponding
Zstandard Parquet files and `dwca_parquet_manifest.json` are retained as the
loaded intake. The manifest records 75,352,491 occurrence rows, 18,680,565
multimedia rows, and 75,352,491 verbatim rows, plus the archive SHA-256.

The Parquet conversion is physical, not semantic: no taxonomy registry, name
enrichment, Flickr result, media bank, YOLOE route, BioCLIP score, human review,
release artifact, report, or live-run metadata is current project evidence.
Earlier material was staging work and is not an input to this production cycle.

## Retained intake receipt

The ground-zero receipt is
`data/reference/gbif_global_papilionoidea_parquet/dwca_parquet_manifest.json`.
It identifies the source archive as
`data/reference/gbif-global-papilionoidea-download-clean.zip`, with SHA-256
`3494944db9bca6917e0176e63852f8c233cdd7da95904c041c55ff8e8d31c6b9` and
24,398,227,955 bytes.

| DWCA member | Rows | Columns | Zstandard Parquet bytes |
| --- | ---: | ---: | ---: |
| `occurrence.txt` | 75,352,491 | 230 | 6,927,653,319 |
| `multimedia.txt` | 18,680,565 | 15 | 573,367,928 |
| `verbatim.txt` | 75,352,491 | 194 | 4,750,194,656 |

## Production sequence

```text
GBIF DWCA: occurrence.txt + multimedia.txt + verbatim.txt
  → fingerprint and validate the raw members
  → normalize and enrich taxon and name evidence
  → create source-bound Flickr keyword/query definitions
  → exhaust bounded Flickr API discovery from species-specific terms outward
  → validate, deduplicate, and retain photo/query provenance
  → optimize and evaluate YOLOE visual-domain routing
  → run BioCLIP evidence through order, superfamily, family, genus, and species
  → attach source-bound common-name evidence and publish governed handoffs
```

### 1. Raw DWCA intake and fingerprinting

The three raw files remain immutable source evidence. The existing intake
manifest records archive/member checksums, byte and row counts, headers, parser
settings, source download identity, and the physical Parquet outputs. The
Parquet conversion must not be mistaken for semantic enrichment.

### 2. Taxonomic and name enrichment

Fingerprinting is followed by normalization of GBIF taxon identities and
source-bound enrichment of accepted names, synonyms, scientific-name variants,
and common names. Candidate common-name sources include iNaturalist, Wikimedia,
Catalogue of Life, and other documented authorities. Every assertion retains
its source, language or region when available, retrieval date, trust decision,
and homonym/query-risk assessment. A name is a retrieval term, never a label
for a returned photograph.

### 3. Flickr discovery

Discovery begins with the most specific eligible species-level terms, then
expands through the accepted keyword set only where that is justified by the
source-bound taxonomy and query policy. The Flickr API loop retains all pages,
cursors, date or result-window partitions, retries, and logical query
associations. Physical API work may be deduplicated; discovery provenance may
not. A Flickr result is a discovery candidate, not a biodiversity occurrence.

### 4. Visual routing

YOLOE is optimized and evaluated as a route/quality gate for the photo bank.
It distinguishes usable butterfly evidence from moths, other insects, life
stages, specimens, artifacts, no-organism images, and ambiguous material. It
does not decide taxonomy or turn a detector score into a probability.

### 5. Hierarchical BioCLIP evidence

Eligible full-frame images enter a BioCLIP evidence path that retains ranked
candidates at order, superfamily, family, genus, and species. The same path
associates source-bound common-name evidence; it does not manufacture names or
promote a model prediction to a verified identity. Raw scores, margins, and
rank position are evidence, not probabilities or occurrence release decisions.

### 6. Downstream handoffs

Only immutable, fingerprinted BioMiner artifacts can be offered to TaxaLens
and ButterflyLens. Human review, quality estimation, rights handling, and
occurrence release remain separate fail-closed gates. At ground zero there is
no downstream handoff to consume.
