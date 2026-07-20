# BioMiner

BioMiner is the scientific evidence engine for a GBIF-first butterfly and
insect discovery pipeline.

## Current production scope: GBIF ground zero

The production cycle begins with a single three-member GBIF Darwin Core Archive
download: `occurrence.txt`, `multimedia.txt`, and `verbatim.txt`. Its validated
Zstandard Parquet representation records 75,352,491 occurrence rows,
18,680,565 multimedia rows, and 75,352,491 verbatim rows. This is a physical
intake only: no semantic transformation, enrichment, Flickr query, YOLOE route,
or BioCLIP score has been produced. Consequently, this repository currently has
no production registry, Flickr corpus, visual evidence, or release artifact.

The complete active sequence and its evidence boundaries are in
[GBIF ground-zero pipeline](docs/PIPELINE_GROUND_ZERO.md).

```text
GBIF raw DWCA → fingerprints and validation → source-bound name enrichment
→ species-first Flickr discovery → provenance-preserving photo bank
→ YOLOE butterfly/moth/other-insect routing → hierarchical BioCLIP evidence
→ governed downstream handoffs
```

Taxonomy and common names may be enriched from iNaturalist, Wikimedia,
Catalogue of Life, and other documented sources, but each assertion remains
source-bound and a query term never labels a photo. YOLOE routes visual
evidence; BioCLIP ranks evidence at order, superfamily, family, genus, and
species. Neither model output nor a Flickr result is a verified occurrence.

## Operating the DWCA intake

The raw archive can be converted to physical Parquet representations with the
streaming converter; this does not perform semantic enrichment:

```bash
python3 scripts/build_dwca_members_parquet.py \
  --archive data/reference/gbif-global-papilionoidea-download-clean.zip \
  --output-dir data/reference/gbif_global_papilionoidea_parquet
```

The converter writes Zstandard Parquet plus a member and archive manifest.
Do not treat its output as a registry or a completed intake until the required
fingerprinting and validation artifacts have been produced.

## Repository boundaries

BioMiner creates immutable scientific artifacts. TaxaLens may consume completed
artifacts for verification, replay, and research interaction; ButterflyLens
may consume governed handoffs for its public/community product. Neither
downstream repository has a current BioMiner production handoff at ground zero.
