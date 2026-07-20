# Current state

## Active production baseline

BioMiner is at **GBIF ground zero**. The active production input is the
three-member GBIF DWCA (`occurrence.txt`, `multimedia.txt`, `verbatim.txt`) and
its checksum-bound physical Parquet representation: 75,352,491 occurrence
rows, 18,680,565 multimedia rows, and 75,352,491 verbatim rows. It has not been
semantically transformed.

Do not use, revive, or cite previous run reports, Flickr data, caches,
registries, reference pools, model outputs, manifests, or release metadata.
They were staging artifacts and have no standing in the reset production cycle.

The required next stage is source/member fingerprinting and validation. The
full sequence and non-negotiable evidence boundaries are in
[GBIF ground-zero pipeline](../PIPELINE_GROUND_ZERO.md).

## Active handoff state

There is no current BioMiner handoff for TaxaLens or ButterflyLens. Downstream
work may consume only immutable, fingerprinted artifacts created after this
ground-zero intake.
