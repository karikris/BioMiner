# BioMiner

BioMiner is a Python 3.14-oriented Flickr image-triage pipeline for Lepidoptera life-stage occurrence screening.

The active workflow stores source metadata, image URLs, geolocation/time fields, hashes, model outputs, life-stage/category fields, and triage bins. Downloaded image files are temporary and are deleted after classification. BioCLIP/BioCLIP 2.5 output is screening evidence only and is not taxonomic validation.

The current report pack focuses on query terms, bbox coverage, occurrence bins, life stages, no-geo records, comment expansion, targeted comment review, missing-data requests, API budget, and code cleanup. Darwin Core mapper/exporter code is retained only for compatibility tests and is not the active occurrence-publication path.

Comment review is a separate post-triage phase. It is used only for selected mismatch, no-geo, missing-date, low-confidence, or unknown-category records, because Flickr comments require separate API calls from `photos.search`.

BioCLIP, BioCLIP 2, BioCLIP 2.5 Huge, PyTorch image classification, and image embedding workflows have moved to `karikris/BioCLIPMiner`, which targets Python 3.12 for the BioCLIP/OpenCLIP/PyTorch runtime.
