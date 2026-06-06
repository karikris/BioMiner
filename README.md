# BioMiner

BioMiner is a Python 3.14-oriented Flickr image-triage pipeline for BioCLIP-based visual evidence.

The active workflow stores source metadata, image URLs, geolocation/time fields, hashes, model outputs, and triage bins. Downloaded image files are temporary and are deleted after classification. Darwin Core export code is retained only for compatibility with existing tests and is not the current implementation focus.

BioCLIP, BioCLIP 2, BioCLIP 2.5 Huge, PyTorch image classification, and image embedding workflows have moved to `karikris/BioCLIPMiner`, which targets Python 3.12 for the BioCLIP/OpenCLIP/PyTorch runtime.
