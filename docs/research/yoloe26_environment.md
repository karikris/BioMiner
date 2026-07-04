# YOLOE-26 Prototype Environment

Recorded before adding the BioMiner YOLOE-26 prototype code, using GitHub `origin/main` as the source of truth.

## Host

- OS: macOS 26.5.1
- Kernel: Darwin 25.5.0
- Platform: macOS on Apple Silicon
- Shell: `/bin/bash`
- Working directory: `/Users/merm0001/Applications/BioMiner`
- Memory: 64 GB
- Free disk: about 1.7 TiB on the main data volume

## Tooling

- `python`: not installed on PATH
- `python3`: 3.9.6
- `python3.12`: 3.12.13
- `python3.14`: 3.14.6
- `uv`: 0.11.24
- `pip`: not installed as a top-level command; `python3 -m pip` is 21.2.4
- `nvidia-smi`: not available
- Core BioMiner Python 3.14 runtime: no `torch` installed

## Repository

- Branch: `feature/yoloe26-prototype`
- Base: `origin/main`
- Base commit: `52f5b83 infra: scope cloud doctor work item`
- Historical audit note: at prototype start, detector helpers lived under `uv run biominer detect --help` and BioCLIP helpers lived under `uv run biominer bioclip --help`.
- Current cleanup direction: model runtime and detector debug helpers are consolidated under `uv run biominer vision --help`.

## External Runtime Layout

Heavy vision dependencies and model files are intentionally outside the BioMiner repository:

- BioMiner checkout: `./BioMiner`
- YOLOE-26 runtime: `./YOLO26/venv/bin/python`
- YOLOE-26 cache: `./YOLO26/cache`
- YOLOE-26 model directory: `./YOLO26/models`
- BioCLIP 2.5 runtime: `./BioCLIP25/venv/bin/python`
- BioCLIP 2.5 cache: `./BioCLIP25/cache`
- BioCLIP 2.5 model directory: `./BioCLIP25/models`

Validation already completed outside the repo:

- YOLOE runtime imports `YOLOE`, loads `yoloe-26s-seg.pt`, and runs `set_classes`.
- BioCLIP runtime loads `hf-hub:imageomics/bioclip-2.5-vith14` through OpenCLIP.
- Both runtimes report Apple MPS availability.
