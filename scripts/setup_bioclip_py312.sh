#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUNTIME_DIR="${BIOMINER_BIOCLIP_RUNTIME_DIR:-.venv-bioclip-py312}"
PYTHON_VERSION="${BIOMINER_BIOCLIP_PYTHON:-3.12}"

uv venv --no-project --allow-existing --python "$PYTHON_VERSION" "$RUNTIME_DIR"

uv pip install --python "$RUNTIME_DIR/bin/python" \
  "torch>=2.12,<2.13" \
  "torchvision>=0.27,<0.28" \
  "open_clip_torch>=3.3,<4" \
  "pillow" \
  "safetensors" \
  "huggingface_hub" \
  "hf_xet"

"$RUNTIME_DIR/bin/python" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda={torch.cuda.is_available()}")
print(f"mps={torch.backends.mps.is_available()}")
PY
