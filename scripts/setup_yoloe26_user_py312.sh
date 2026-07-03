#!/usr/bin/env bash
set -euo pipefail

ROOT="${BIOMINER_YOLO26_ROOT:-${HOME}/Applications/YOLO26}"
VENV="${ROOT}/venv"
CACHE="${ROOT}/cache"
MODELS="${ROOT}/models"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if command -v sudo >/dev/null 2>&1 && [[ "${1:-}" == "sudo" ]]; then
  echo "Do not run this setup script with sudo." >&2
  exit 2
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python 3.12 is required. Set PYTHON_BIN=/path/to/python3.12 if it is not on PATH." >&2
  exit 2
fi

mkdir -p "${ROOT}" "${CACHE}/huggingface/hub" "${CACHE}/torch" "${CACHE}/ultralytics" "${CACHE}/images" "${CACHE}/object_crops" "${MODELS}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV}"
fi

export HF_HOME="${CACHE}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE}/huggingface/hub"
export TORCH_HOME="${CACHE}/torch"
export YOLO_CONFIG_DIR="${CACHE}/ultralytics"

"${VENV}/bin/python" -m pip install --upgrade pip wheel setuptools

PLATFORM="$(uname -s)"
ARCH="$(uname -m)"
TORCH_INDEX=()
if [[ "${PLATFORM}" == "Darwin" ]]; then
  echo "Detected macOS ${ARCH}; installing standard PyTorch wheels with MPS support when available."
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "Detected NVIDIA tooling; installing PyTorch CUDA 12.9 wheels."
  TORCH_INDEX=(--index-url https://download.pytorch.org/whl/cu129)
else
  echo "No NVIDIA tooling detected; installing CPU/standard PyTorch wheels."
fi

"${VENV}/bin/python" -m pip install "${TORCH_INDEX[@]}" torch torchvision torchaudio
"${VENV}/bin/python" -m pip install \
  ultralytics \
  pillow \
  numpy \
  opencv-python-headless \
  polars \
  pyarrow \
  huggingface_hub \
  safetensors

cat <<EOF

YOLOE-26 runtime installed.

Runtime python:
  ${VENV}/bin/python

Environment hints:
  export HF_HOME="${HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
  export TORCH_HOME="${TORCH_HOME}"
  export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR}"
  export BIOMINER_YOLO26_MODEL_DIR="${MODELS}"

EOF

"${VENV}/bin/python" - <<'PY'
from __future__ import annotations

import importlib.metadata
import platform
import torch

print("python", platform.python_version())
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("mps_available", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
try:
    print("ultralytics", importlib.metadata.version("ultralytics"))
except importlib.metadata.PackageNotFoundError:
    print("ultralytics", "not_installed")
PY

