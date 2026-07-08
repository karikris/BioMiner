#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
DEFAULT_BASE_PATH="$(cd "${REPO_ROOT}/.." && pwd -P)"
BASE_PATH="${BIOMINER_BASE_PATH:-${BIOMINER_RUNTIME_BASE_PATH:-${DEFAULT_BASE_PATH}}}"
ROOT="${BIOMINER_YOLO26_ROOT:-${BASE_PATH}/YOLO26}"
VENV="${ROOT}/venv"
CACHE="${ROOT}/cache"
MODELS="${ROOT}/models"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if command -v sudo >/dev/null 2>&1 && [[ "${1:-}" == "sudo" ]]; then
  echo "Do not run this setup script with sudo." >&2
  exit 2
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "WARNING: Python 3.12 was not found on PATH." >&2
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
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

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

Runtime base path:
  ${BASE_PATH}

Runtime python:
  ${VENV}/bin/python

Cache directories:
  HF_HOME=${HF_HOME}
  HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}
  TORCH_HOME=${TORCH_HOME}
  YOLO_CONFIG_DIR=${YOLO_CONFIG_DIR}
  image cache=${CACHE}/images
  object crop cache=${CACHE}/object_crops
  model dir=${MODELS}

Environment hints:
  export HF_HOME="${HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
  export TORCH_HOME="${TORCH_HOME}"
  export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR}"
  export BIOMINER_YOLO26_MODEL_DIR="${MODELS}"
  export BIOMINER_BASE_PATH="${BASE_PATH}"
  export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK}"

EOF

"${VENV}/bin/python" - <<'PY'
from __future__ import annotations

import importlib.metadata
import platform
import torch

mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print("python", platform.python_version())
print("runtime_python", __import__("sys").executable)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("mps_available", mps_available)
try:
    print("ultralytics", importlib.metadata.version("ultralytics"))
except importlib.metadata.PackageNotFoundError:
    print("ultralytics", "not_installed")
if platform.system() == "Darwin" and not mps_available:
    print("WARNING: MPS is unavailable in this YOLOE-26 runtime. Use --device cpu or fix the Python/PyTorch installation.")
PY
