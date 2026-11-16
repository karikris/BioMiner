#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
runtime_base="${BIOMINER_BASE_PATH:-$(cd "${repo_root}/.." && pwd -P)}"
embedding_root="${BIOMINER_EMBEDDINGGEMMA2_ROOT:-${runtime_base}/EmbeddingGemma2}"
embedding_python="${PYTHON_BIN:-python3.14}"

"${embedding_python}" -c 'import sys; assert sys.version_info >= (3, 14, 7), "Python >=3.14.7 is required"'
if [[ ! -x "${embedding_root}/venv/bin/python" ]]; then
  "${embedding_python}" -m venv "${embedding_root}/venv"
fi
"${embedding_root}/venv/bin/python" -m pip install --upgrade pip
"${embedding_root}/venv/bin/python" -m pip install "${repo_root}[embeddinggemma2]"
printf 'EmbeddingGemma 2 runtime: %s\n' "${embedding_root}/venv/bin/python"
printf 'Model weights are fetched only by an explicit prefetch or inference command.\n'
