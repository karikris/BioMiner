#!/usr/bin/env bash
# Explicit interpreter selection avoids silently activating a contained runtime.
set -euo pipefail
: "${BIOMINER_PYTHON:?Set BIOMINER_PYTHON to the qualified Python interpreter}"
test -x "$BIOMINER_PYTHON"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
launch_root="${XDG_STATE_HOME:-$HOME/.local/state}/biominer/image-keyword-launches"
mkdir -p "$launch_root"
launch_dir="$(mktemp -d "$launch_root/run-XXXXXXXX")"
mkdir "$launch_dir/scratch"
nohup env -i HOME="$HOME" LANG=C.UTF-8 PATH=/usr/bin:/bin \
    PYTHONUNBUFFERED=1 POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    "$BIOMINER_PYTHON" -I -B -m biominer.polars_runtime \
    --scratch-parent "$launch_dir/scratch" --report-dir "$launch_dir/runtime" \
    --timeout 86400 --grace-seconds 30 -- \
    "$repo_root/scripts/enrich_butterfly_image_species.py" "$@" \
    > "$launch_dir/launcher.log" 2>&1 < /dev/null &
worker_pid=$!
printf '%s\n' "$worker_pid" > "$launch_dir/pid"
printf 'Supervisor PID: %s\nLauncher log: %s/launcher.log\n' "$worker_pid" "$launch_dir"
