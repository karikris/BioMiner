#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_REGISTRY="$ROOT/data/registry/butterflies-v1"
CANONICAL_REGISTRY="/home/toffe/github/karikris/BioMiner/data/registry/butterflies-v1"
if [[ ! -f "$DEFAULT_REGISTRY/taxa.parquet" && -f "$CANONICAL_REGISTRY/taxa.parquet" ]]; then
  DEFAULT_REGISTRY="$CANONICAL_REGISTRY"
fi

REGISTRY_DIR="${REGISTRY_DIR:-$DEFAULT_REGISTRY}"
REPORT_DIR="${REPORT_DIR:-$ROOT/reports}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
RUN_ID="${RUN_ID:-registry_enrich_wikidata_butterflies_v1}"

mkdir -p "$REPORT_DIR" "$LOG_DIR"
if [[ ! -f "$REGISTRY_DIR/taxa.parquet" ]]; then
  echo "missing registry input: $REGISTRY_DIR/taxa.parquet" >&2
  exit 1
fi

LOG="$LOG_DIR/${RUN_ID}_$(date -u +%Y%m%d_%H%M%S).log"
nohup "$ROOT/.venv/bin/biominer" registry enrich-sources \
  --registry-dir "$REGISTRY_DIR" \
  --sources wikidata \
  --workers 1 \
  --progress-every "${PROGRESS_EVERY:-100}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-500}" \
  --max-retries "${MAX_RETRIES:-5}" \
  --report-dir "$REPORT_DIR" \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$REPORT_DIR/${RUN_ID}.pid"
echo "started ${RUN_ID} pid=${PID} log=${LOG} registry=${REGISTRY_DIR}"
