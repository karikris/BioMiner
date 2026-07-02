from __future__ import annotations


POSTGRES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS biominer_runs (
  run_id text PRIMARY KEY,
  job_name text NOT NULL,
  stage text NOT NULL,
  registry_version text,
  status text NOT NULL,
  started_at timestamptz NOT NULL,
  ended_at timestamptz,
  config_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  summary_json jsonb
);

CREATE TABLE IF NOT EXISTS biominer_work_items (
  work_key text PRIMARY KEY,
  job_name text NOT NULL,
  stage text NOT NULL,
  registry_version text,
  status text NOT NULL,
  payload_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  claimed_by text,
  claimed_at timestamptz,
  completed_at timestamptz,
  output_uri text,
  checksum text,
  row_count bigint,
  attempt_count int NOT NULL DEFAULT 0,
  error text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_biominer_work_items_pending
ON biominer_work_items(job_name, stage, registry_version, status, created_at, work_key);

CREATE TABLE IF NOT EXISTS biominer_api_call_ledger (
  id bigserial PRIMARY KEY,
  job_name text NOT NULL,
  work_key text,
  endpoint text NOT NULL,
  status text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz,
  duration_sec double precision,
  http_status int
);

CREATE TABLE IF NOT EXISTS biominer_parquet_shards (
  shard_id text PRIMARY KEY,
  job_name text NOT NULL,
  registry_version text,
  stage text NOT NULL,
  run_id text NOT NULL,
  worker_id text NOT NULL,
  uri text NOT NULL UNIQUE,
  row_count bigint,
  byte_count bigint,
  checksum text,
  metadata_json jsonb,
  committed_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS biominer_compaction_inputs (
  compaction_run_id text NOT NULL,
  output_shard_id text NOT NULL,
  source_shard_id text NOT NULL,
  source_uri text NOT NULL,
  job_name text NOT NULL,
  source_stage text NOT NULL,
  output_stage text NOT NULL,
  registry_version text,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (output_shard_id, source_shard_id)
);

CREATE INDEX IF NOT EXISTS idx_biominer_compaction_inputs_source
ON biominer_compaction_inputs(job_name, source_stage, registry_version, source_shard_id);
"""


POSTGRES_CLAIM_SQL = """
WITH picked AS (
  SELECT work_key
  FROM biominer_work_items
  WHERE job_name = %s
    AND stage = %s
    AND (registry_version IS NOT DISTINCT FROM %s)
    AND status = 'pending'
  ORDER BY created_at, work_key
  FOR UPDATE SKIP LOCKED
  LIMIT %s
)
UPDATE biominer_work_items w
SET status = 'claimed',
    claimed_by = %s,
    claimed_at = now(),
    attempt_count = attempt_count + 1
FROM picked
WHERE w.work_key = picked.work_key
RETURNING w.*;
"""
