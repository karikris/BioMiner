# Production Workflow Refactor Guardrails

These guardrails apply during the cleanup toward one production BioMiner workflow.

## Artifact Hygiene

Do not commit generated or machine-local artifacts:

- virtual environments: `.venv*`
- secrets: `.env`, `secrets.env`, API keys, credentials, connection strings
- local operational state: `data/state/`, `*.sqlite`, `*.db`
- caches and downloaded images: `data/cache/`, `data/raw/`, staged images, debug crops
- generated tabular outputs: `*.parquet`, `*.duckdb`, `staging/`, `runs/`
- model files and weights: `*.pt`, `*.pth`, `*.safetensors`, `*.onnx`, `*.ts`

Intentional markdown audit documents under `reports/*.md` may be committed. Generated run reports, JSON metrics, HTML previews, bucket views, and run artifacts remain ignored.

## Workflow Cleanup Rules

- Keep the public biological workflow simple and production-oriented.
- Prefer removing old commands over hiding them once replacement tests are in place.
- Keep local filesystem and SQLite paths only as explicit dev/test fallback, not production defaults.
- Do not make YOLO or YOLOE a species classifier; they are object proposal backends only.
- Do not store reviewed YOLOE boxes for later training in this cleanup path.
- Do not expose low-level registry internals as normal user workflow commands.
- Treat metadata text hints as flags/review evidence, not as a hard pre-visual drop or external keyword file path.
- Use object evidence bucket logic as the single source of truth for production bucket assignment.

## Verification Rules

- Run focused tests after each task and the full suite before each phase push.
- If a phase temporarily allows a failing test, the commit message and phase notes must name the failure and the next task that fixes it before the phase is pushed.
- Do not add tests that require live S3, live Postgres, Flickr credentials, CUDA, BioCLIP weights, YOLO weights, or downloaded images.
