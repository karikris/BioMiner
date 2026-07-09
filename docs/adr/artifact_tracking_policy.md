# Artifact Tracking Policy

Status: accepted

BioMiner treats generated reports, run manifests, metrics, plots, local decks, and pipeline outputs as runtime artifacts. They belong under `reports/`, `runs/`, `staging/`, or external object storage and are not committed to the repository.

Source-controlled documentation must be durable project guidance: README sections, workflow docs, runbooks, schemas, and ADRs. If a generated audit report contains a decision worth preserving, move only the decision into `docs/adr/` and delete the report artifact from version control.

The `.gitignore` policy intentionally ignores all files below `reports/`. Tests must validate maintained docs and policy, not historical report files.
