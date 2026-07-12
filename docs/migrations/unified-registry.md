# Unified registry cutover

The cutover is local-first and destructive remote actions are deliberately separate.

1. Build from CoL XR dataset `315557`, release `COL26.6 XR`.
2. Audit and publish to a temporary local `current` directory.
3. Build the taxonomy text-embedding cache from that directory.
4. Run a mixed-family model-free dry run and, when the Python 3.12 sidecar is available, the BioCLIP runtime check.
5. Only after acceptance, remove the incompatible remote registry and classification-v3 cache.
6. Upload the six Parquet files, then `manifest.json` last.
7. Upload the rebuilt cache under `cache/taxonomy/current/` and reload workers.

Do not recreate historical registry aliases, classification-v3 roots, or test roots. The S3 deletion and upload are operator-authorized deployment steps; local registry build and tests never perform them.
