# ADR: production command surface

Status: accepted.

`biominer run` is the sole production workflow. Registry build/classification/audit, evaluation, and infrastructure doctors remain public because they create or validate production inputs rather than execute a competing visual pipeline.

Runtime checks, model prefetch, smoke tests, previews, and benchmarks are nested under `biominer dev`. Direct detect, screen, rolling-screen, score, and ablation commands are removed immediately and have no aliases.

The parser and tests must reject removed commands. Current documentation describes only supported commands; historical migration and deprecation documents are not maintained.
