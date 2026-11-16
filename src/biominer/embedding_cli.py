"""Developer-only EmbeddingGemma 2 controls; model imports stay in the runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from biominer.detection.embeddinggemma2 import (
    DEFAULT_EMBEDDINGGEMMA2_MODEL,
    DEFAULT_EMBEDDINGGEMMA2_REVISION,
    EmbeddingGemma2Config,
)
from biominer.runtime_paths import EMBEDDINGGEMMA2_DIR, embeddinggemma2_runtime_env


COMMANDS = frozenset({"embeddinggemma2-runtime-check", "embeddinggemma2-prefetch",
                      "embeddinggemma2-smoke", "embedding-flickr-sample", "embedding-flickr-sharded"})


def add_embedding_commands(subparsers: Any) -> None:
    for command in sorted(COMMANDS):
        parser = subparsers.add_parser(command, help="provisional full-frame embedding evidence")
        parser.add_argument("--runtime-python", default=str(EMBEDDINGGEMMA2_DIR / "venv/bin/python"))
        parser.add_argument("--model-id", default=DEFAULT_EMBEDDINGGEMMA2_MODEL)
        parser.add_argument("--revision", default=DEFAULT_EMBEDDINGGEMMA2_REVISION)
        parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
        parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
        parser.add_argument("--min-similarity", type=float, default=0.25)
        parser.add_argument("--min-margin", type=float, default=0.05)
        if command == "embeddinggemma2-smoke":
            parser.add_argument("--image", required=True)
            parser.add_argument("--output-dir", default="reports/embeddinggemma2_smoke")
        if command.startswith("embedding-flickr"):
            parser.add_argument("--state-db", required=True)
            parser.add_argument("--output-dir", required=True)
            parser.add_argument("--reports-dir", default="reports")
            parser.add_argument("--sample-seed", default="flickr-embeddinggemma2-v1")
            parser.add_argument("--download-workers", type=int, default=4)
            parser.add_argument("--batch-size", type=int, default=8)
            parser.add_argument("--timeout-seconds", type=float, default=30.0)
            parser.add_argument("--retries", type=int, default=2)
            parser.add_argument("--max-image-bytes", type=int, default=20_000_000)
            parser.add_argument("--max-attempts", type=int, default=3)
            if command.endswith("sample"):
                parser.add_argument("--sample-size", type=int, default=10_000)
            else:
                parser.add_argument("--shards", type=int, default=4)
                parser.add_argument("--expected-images", type=int)


def run_embedding_command(args: argparse.Namespace) -> int:
    try:
        config = EmbeddingGemma2Config(model_id=args.model_id, revision=args.revision,
            device=args.device, dtype=args.dtype, min_similarity=args.min_similarity,
            min_margin=args.min_margin)
        if args.vision_command.startswith("embedding-flickr"):
            return _run_screening(args, config)
        runtime = Path(args.runtime_python).expanduser().absolute()
        if not runtime.is_file():
            raise ValueError(f"EmbeddingGemma 2 runtime Python not found: {runtime}")
        operation = args.vision_command.removeprefix("embeddinggemma2-")
        command = [str(runtime), "-m", "biominer.detection.embedding_runtime", operation,
                   "--config-json", json.dumps(config.as_dict(), sort_keys=True)]
        if operation == "smoke":
            command.extend(["--image", str(Path(args.image).expanduser().resolve()),
                            "--output-dir", str(Path(args.output_dir).expanduser().resolve())])
        result = subprocess.run(command, capture_output=True, text=True, check=False,
                                env=_runtime_env(runtime))
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "embedding runtime failed")
        print(result.stdout.strip())
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 2


def _run_screening(args, model):
    from biominer.flickr_fetch.embedding_screening import EmbeddingScreeningConfig, run_embedding_screening

    result = run_embedding_screening(EmbeddingScreeningConfig(
        state_db=Path(args.state_db), output_dir=Path(args.output_dir), reports_dir=Path(args.reports_dir),
        runtime_python=Path(args.runtime_python), model=model,
        sample_size=getattr(args, "sample_size", None), sample_seed=args.sample_seed,
        shard_count=getattr(args, "shards", 1), expected_images=getattr(args, "expected_images", None),
        download_workers=args.download_workers, batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds, retries=args.retries,
        max_image_bytes=args.max_image_bytes, max_attempts=args.max_attempts,
    ))
    print(json.dumps({"manifest": str(result.manifest_path), "routes": str(result.results_path),
                      "failures": str(result.failures_path), "report": result.report}, indent=2, sort_keys=True))
    return 0 if result.report["status"] == "complete" else 2


def _runtime_env(runtime: Path) -> dict[str, str]:
    return embeddinggemma2_runtime_env(runtime)
