"""Explicit local experiment CLI; heavyweight model libraries are imported lazily."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from biominer.embeddinggemma.runtime import MODEL_REVISION, VISION_BUDGETS


def add_parser(subparsers):
    vision = subparsers.add_parser(
        "vision", help="experimental visual evidence workflows"
    )
    sub = vision.add_subparsers(dest="vision_command", required=True)
    gemma = sub.add_parser("embeddinggemma", help="local EmbeddingGemma 2 benchmark")
    commands = gemma.add_subparsers(dest="embeddinggemma_command", required=True)
    for name in ("runtime-check", "prepare", "encode", "score", "compare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
        if name in {"runtime-check", "encode", "score", "run"}:
            command.add_argument("--device", choices=("mps", "cpu"), default="mps")
            command.add_argument("--dtype", choices=("float32",), default="float32")
            command.add_argument("--revision", default=MODEL_REVISION)
            command.add_argument(
                "--model-cache", type=Path, default=Path("data/models/embeddinggemma2")
            )
        if name in {"prepare", "run"}:
            command.add_argument("--input", type=Path, action="append", required=True)
            command.add_argument("--sample-per-group", type=int, default=500)
            command.add_argument("--seed", default="embeddinggemma-v1")
            command.add_argument("--sample-selection", type=Path)
        if name in {"prepare", "score", "run"}:
            command.add_argument("--label-bank", type=Path, required=True)
        if name in {"encode", "score"}:
            command.add_argument("--prepared-dir", type=Path, required=True)
            command.add_argument(
                "--budget", type=int, choices=VISION_BUDGETS, required=True
            )
        if name == "run":
            command.add_argument(
                "--configuration",
                action="append",
                choices=tuple(
                    f"{budget}-{dimension}"
                    for dimension in (256, 768)
                    for budget in VISION_BUDGETS
                ),
                help="Repeat in requested execution order; default preserves the first benchmark",
            )
        if name in {"score", "run"}:
            command.add_argument(
                "--binary",
                action="store_true",
                help="Also publish butterfly/not-butterfly evidence with explicit unknowns",
            )
        if name in {"encode", "score", "run"}:
            command.add_argument(
                "--cache-dir", type=Path, default=Path("data/cache/embeddinggemma2")
            )
            command.add_argument(
                "--cohort", choices=("cache", "reviewed", "all"), default="all"
            )
            command.add_argument("--limit", type=int)
        if name in {"encode", "run"}:
            command.add_argument("--batch-size", type=int, default=4)
            command.add_argument("--download-workers", type=int, default=4)
        if name == "score":
            command.add_argument("--embedding-dir", type=Path, required=True)
            command.add_argument(
                "--dimension", type=int, choices=(256, 768), required=True
            )
        if name == "compare":
            command.add_argument(
                "--configuration-dir", type=Path, action="append", default=[]
            )
        if name == "runtime-check":
            command.add_argument("--image", type=Path)
            command.add_argument("--skip-cpu-reference", action="store_true")
            command.add_argument(
                "--budget", type=int, choices=VISION_BUDGETS, action="append"
            )
    return vision


def run_command(args: argparse.Namespace) -> int:
    from biominer.embeddinggemma import workflow

    try:
        result = workflow.execute(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        # URL-bearing third-party exceptions must never enter normal logs.
        message = str(exc)
        if "://" in message:
            message = type(exc).__name__ + ": inspect local failure evidence"
        print(message, file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
