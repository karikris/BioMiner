"""Bounded, restartable local experiment orchestration, independent of production."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.embeddinggemma.common import (
    atomic_json,
    atomic_parquet,
    exclusive_output,
    file_sha256,
    now,
    producer_sha,
    run_lock,
    source_fingerprint,
)


def select_items(
    items: pl.DataFrame, cohort: str, limit: int | None = None
) -> pl.DataFrame:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if cohort == "cache":
        items = items.filter(pl.col("cache_group").is_not_null())
    elif cohort == "reviewed":
        items = items.filter(pl.col("review_status") != "unreviewed")
    elif cohort != "all":
        raise ValueError("Unknown cohort")
    items = items.sort("item_id")
    return items.head(limit) if limit else items


def read_prepared(directory: Path) -> pl.DataFrame:
    from biominer.embeddinggemma.prepare import SCHEMA_VERSION

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Prepared manifest schema is incompatible")
    for name in ("items.parquet", "associations.parquet"):
        if file_sha256(directory / name) != manifest.get("artifacts", {}).get(
            name, {}
        ).get("sha256"):
            raise ValueError("Prepared artifact checksum differs from manifest")
    return pl.read_parquet(directory / "items.parquet")


def cohort_media(items, cache_dir: Path, root: Path, cohort: str, *, workers: int):
    from biominer.embeddinggemma.media import cache_items

    root = Path(root)
    path = root / f"media-{cohort}.parquet"
    receipt_path = root / f"media-{cohort}.json"
    identity = canonical_semantic_fingerprint(
        {"cohort": cohort, "items": items.sort("item_id").to_dicts()}
    )
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["fingerprint"] != identity or file_sha256(path) != receipt["sha256"]:
            raise ValueError("Cohort media snapshot fingerprint or checksum changed")
        media = pl.read_parquet(path)
    else:
        before = time.monotonic()
        media = cache_items(items, cache_dir, workers=workers)
        atomic_parquet(path, media)
        atomic_json(
            receipt_path,
            {
                "schema_version": "embeddinggemma-cohort-media-v1",
                "fingerprint": identity,
                "sha256": file_sha256(path),
                "created_at": now(),
                "rows": media.height,
                "download_seconds": time.monotonic() - before,
                "policy": "frozen_cohort_statuses_and_bytes_for_paired_resume",
            },
        )
    if set(media["item_id"]) != set(items["item_id"]) or media.height != items.height:
        raise ValueError("Cohort media identities differ from selected inputs")
    for row in media.filter(pl.col("status") == "ok").iter_rows(named=True):
        if file_sha256(row["local_path"]) != row["image_hash"]:
            raise ValueError("Cohort cached image checksum changed")
    return media


@exclusive_output()
def runtime_check(
    encoder, output_dir: Path, *, image: Path | None = None, cpu_reference: bool = True
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "embeddinggemma-runtime-check-v1",
        "status": "running",
        "started_at": now(),
        "source_fingerprint": source_fingerprint(),
    }
    atomic_json(output_dir / "receipt.json", receipt)
    try:
        return _runtime_check(
            encoder, output_dir, image=image, cpu_reference=cpu_reference
        )
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, updated_at=now())
        atomic_json(output_dir / "receipt.json", receipt)
        raise


def _runtime_check(
    encoder, output_dir: Path, *, image: Path | None = None, cpu_reference: bool = True
) -> dict:
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps

    from biominer.embeddinggemma.runtime import SentenceEncoder

    output_dir.mkdir(parents=True, exist_ok=True)
    if image is None:
        image = output_dir / "synthetic.png"
        synthetic = Image.new("RGB", (1024, 1024), (31, 92, 60))
        draw = ImageDraw.Draw(synthetic)
        draw.ellipse((400, 250, 650, 700), fill=(211, 165, 32))
        draw.line((0, 0, 1024, 1024), fill=(200, 200, 190), width=8)
        synthetic.save(image)
    rows = []
    vectors = {}
    for budget in (280, 1120):
        with Image.open(image) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
            features = encoder.model[0].preprocess(
                [{"image": rgb}],
                prompt="",
                processing_kwargs={"image": {"max_soft_tokens": budget}},
            )
            token_id = encoder.model[0].processor.image_token_id
            tokens = int((features["input_ids"] == token_id).sum().item())
        if not 0 < tokens <= budget:
            raise RuntimeError("Processor visual token budget verification failed")
        before = time.monotonic()
        vector = encoder.encode_images([image], budget)
        rows.append(
            {
                "budget": budget,
                "effective_visual_tokens": tokens,
                "seconds": time.monotonic() - before,
                "norm": float(np.linalg.norm(vector)),
                "finite": bool(np.isfinite(vector).all()),
            }
        )
        vectors[budget] = vector
    if (
        image.name == "synthetic.png"
        and rows[1]["effective_visual_tokens"] <= rows[0]["effective_visual_tokens"]
    ):
        raise RuntimeError("Higher token budget did not change effective visual input")
    cpu = None
    if cpu_reference:
        reference = (
            encoder
            if encoder.config.device == "cpu"
            else SentenceEncoder(
                replace(encoder.config, device="cpu"), cache_dir=encoder.cache_dir
            )
        )
        cpu = []
        for budget in (280, 1120):
            values = reference.encode_images([image], budget)
            cosine = float((values * vectors[budget]).sum())
            maximum = float(np.max(np.abs(values - vectors[budget])))
            cpu.append(
                {"budget": budget, "cosine": cosine, "max_absolute_difference": maximum}
            )
            if cosine < 0.999 or maximum > 0.01:
                raise RuntimeError("MPS float32 differs materially from CPU reference")
        if reference is not encoder:
            del reference
    encoder.encode_text(
        ["A photograph of a real butterfly.", "A drawing of a butterfly."]
    )
    result = {
        "schema_version": "embeddinggemma-runtime-check-v1",
        "created_at": now(),
        "status": "passed",
        "source_fingerprint": source_fingerprint(),
        "image_hash": file_sha256(image),
        "budgets": rows,
        "cpu_reference": cpu,
        "metrics": encoder.metrics(),
        "producer_sha": producer_sha(),
        "fingerprints": {str(b): encoder.config.fingerprint(b) for b in (280, 1120)},
        "bfloat16": "not_evaluated; float32 is the benchmark precision",
    }
    atomic_json(output_dir / "receipt.json", result)
    return result


@exclusive_output("directory")
def text_embeddings(encoder, labels, directory: Path):
    import numpy as np

    from biominer.embeddinggemma.labels import label_fingerprint
    from biominer.embeddinggemma.runtime import normalize_vectors

    directory.mkdir(parents=True, exist_ok=True)
    identity = canonical_semantic_fingerprint(
        {
            "labels": label_fingerprint(labels),
            "runtime": encoder.config.fingerprint(280),
            "prompt": "Classification",
        }
    )
    path = directory / "embeddings.parquet"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["fingerprint"] != identity
            or file_sha256(path) != manifest["sha256"]
        ):
            raise ValueError("Text embedding fingerprint or checksum changed")
        frame = pl.read_parquet(path)
        if frame["label_id"].to_list() != [label["label_id"] for label in labels]:
            raise ValueError("Text embedding label order changed")
        values = np.asarray(frame["embedding"].to_list(), dtype=np.float32)
        normalize_vectors(values)  # Validate without changing persisted float32 bits.
        return values
    vectors = encoder.encode_text([label["text"] for label in labels])
    vectors = normalize_vectors(vectors)
    if len(vectors) != len(labels):
        raise ValueError("Text encoder returned wrong label count")
    atomic_parquet(
        path,
        pl.DataFrame(
            {
                "label_id": [label["label_id"] for label in labels],
                "embedding": np.asarray(vectors, dtype=np.float32).tolist(),
            },
            schema={"label_id": pl.String, "embedding": pl.List(pl.Float32)},
        ),
    )
    atomic_json(
        manifest_path,
        {
            "fingerprint": identity,
            "sha256": file_sha256(path),
            "labels": len(labels),
            "prompt": "Classification",
            "producer_sha": producer_sha(),
        },
    )
    return vectors


def score_cohort(items, media, labels, text, root, cohort, budget, dimension, config):
    from biominer.embeddinggemma.encode import read_embeddings
    from biominer.embeddinggemma.scoring import score_configuration

    embedding_dir = root / "embeddings" / str(budget)
    manifest = json.loads((embedding_dir / "manifest.json").read_text())
    cached_bytes = (
        media.filter(pl.col("status") == "ok")
        .unique("image_hash")["bytes_downloaded"]
        .sum()
    )
    allowed = set(media.filter(pl.col("status") == "ok")["image_hash"])
    embeddings = read_embeddings(embedding_dir).filter(
        pl.col("image_hash").is_in(allowed)
    )
    return score_configuration(
        items,
        media,
        embeddings,
        labels,
        text,
        root / "scores" / cohort / f"{budget}-{dimension}",
        budget=budget,
        dimension=dimension,
        provenance={
            "fingerprint": manifest["fingerprint"],
            "metrics": {
                **manifest.get("metrics", {}),
                "cached_unique_image_bytes": cached_bytes,
            },
            "comparison_runtime_fingerprint": config.fingerprint(280),
            "cohort": cohort,
            "source_fingerprint": source_fingerprint(),
            "items_fingerprint": canonical_semantic_fingerprint(
                items["item_id"].to_list()
            ),
            "input_manifest_sha256": file_sha256(root / "prepared" / "manifest.json")
            if (root / "prepared" / "manifest.json").exists()
            else None,
            "completed_batches": manifest.get("completed_batches", []),
        },
    )


@contextmanager
def _run_status(root: Path):
    with run_lock(root):
        status = {
            "schema_version": "embeddinggemma-run-v1",
            "status": "running",
            "pid": os.getpid(),
            "started_at": now(),
            "producer_sha": producer_sha(),
            "source_fingerprint": source_fingerprint(),
            "current_stage": "prepare",
            "stages": [],
        }
        atomic_json(root / "run.json", status)
        try:
            yield status
        except BaseException as exc:
            status.update(
                status="failed", error_type=type(exc).__name__, updated_at=now()
            )
            raise
        finally:
            atomic_json(root / "run.json", status)


def audit_cache(items, media, root: Path, *, requested: int):
    joined = items.select("item_id", "cache_group").join(media, on="item_id")
    valid = joined.filter(pl.col("status") == "ok")
    conflicts = set(
        valid.group_by("image_hash")
        .agg(pl.col("cache_group").n_unique().alias("groups"))
        .filter(pl.col("groups") > 1)["image_hash"]
    )
    unique = (
        valid.filter(~pl.col("image_hash").is_in(conflicts))
        .sort("item_id")
        .unique("image_hash", maintain_order=True)
    )
    allowed = unique["item_id"].to_list()
    # Cache benchmark uses actual distinct successfully downloaded image content.
    selected = items.filter(pl.col("item_id").is_in(allowed))
    audited = media.filter(pl.col("item_id").is_in(allowed))
    groups = {}
    for group in ("butterfly", "other_biological", "non_biological"):
        count = selected.filter(pl.col("cache_group") == group).height
        groups[group] = {
            "selected": count,
            "requested": requested,
            "shortfall": max(0, requested - count),
        }
    atomic_json(
        Path(root) / "cache-audit.json",
        {
            "schema_version": "embeddinggemma-cache-audit-v1",
            "created_at": now(),
            "groups": groups,
            "cross_group_content_conflicts": len(conflicts),
            "selected_items": items.height,
            "valid_unique_items": selected.height,
            "excluded_items": items.height - selected.height,
            "download_statuses": dict(media.group_by("status").len().iter_rows()),
            "cached_unique_bytes": valid.unique("image_hash")["bytes_downloaded"].sum(),
        },
    )
    return selected, audited


def run_experiment(args, encoder=None):
    from biominer.embeddinggemma.encode import encode_media
    from biominer.embeddinggemma.labels import load_labels
    from biominer.embeddinggemma.prepare import prepare
    from biominer.embeddinggemma.report import compare_configurations
    from biominer.embeddinggemma.runtime import RuntimeConfig, SentenceEncoder

    root = args.output_dir
    with _run_status(root) as status:
        labels = load_labels(args.label_bank)
        prepare(
            args.input,
            root / "prepared",
            seed=args.seed,
            sample_per_group=args.sample_per_group,
        )
        items = read_prepared(root / "prepared")
        atomic_parquet(root / "labels.parquet", pl.DataFrame(labels))
        config = RuntimeConfig(
            revision=args.revision, device=args.device, dtype=args.dtype
        )
        encoder = encoder or SentenceEncoder(config, cache_dir=args.model_cache)
        receipt_path = root / "runtime-check" / "receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "passed" or receipt.get("fingerprints") != {
                str(b): config.fingerprint(b) for b in (280, 1120)
            }:
                raise ValueError("Runtime receipt fingerprint changed")
        else:
            runtime_check(encoder, root / "runtime-check")
        text = text_embeddings(encoder, labels, root / "text")
        status.update(
            {
                "schema_version": "embeddinggemma-run-v1",
                "pid": os.getpid(),
                "started_at": now(),
                "producer_sha": producer_sha(),
                "status": "running",
                "stages": [],
                "source_fingerprint": source_fingerprint(),
                "evidence_maturity": "experimental_model_evidence",
                "root": str(root.resolve()),
            }
        )
        atomic_json(root / "run.json", status)
        cohorts = ["cache", "reviewed", "all"]
        cohorts = cohorts[: cohorts.index(args.cohort) + 1]
        try:
            for cohort in cohorts:
                selected = select_items(items, cohort, args.limit)
                status.update(
                    current_cohort=cohort, current_stage="download", updated_at=now()
                )
                atomic_json(root / "run.json", status)
                media = cohort_media(
                    selected,
                    args.cache_dir,
                    root,
                    cohort,
                    workers=args.download_workers,
                )
                if cohort == "cache":
                    selected, media = audit_cache(
                        selected, media, root, requested=args.sample_per_group
                    )
                configuration_dirs = []
                for dimension in (256, 768):
                    for budget in (280, 1120):
                        status.update(
                            current_stage=f"{budget}-{dimension}", updated_at=now()
                        )
                        atomic_json(root / "run.json", status)
                        if dimension == 256:
                            result = encode_media(
                                media,
                                root / "embeddings" / str(budget),
                                encoder=encoder,
                                budget=budget,
                                batch_size=args.batch_size,
                            )
                            status["stages"].append(
                                {"cohort": cohort, "budget": budget, **result}
                            )
                        score_cohort(
                            selected,
                            media,
                            labels,
                            text,
                            root,
                            cohort,
                            budget,
                            dimension,
                            config,
                        )
                        configuration_dirs.append(
                            root / "scores" / cohort / f"{budget}-{dimension}"
                        )
                        compare_configurations(
                            configuration_dirs, root / "comparison" / cohort
                        )
                status.update(updated_at=now(), last_complete_cohort=cohort)
                atomic_json(root / "run.json", status)
            status.update(
                status="complete", completed_at=now(), current_stage="complete"
            )
        except BaseException as exc:
            status.update(
                status="failed", error_type=type(exc).__name__, updated_at=now()
            )
            raise
        finally:
            atomic_json(root / "run.json", status)
        return {
            "status": status["status"],
            "items": items.height,
            "report": str(root / "comparison" / args.cohort / "index.html"),
        }


def execute(args):
    command = args.embeddinggemma_command
    if command == "compare":
        from biominer.embeddinggemma.report import compare_configurations

        if not args.configuration_dir:
            raise ValueError("Provide at least one --configuration-dir")
        result = compare_configurations(args.configuration_dir, args.output_dir)
        return {
            "configurations": len(result["configurations"]),
            "report": str(args.output_dir / "index.html"),
        }
    if command == "prepare":
        from biominer.embeddinggemma.labels import load_labels
        from biominer.embeddinggemma.prepare import prepare

        labels = load_labels(args.label_bank)
        receipt = prepare(
            args.input,
            args.output_dir,
            seed=args.seed,
            sample_per_group=args.sample_per_group,
        )
        atomic_parquet(args.output_dir / "labels.parquet", pl.DataFrame(labels))
        return {
            "status": "prepared",
            "output_dir": str(args.output_dir),
            "counts": receipt.get("counts"),
        }
    if command == "run":
        return run_experiment(args)
    from biominer.embeddinggemma.runtime import RuntimeConfig, SentenceEncoder

    config = RuntimeConfig(revision=args.revision, device=args.device, dtype=args.dtype)
    encoder = SentenceEncoder(config, cache_dir=args.model_cache)
    if command == "runtime-check":
        return runtime_check(
            encoder,
            args.output_dir,
            image=args.image,
            cpu_reference=not args.skip_cpu_reference,
        )
    from biominer.embeddinggemma.encode import encode_media, read_embeddings
    from biominer.embeddinggemma.labels import load_labels
    from biominer.embeddinggemma.media import cache_items, read_cached_items
    from biominer.embeddinggemma.scoring import score_configuration

    items = select_items(read_prepared(args.prepared_dir), args.cohort, args.limit)
    if command == "encode":
        media = cache_items(items, args.cache_dir, workers=args.download_workers)
        return encode_media(
            media,
            args.output_dir,
            encoder=encoder,
            budget=args.budget,
            batch_size=args.batch_size,
        )
    labels = load_labels(args.label_bank)
    media = read_cached_items(items, args.cache_dir)
    manifest = json.loads((args.embedding_dir / "manifest.json").read_text())
    if manifest["fingerprint"] != config.fingerprint(args.budget):
        raise ValueError("Image/text runtime fingerprint mismatch")
    text = text_embeddings(encoder, labels, args.output_dir / "text")
    return score_configuration(
        items,
        media,
        read_embeddings(args.embedding_dir),
        labels,
        text,
        args.output_dir,
        budget=args.budget,
        dimension=args.dimension,
        provenance={
            "fingerprint": manifest["fingerprint"],
            "metrics": manifest.get("metrics", {}),
            "comparison_runtime_fingerprint": config.fingerprint(280),
            "cohort": args.cohort,
            "input_manifest_sha256": file_sha256(args.prepared_dir / "manifest.json"),
            "source_fingerprint": source_fingerprint(),
        },
    )
