"""Local, resumable full-frame BioCLIP suggestions, separate from human review.

The model sees only pixels. Metadata is retained for inspection and review sampling,
never used to manufacture an image prediction or a wild-occurrence decision.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import warnings

import numpy as np
from PIL import Image, ImageOps
import polars as pl

from biominer.bioclip.bioclip import PersistentBioClipScorer
from biominer.bioclip.bioclip_worker import decoded_rgb_image_content_hash
from biominer.bioclip.model_registry import BioClipRuntime, ModelConfig
from biominer.common.semantic_hash import canonical_semantic_fingerprint

MODEL = "imageomics/bioclip-2.5-vith14"
REVISION = "191d741545e4c741cdef4b22c6eb69c945c1e592"
PREPARE_VERSION = "exif-transpose-rgb-png-full-frame-v1"
PREFIX = "bioclip_"


def utc_now():
    return datetime.now(UTC).isoformat()


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_labels(path):
    config = json.loads(Path(path).read_text())
    if not config.get("version") or not config.get("templates"):
        raise ValueError("labels require a version and templates")
    if set(config.get("heads", {})) != {"subject", "context"}:
        raise ValueError("labels require separate subject and context heads")
    for template in config["templates"]:
        if "{description}" not in template:
            raise ValueError("each template must contain {description}")
        template.format(description="test")
    for labels in config["heads"].values():
        if len(labels) < 3 or any(not k or not v for k, v in labels.items()):
            raise ValueError("each head requires at least three named descriptions")
        for descriptions in labels.values():
            descriptions = [descriptions] if isinstance(descriptions, str) else descriptions
            if not isinstance(descriptions, list) or any(not isinstance(v, str) or not v.strip() for v in descriptions):
                raise ValueError("label descriptions must be a string or a nonempty list of strings")
    return config


def normalize_vectors(values):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("embeddings must be a finite matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("zero embedding")
    return array / norms


def text_prototypes(scorer, config):
    keys, prompts, counts = [], [], []
    for head, labels in config["heads"].items():
        for label, descriptions in labels.items():
            keys.append((head, label))
            descriptions = [descriptions] if isinstance(descriptions, str) else descriptions
            texts = [t.format(description=d) for d in descriptions for t in config["templates"]]
            counts.append(len(texts))
            prompts.extend(texts)
    vectors = normalize_vectors(scorer.embed_text_labels(prompts))
    offset, means = 0, []
    for count in counts:
        means.append(vectors[offset:offset + count].mean(axis=0))
        offset += count
    vectors = normalize_vectors(means)
    return keys, vectors, prompts


def score_vector(vector, keys, prototypes):
    similarities = normalize_vectors([vector])[0] @ prototypes.T
    output = {"bioclip_status": "scored", "bioclip_wild_status": "unresolved"}
    for head in ("subject", "context"):
        ranked = sorted(
            ((label, float(similarities[i])) for i, (h, label) in enumerate(keys) if h == head),
            key=lambda item: (-item[1], item[0]),
        )
        for rank, (label, score) in enumerate(ranked[:3], 1):
            output[f"bioclip_{head}_top{rank}"] = label
            output[f"bioclip_{head}_top{rank}_cosine"] = score
        output[f"bioclip_{head}_margin"] = ranked[0][1] - ranked[1][1]
        for label, score in ranked:
            output[f"bioclip_{head}_cosine__{label}"] = score
    return output


class RunState:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS media(photo_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS embeddings(cache_key TEXT PRIMARY KEY, vector BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions(
              photo_id TEXT, scoring_key TEXT, payload TEXT NOT NULL,
              PRIMARY KEY(photo_id, scoring_key));
        """)

    def media(self):
        return {i: json.loads(p) for i, p in self.connection.execute("SELECT * FROM media")}

    def save_media(self, photo_id, payload):
        self.connection.execute("INSERT OR REPLACE INTO media VALUES (?, ?)", (photo_id, json.dumps(payload)))

    def embedding(self, key):
        row = self.connection.execute("SELECT vector FROM embeddings WHERE cache_key=?", (key,)).fetchone()
        return np.frombuffer(row[0], dtype="<f4").copy() if row else None

    def save_embedding(self, key, vector):
        self.connection.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, np.asarray(vector, dtype="<f4").tobytes()))

    def predictions(self, scoring_key):
        return {i: json.loads(p) for i, p in self.connection.execute(
            "SELECT photo_id,payload FROM predictions WHERE scoring_key=?", (scoring_key,))}

    def save_prediction(self, photo_id, scoring_key, payload):
        self.connection.execute("INSERT OR REPLACE INTO predictions VALUES (?, ?, ?)", (photo_id, scoring_key, json.dumps(payload)))

    def commit(self):
        self.connection.commit()


def embedding_key(media, attestation):
    return canonical_semantic_fingerprint({
        "pixel_sha256": media["pixel_sha256"], "preparation": PREPARE_VERSION,
        "model": attestation["model_fingerprint"],
        "preprocessing": attestation["preprocessing_fingerprint"],
    })


class MediaDownloader:
    def __init__(self, root, supplied_urls, cached=None, *, opener=urlopen,
                 max_bytes=20 * 1024 * 1024, max_pixels=50_000_000,
                 attempts=3, backoff=1, retry_failures=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.supplied_urls, self.cached, self.opener = supplied_urls, cached or {}, opener
        self.max_bytes, self.max_pixels = max_bytes, max_pixels
        self.attempts, self.backoff, self.retry_failures = attempts, backoff, retry_failures

    def download(self, row):
        photo_id = str(row["flickr_photo_id"])
        previous = self.cached.get(photo_id)
        if previous and previous.get("status") == "ready":
            path = Path(previous["canonical_path"])
            raw = Path(previous["original_path"])
            if (path.is_file() and raw.is_file()
                    and sha256_file(path) == previous["canonical_sha256"]
                    and sha256_file(raw) == previous["byte_sha256"]):
                return photo_id, previous
        elif previous and not self.retry_failures:
            return photo_id, previous
        choices = self.supplied_urls.get(photo_id)
        if not choices:
            choices = [(row.get("image_url_kind", "source"), row.get("image_url", ""))]
        failures = []
        for rendition, url in choices:
            if urlparse(url).scheme != "https":
                failures.append(f"{rendition}: HTTPS required")
                continue
            for attempt in range(self.attempts):
                try:
                    request = Request(url, headers={"User-Agent": "BioMiner/0.1 (full-frame research image classification)"})
                    with self.opener(request, timeout=30) as response:
                        if response.status != 200 or not response.headers.get("Content-Type", "").lower().startswith("image/"):
                            raise ValueError("HTTP response is not a successful image")
                        if int(response.headers.get("Content-Length", "0")) > self.max_bytes:
                            raise ValueError("image exceeds byte limit")
                        body = bytearray()
                        while chunk := response.read(min(65536, self.max_bytes + 1 - len(body))):
                            body.extend(chunk)
                            if len(body) > self.max_bytes:
                                raise ValueError("image exceeds byte limit")
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", Image.DecompressionBombWarning)
                        with Image.open(io.BytesIO(body)) as image:
                            if image.width * image.height > self.max_pixels:
                                raise ValueError("image exceeds decoded pixel limit")
                            image.load()
                            canonical = ImageOps.exif_transpose(image).convert("RGB")
                    pixel_hash = decoded_rgb_image_content_hash(canonical)
                    file_hash = hashlib.sha256(body).hexdigest()
                    path = self.root / f"{pixel_hash}.png"
                    original = self.root / f"{file_hash}.original"
                    # Per-photo temporary files avoid races between identical media downloads.
                    temporary = self.root / f"{photo_id}.png.tmp"
                    canonical.save(temporary, format="PNG")
                    temporary.replace(path)
                    temporary = self.root / f"{photo_id}.original.tmp"
                    temporary.write_bytes(body)
                    temporary.replace(original)
                    return photo_id, {
                        "status": "ready", "selected_url": url, "rendition": rendition,
                        "pixel_sha256": pixel_hash, "byte_sha256": file_hash,
                        "canonical_sha256": sha256_file(path), "canonical_path": str(path.resolve()),
                        "original_path": str(original.resolve()), "width": canonical.width,
                        "height": canonical.height, "downloaded_at_utc": utc_now(),
                        "earlier_attempt_errors": failures,
                    }
                except Exception as error:
                    failures.append(f"{rendition} attempt {attempt + 1}: {type(error).__name__}: {error}")
                    if attempt + 1 < self.attempts:
                        time.sleep(self.backoff * 2 ** attempt)
        return photo_id, {"status": "media_failure", "errors": failures, "attempted_at_utc": utc_now()}


def supplied_urls(path):
    if path is None:
        return {}
    connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    result = {}
    try:
        for photo_id, payload in connection.execute("SELECT flickr_photo_id,raw_json FROM source_records"):
            raw = json.loads(payload)
            result[str(photo_id)] = [(key, raw[key]) for key in ("url_l", "url_m") if raw.get(key)]
    finally:
        connection.close()
    return result


def snapshot_inputs(source, metadata, root):
    identity_path = root / "input_manifest.json"
    requested_sha = sha256_file(source)
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        if requested_sha != identity["source_sha256"]:
            raise ValueError("resume source fingerprint changed; choose a new output directory")
        for filename, digest in identity["snapshots"].items():
            if sha256_file(root / filename) != digest:
                raise ValueError("immutable input snapshot was modified")
    else:
        shutil.copyfile(source, root / "input_source.parquet")
        snapshots = {"input_source.parquet": sha256_file(root / "input_source.parquet")}
        if metadata:
            shutil.copyfile(metadata, root / "input_metadata.parquet")
            snapshots["input_metadata.parquet"] = sha256_file(root / "input_metadata.parquet")
        identity = {"source_sha256": requested_sha, "source": str(Path(source).resolve()),
                    "metadata": str(Path(metadata).resolve()) if metadata else None,
                    "snapshots": snapshots, "created_at_utc": utc_now()}
        write_json(identity_path, identity)
    source_frame = pl.read_parquet(root / "input_source.parquet")
    if source_frame["flickr_photo_id"].null_count() or source_frame["flickr_photo_id"].n_unique() != len(source_frame):
        raise ValueError("source requires exactly one non-null row per Flickr photo ID")
    if not len(source_frame) or any(str(c).startswith(PREFIX) for c in source_frame.columns):
        raise ValueError("input must be nonempty source data without reserved bioclip_ columns")
    rows = source_frame.to_dicts()
    if "input_metadata.parquet" in identity["snapshots"]:
        metadata_frame = pl.read_parquet(root / "input_metadata.parquet")
        if metadata_frame["flickr_photo_id"].n_unique() != len(metadata_frame):
            raise ValueError("metadata IDs must be unique")
        metadata_rows = {str(r["flickr_photo_id"]): r for r in metadata_frame.to_dicts()}
        for row in rows:
            extra = metadata_rows.get(str(row["flickr_photo_id"]))
            if extra is None or extra.get("source_record_hash") != row.get("source_record_hash"):
                raise ValueError("metadata must join 1:1 to identical source records")
            if any(k.startswith(PREFIX) for k in extra):
                raise ValueError("metadata contains reserved bioclip_ columns")
            row.update(extra)
    return rows, identity


def flatten(value):
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value


def excel_chunks(value, limit=32767):
    """Excel's limit counts UTF-16 code units; preserve every code point."""
    if not isinstance(value, str):
        return [value]
    parts, current, size = [], [], 0
    for character in value:
        units = 2 if ord(character) > 0xFFFF else 1
        if size + units > limit:
            parts.append("".join(current))
            current, size = [], 0
        current.append(character)
        size += units
    return parts + ["".join(current)]


def input_geolocation_check(row):
    try:
        latitude, longitude = float(row.get("latitude")), float(row.get("longitude"))
    except (TypeError, ValueError):
        return "missing_or_invalid_coordinates"
    if not math.isfinite(latitude) or not math.isfinite(longitude) or not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return "missing_or_invalid_coordinates"
    if latitude == 0 and longitude == 0:
        return "zero_zero_requires_review"
    return "present_unverified"


def review_sample(rows, count=200, seed="held-out-review-v1", exclude_ids=()):
    excluded = set(exclude_ids)
    scored = [r for r in rows if r["bioclip_status"] == "scored" and str(r["flickr_photo_id"]) not in excluded]
    ordered = sorted(scored, key=lambda r: hashlib.sha256((seed + str(r["flickr_photo_id"])).encode()).hexdigest())
    random = ordered[:count // 2]
    selected = {str(r["flickr_photo_id"]) for r in random}
    flags = ("artwork_detected", "specimen_detected", "tattoo_detected", "captive_detected", "museum_detected")
    def risk(row):
        geo_risk = input_geolocation_check(row) != "present_unverified"
        context_risk = row["bioclip_context_top1"] != "ordinary_organism_photo"
        metadata_risk = any(row.get(flag) is True for flag in flags)
        margin = row["bioclip_subject_margin"]
        return (-int(geo_risk), -int(context_risk) - int(metadata_risk), margin, str(row["flickr_photo_id"]))
    targeted = sorted((r for r in ordered if str(r["flickr_photo_id"]) not in selected), key=risk)[:count - len(random)]
    return [dict(r, bioclip_sample_stratum=stratum, manual_subject="", manual_context="", manual_notes="")
            for stratum, group in (("random", random), ("targeted_risk", targeted)) for r in group]


def export_workbook(path, rows, sample):
    import xlsxwriter
    columns = list(dict.fromkeys(k for row in rows for k in row))
    widths = {k: max(len(excel_chunks(flatten(r.get(k)))) for r in rows) for k in columns}
    expanded = [(k, part, k if part == 0 else f"{k}__part_{part + 1}") for k in columns for part in range(widths[k])]
    groups = [("All AI suggestions", rows)]
    subjects = sorted({r.get("bioclip_subject_top1") for r in rows if r["bioclip_status"] == "scored"})
    groups += [(s[:31], [r for r in rows if r.get("bioclip_subject_top1") == s]) for s in subjects]
    failures = [r for r in rows if r["bioclip_status"] != "scored"]
    if failures:
        groups.append(("Unscored images", failures))
    temporary = Path(path).with_suffix(".tmp.xlsx")
    with xlsxwriter.Workbook(temporary, {"constant_memory": True, "strings_to_formulas": False,
                                        "strings_to_urls": False}) as book:
        header = book.add_format({"bold": True, "bg_color": "#dce8ef"})
        for name, group in groups:
            sheet = book.add_worksheet(name)
            sheet.write_row(0, 0, [c for _, _, c in expanded], header)
            for index, row in enumerate(group, 1):
                cells = {k: excel_chunks(flatten(row.get(k))) for k in columns}
                sheet.write_row(index, 0, [cells[k][p] if p < len(cells[k]) else None for k, p, _ in expanded])
            sheet.freeze_panes(1, 1)
            sheet.autofilter(0, 0, len(group), len(expanded) - 1)
            sheet.set_column(0, len(expanded) - 1, 18)
        if sample:
            sheet = book.add_worksheet("Review sample")
            keys = ["flickr_photo_id", "photo_page_url", "bioclip_sample_stratum", "bioclip_subject_top1", "bioclip_context_top1", "manual_subject", "manual_context", "manual_notes"]
            sheet.write_row(0, 0, keys, header)
            for index, row in enumerate(sample, 1):
                sheet.write_row(index, 0, [row.get(k) for k in keys])
            sheet.freeze_panes(1, 1)
            sheet.set_column(0, len(keys) - 1, 24)
    temporary.replace(path)


def export_results(rows, predictions, media, result_dir, keys, manifest, *, final=False):
    combined = []
    for row in rows:
        photo_id = str(row["flickr_photo_id"])
        output = dict(row, **predictions.get(photo_id, {"bioclip_status": "pending"}))
        output["bioclip_input_geolocation_check"] = input_geolocation_check(row)
        output["bioclip_scoring_fingerprint"] = manifest["scoring_fingerprint"]
        if photo_id in media:
            item = media[photo_id]
            for key in ("selected_url", "rendition", "pixel_sha256", "byte_sha256", "width", "height"):
                output[f"bioclip_media_{key}"] = item.get(key)
        combined.append({k: flatten(v) for k, v in output.items()})
    frame = pl.DataFrame(combined, infer_schema_length=None)
    for name, writer in (("image_suggestions.parquet", "write_parquet"), ("image_suggestions.csv", "write_csv")):
        temporary = result_dir / (name + ".tmp")
        getattr(frame, writer)(temporary)
        temporary.replace(result_dir / name)
    counts = dict(frame.group_by("bioclip_status").len().iter_rows())
    progress = dict(manifest, updated_at_utc=utc_now(), counts=counts, total=len(rows),
                    complete=counts.get("pending", 0) == 0, artifacts_finalized=False)
    write_json(result_dir / "progress.json", progress)
    print(json.dumps({"updated_at_utc": progress["updated_at_utc"], "counts": counts, "total": len(rows)}), flush=True)
    if final:
        scores = [{"flickr_photo_id": str(row["flickr_photo_id"]), "head": head, "label": label,
                   "cosine_similarity": row[f"bioclip_{head}_cosine__{label}"],
                   "scoring_fingerprint": manifest["scoring_fingerprint"]}
                  for row in combined if row["bioclip_status"] == "scored" for head, label in keys]
        pl.DataFrame(scores).write_parquet(result_dir / "all_label_scores.parquet")
        development_ids = [str(r["flickr_photo_id"]) for r in sorted(
            rows, key=lambda r: hashlib.sha256(("42" + str(r["flickr_photo_id"])).encode()).hexdigest())[:32]]
        # The smoke images informed prompt design, so keep them out of the review audit.
        # Tiny synthetic datasets still need an exportable sample for local checks.
        excluded = development_ids if len(rows) > 232 else []
        sample = review_sample(combined, exclude_ids=excluded)
        pl.DataFrame(sample, infer_schema_length=None).write_csv(result_dir / "review_sample_200.csv")
        write_json(result_dir / "review_sample_design.json", {
            "random_count": sum(r["bioclip_sample_stratum"] == "random" for r in sample),
            "targeted_count": sum(r["bioclip_sample_stratum"] == "targeted_risk" for r in sample),
            "excluded_prompt_development_photo_ids": excluded,
            "eligible_scored_population": sum(r["bioclip_status"] == "scored" and str(r["flickr_photo_id"]) not in excluded for r in combined),
            "interpretation": "Only the random stratum estimates performance in the eligible population; targeted rows find failures. No accuracy measured yet."
        })
        export_workbook(result_dir / "image_suggestions.xlsx", combined, sample)
        progress["artifacts_finalized"] = True
        progress["artifact_sha256"] = {p.name: sha256_file(p) for p in result_dir.iterdir() if p.suffix in (".parquet", ".csv", ".xlsx", ".npy")}
        write_json(result_dir / "progress.json", progress)
    return progress


def attest(scorer):
    scorer.ensure_model_attestation()
    if scorer.effective_image_resize_mode != "longest":
        raise ValueError("full-frame longest-side preprocessing required")
    identity = {"model": scorer.model_id, "revision": scorer.model_revision,
                "weights_sha256": scorer.model_weights_sha256,
                "open_clip_version": scorer.open_clip_version,
                "open_clip_config_sha256": scorer.open_clip_config_sha256}
    if any(not value for value in identity.values()) or not scorer.preprocessing_fingerprint:
        raise ValueError("incomplete model/preprocessing attestation")
    scorer.pin_reference_model_identity(
        model_weights_sha256=scorer.model_weights_sha256,
        open_clip_version=scorer.open_clip_version,
        open_clip_config_sha256=scorer.open_clip_config_sha256,
        preprocessing_fingerprint=scorer.preprocessing_fingerprint,
        image_resize_mode="longest",
    )
    return dict(identity, model_fingerprint=canonical_semantic_fingerprint(identity),
                preprocessing_fingerprint=scorer.preprocessing_fingerprint,
                preprocessing_config=scorer.preprocessing_config,
                media_preparation=PREPARE_VERSION, image_resize_mode="longest")


def run_inference(rows, state, downloader, scorer, config, result_dir, attestation,
                  manifest, *, batch_size=8, download_workers=4, smoke_size=32, retry_failures=False):
    keys, prototypes, prompts = text_prototypes(scorer, config)
    write_json(result_dir / "prompts.json", dict(config=config, prompts=prompts, keys=keys))
    np.save(result_dir / "text_prototypes.npy", prototypes)
    scoring_key = manifest["scoring_fingerprint"]
    predictions = state.predictions(scoring_key)
    media = state.media()
    if retry_failures:
        predictions = {k: v for k, v in predictions.items() if v.get("bioclip_status") == "scored"}
    ordered = sorted(rows, key=lambda r: hashlib.sha256(("42" + str(r["flickr_photo_id"])).encode()).hexdigest())
    batch_limit = batch_size
    counters = {"encoded_images": 0, "embedding_cache_hits": 0, "oom_retries": 0}
    started = time.monotonic()
    last_checkpoint = 0

    def checkpoint(final=False):
        state.commit()
        current = dict(manifest, counters=counters, elapsed_seconds=time.monotonic() - started,
                       worker_cache_metrics=scorer.cache_metrics, memory_metrics=scorer.memory_metrics,
                       current_batch_size=batch_limit)
        return export_results(rows, predictions, media, result_dir, keys, current, final=final)

    def process_batch(batch):
        nonlocal batch_limit
        needed = {}
        vectors = {}
        for photo_id, item in batch:
            cache_key = embedding_key(item, attestation)
            vector = state.embedding(cache_key)
            if vector is None:
                needed.setdefault(cache_key, (photo_id, item))
            else:
                vectors[cache_key] = vector
                counters["embedding_cache_hits"] += 1
        remaining = list(needed.items())
        while remaining:
            current = remaining[:batch_limit]
            try:
                raw = scorer.embed_image_paths([Path(item["canonical_path"]) for _, (_, item) in current])
            except Exception as error:
                if "memory" in str(error).lower() and batch_limit > 1:
                    batch_limit = max(1, batch_limit // 2)
                    counters["oom_retries"] += 1
                    print(f"Memory pressure: reducing batch to {batch_limit}", flush=True)
                    continue
                checkpoint()
                raise
            vectors_matrix = np.asarray(raw, dtype=np.float32)
            norms = np.linalg.norm(vectors_matrix, axis=1)
            if (vectors_matrix.shape != (len(current), prototypes.shape[1])
                    or not np.isfinite(vectors_matrix).all() or not np.allclose(norms, 1, atol=0.01)):
                raise ValueError("invalid image embedding shape, norm or values")
            expected = [item["pixel_sha256"] for _, (_, item) in current]
            if scorer.last_image_content_hashes != expected:
                raise ValueError("worker decoded content differs from downloaded canonical media")
            for (cache_key, _), vector in zip(current, vectors_matrix, strict=True):
                state.save_embedding(cache_key, vector)
                vectors[cache_key] = vector
                counters["encoded_images"] += 1
            remaining = remaining[len(current):]
        for photo_id, item in batch:
            prediction = score_vector(vectors[embedding_key(item, attestation)], keys, prototypes)
            prediction["bioclip_scored_at_utc"] = utc_now()
            state.save_prediction(photo_id, scoring_key, prediction)
            predictions[photo_id] = prediction
        state.commit()

    checkpoint()
    for phase_name, phase_rows in (("smoke", ordered[:smoke_size]), ("remaining", ordered[smoke_size:])):
        phase_started = time.monotonic()
        pending = [r for r in phase_rows if str(r["flickr_photo_id"]) not in predictions]
        batch = []
        with ThreadPoolExecutor(max_workers=download_workers) as pool:
            for photo_id, item in pool.map(downloader.download, pending, buffersize=32):
                state.save_media(photo_id, item)
                media[photo_id] = item
                if item["status"] != "ready":
                    prediction = {"bioclip_status": "media_failure", "bioclip_error": json.dumps(item["errors"]),
                                  "bioclip_wild_status": "unresolved"}
                    predictions[photo_id] = prediction
                    state.save_prediction(photo_id, scoring_key, prediction)
                else:
                    batch.append((photo_id, item))
                if len(batch) >= batch_limit:
                    process_batch(batch)
                    batch = []
                if len(predictions) - last_checkpoint >= 100:
                    checkpoint()
                    last_checkpoint = len(predictions)
            if batch:
                process_batch(batch)
        checkpoint()
        if phase_name == "smoke":
            elapsed = time.monotonic() - phase_started
            scored = sum(predictions[str(r["flickr_photo_id"])]["bioclip_status"] == "scored" for r in phase_rows)
            smoke = {"rows": len(phase_rows), "scored": scored, "new_rows_attempted": len(pending),
                     "elapsed_seconds": elapsed, "memory_metrics": scorer.memory_metrics,
                     "photo_ids": [str(r["flickr_photo_id"]) for r in phase_rows],
                     "checks": "finite normalized embeddings; decoded-content equality; pinned model; full frame"}
            if pending:
                smoke["seconds_per_attempt"] = elapsed / len(pending)
                smoke["estimated_remaining_seconds"] = smoke["seconds_per_attempt"] * (len(rows) - len(predictions))
            write_json(result_dir / "smoke_test.json", smoke)
            print(json.dumps({"smoke_test": smoke}), flush=True)
            if scored == 0:
                raise RuntimeError("smoke test produced no usable embeddings; stopping")
    return checkpoint(final=True)


@contextmanager
def run_lock(root):
    with (root / "runner.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another runner owns this output directory") from error
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--discovery-state", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-home", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda", "auto"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.download_workers < 1:
        parser.error("worker and batch counts must be positive")
    config = load_labels(args.labels)
    args.output.mkdir(parents=True, exist_ok=True)
    with run_lock(args.output):
        rows, identity = snapshot_inputs(args.source, args.metadata, args.output)
        state = RunState(args.output / "run_state.sqlite")
        urls = supplied_urls(args.discovery_state)
        downloader = MediaDownloader(args.output / "media", urls, state.media(), retry_failures=args.retry_failures)
        model = ModelConfig("bioclip25-custom-labels", "BioCLIP 2.5 custom labels", "evidence", "frozen",
                            "custom labels", MODEL, REVISION, "open_clip_torch", "3.3.0", REVISION)
        runtime = BioClipRuntime(model, args.model_home, args.model_python, "3.3.0", True)
        with PersistentBioClipScorer(runtime=runtime, hf_cache_dir=args.hf_cache, device=args.device,
                                     image_resize_mode="longest", preprocess_workers=2) as scorer:
            attestation = attest(scorer)
            scoring_key = canonical_semantic_fingerprint({"labels": config, "attestation": attestation,
                                                         "scoring": "normalized-template-mean-cosine-v1"})
            result_dir = args.output / "results" / scoring_key.removeprefix("sha256:")
            result_dir.mkdir(parents=True, exist_ok=True)
            manifest = {"schema": "biominer-flickr-custom-labels-v1", "scoring_fingerprint": scoring_key,
                        "input": identity, "attestation": attestation, "labels": config,
                        "score_semantics": "cosine similarities; not calibrated probabilities",
                        "maturity": "AI suggestions only; human review and wild status unresolved",
                        "started_at_utc": utc_now(), "pid": os.getpid(), "device": args.device,
                        "batch_size": args.batch_size, "download_workers": args.download_workers,
                        "output_directory": str(result_dir.resolve())}
            write_json(result_dir / "run_manifest.json", manifest)
            write_json(args.output / "current_results.json", {"directory": str(result_dir.resolve())})
            try:
                result = run_inference(rows, state, downloader, scorer, config, result_dir, attestation, manifest,
                                       batch_size=args.batch_size, download_workers=args.download_workers,
                                       retry_failures=args.retry_failures)
                print(json.dumps({"finished": result}), flush=True)
            except BaseException as error:
                state.commit()
                write_json(result_dir / "runner_error.json", {"error": repr(error), "at_utc": utc_now()})
                raise
            finally:
                state.connection.close()


if __name__ == "__main__":
    main()
