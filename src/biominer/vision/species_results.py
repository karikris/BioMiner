"""Local species model evidence artifacts; historical review remains separate."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

SMOKE_POLICY = "deterministic-round-robin-selection-reason-and-collection-v1"


def collection_names(row):
    value = row.get("source_collections") or []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    if not isinstance(value, list):
        value = [str(value)]
    return tuple(sorted({str(name) for name in value}))


def species_smoke_order(rows):
    """Round robin deterministic strata, then process the complete selected union."""
    strata = defaultdict(lambda: defaultdict(list))
    for row in rows:
        strata[str(row.get("selection_reason") or "unknown")][
            collection_names(row)
        ].append(row)
    groups = {}
    for reason, collections in strata.items():
        groups[reason] = deque(
            deque(
                sorted(
                    collections[name],
                    key=lambda row: hashlib.sha256(
                        ("42" + str(row["flickr_photo_id"])).encode()
                    ).hexdigest(),
                )
            )
            for name in sorted(collections)
        )
    ordered = []
    while groups:
        for key in sorted(groups):
            collection = groups[key].popleft()
            ordered.append(collection.popleft())
            if collection:
                groups[key].append(collection)
        groups = {key: group for key, group in groups.items() if group}
    return ordered


def historical_target_comparison(rows, target=None):
    historical = [
        row
        for row in rows
        if target and row.get("human_label") == f"target_scientific_name:{target}"
    ]
    scored = [row for row in historical if row["bioclip_status"] == "scored"]
    top1 = sum(row.get("bioclip_species_top1") == target for row in scored)
    top5 = sum(
        target in [row.get(f"bioclip_species_top{rank}") for rank in range(1, 6)]
        for row in scored
    )
    return {
        "target_scientific_name": target,
        "human_label": f"target_scientific_name:{target}" if target else None,
        "status": "available" if target else "unavailable_no_target_configured",
        "historically_labeled_rows": len(historical),
        "scored_labeled_rows": len(scored),
        "unscored_labeled_rows": len(historical) - len(scored),
        "top1_matches": top1,
        "top5_matches": top5,
        "top1_match_fraction_of_scored": top1 / len(scored) if scored else None,
        "top5_match_fraction_of_scored": top5 / len(scored) if scored else None,
        "interpretation": (
            "Agreement with historical human labels; not ground truth or an accuracy estimate."
            if target
            else "No historical comparison target configured; comparison metrics unavailable."
        ),
    }


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_long_scores(rows, keys, path, scoring_fingerprint):
    # Bound the expanded species table independently of selected image count.
    schema = pa.schema(
        [
            ("item_id", pa.string()),
            ("flickr_photo_id", pa.string()),
            ("head", pa.string()),
            ("label", pa.string()),
            ("cosine_similarity", pa.float64()),
            ("scoring_fingerprint", pa.string()),
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        batch = []
        for row in rows:
            if row["bioclip_status"] != "scored":
                continue
            for head, label in keys:
                batch.append(
                    {
                        "item_id": str(row["item_id"])
                        if row.get("item_id") is not None
                        else None,
                        "flickr_photo_id": str(row["flickr_photo_id"]),
                        "head": head,
                        "label": label,
                        "cosine_similarity": row[f"bioclip_{head}_cosine__{label}"],
                        "scoring_fingerprint": scoring_fingerprint,
                    }
                )
                if len(batch) >= 8192:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    temporary.replace(path)


def _local_uri(value):
    if not value:
        return None
    path = Path(value)
    return path.as_uri() if path.is_absolute() and path.is_file() else None


def _write_report(rows, path, progress):
    temporary = path.with_suffix(path.suffix + ".tmp")
    comparison = progress["historical_target_comparison"]
    target = comparison["target_scientific_name"]
    with temporary.open("w") as handle:
        handle.write("""<!doctype html><html lang="en"><meta charset="utf-8">
<title>BioCLIP species suggestions</title><style>
body{font:16px system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;background:#f7f8f6;color:#18231e}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1rem}
article{background:white;padding:1rem;border:1px solid #cbd4ce;border-radius:8px;overflow-wrap:anywhere}
img{width:100%;height:210px;object-fit:contain;background:#f0f1ee}li{padding:.15rem 0}small{color:#4d5e53}
</style><h1>BioCLIP species suggestions</h1>
<p>Frozen BioCLIP model evidence from full-frame pixels. Cosine similarities and margins are not probabilities.
Historical human labels are retained separately; they are not ground truth. Wild status remains unresolved.</p>""")
        handle.write(
            f"<p>Total: {progress['total']}. Scored: {progress['counts']['scored']}. "
            f"Media failures: {progress['counts']['media_failure']}. Pending: {progress['counts']['pending']}.</p>"
        )
        if target:
            handle.write(
                f"<h2>{escape(target)}: agreement with historical human labels</h2><p>"
                f"{comparison['historically_labeled_rows']} historically labeled images; "
                f"{comparison['scored_labeled_rows']} scored and {comparison['unscored_labeled_rows']} unscored. "
                f"Top 1 matches: {comparison['top1_matches']}; top 5 matches: {comparison['top5_matches']}. "
                "These are historical label comparisons, not a measured biological accuracy.</p>"
            )
        else:
            handle.write(
                f"<h2>Historical human label comparison</h2><p>{escape(comparison['interpretation'])}</p>"
            )
        handle.write(
            '<p><a href="image_suggestions.parquet">Image suggestions (Parquet)</a> · '
            '<a href="all_label_scores.parquet">All species cosines (Parquet)</a> · '
            '<a href="progress.json">Run metadata and counts</a></p><div class="grid">'
        )
        for row in rows:
            photo_id = escape(str(row["flickr_photo_id"]))
            handle.write(f"<article><h2>Photo {photo_id}</h2>")
            original_uri = _local_uri(row.get("bioclip_media_original_path"))
            canonical_uri = _local_uri(row.get("bioclip_media_canonical_path"))
            # An unverified/missing source never becomes a supposedly verified thumbnail.
            if original_uri:
                thumbnail_uri = canonical_uri or original_uri
                handle.write(
                    f'<a href="{escape(original_uri, quote=True)}"><img loading="lazy" '
                    f'src="{escape(thumbnail_uri, quote=True)}" alt="Cached photo {photo_id}"></a>'
                )
            handle.write(
                f"<p>Status: {escape(str(row['bioclip_status']))}<br>"
                f"Selection: {escape(str(row.get('selection_reason') or 'unknown'))}<br>"
                f"Historical human label: {escape(str(row.get('human_label') or 'unreviewed'))}<br>"
                f"Historical group: {escape(str(row.get('human_group') or 'unreviewed'))}</p>"
            )
            if row["bioclip_status"] == "scored":
                handle.write("<ol>")
                for rank in range(1, 6):
                    name = row.get(f"bioclip_species_top{rank}")
                    cosine = row.get(f"bioclip_species_top{rank}_cosine")
                    if name is not None:
                        handle.write(
                            f"<li>{escape(name)} <small>cosine {cosine:.5f}</small></li>"
                        )
                handle.write(
                    f"</ol><p>Top 1–2 cosine margin: {row['bioclip_species_margin']:.5f}</p>"
                )
            elif row.get("bioclip_error"):
                handle.write(f"<p>{escape(str(row['bioclip_error']))}</p>")
            handle.write("</article>")
        handle.write("</div></html>")
    temporary.replace(path)


def export_species_results(rows, result_dir, keys, manifest, *, final=False):
    score_types = {f"bioclip_species_top{rank}": pl.String for rank in range(1, 6)}
    score_types.update(
        {f"bioclip_species_top{rank}_cosine": pl.Float64 for rank in range(1, 6)}
    )
    score_types["bioclip_species_margin"] = pl.Float64
    score_types.update(
        {f"bioclip_species_cosine__{name}": pl.Float64 for _, name in keys}
    )
    for row in rows:
        for rank in range(1, 6):
            row.setdefault(f"bioclip_species_top{rank}", None)
            row.setdefault(f"bioclip_species_top{rank}_cosine", None)
        row.setdefault("bioclip_species_margin", None)
        row.setdefault("bioclip_wild_status", "unresolved")
    frame = pl.DataFrame(rows, infer_schema_length=None, schema_overrides=score_types)
    missing_scores = [
        pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in score_types.items()
        if name not in frame.columns
    ]
    if missing_scores:
        frame = frame.with_columns(missing_scores)
    temporary = result_dir / "image_suggestions.parquet.tmp"
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(result_dir / "image_suggestions.parquet")
    actual = Counter(row["bioclip_status"] for row in rows)
    counts = dict.fromkeys(("scored", "media_failure", "pending"), 0)
    counts.update(actual)
    progress = dict(
        manifest,
        schema="biominer-bioclip-species-evidence-v1",
        updated_at_utc=datetime.now(UTC).isoformat(),
        counts=counts,
        total=len(rows),
        complete=counts["pending"] == 0,
        artifacts_finalized=False,
        score_semantics="Cosine similarities and margins; not probabilities.",
        selection_counts=dict(
            Counter(row.get("selection_reason") or "unknown" for row in rows)
        ),
        source_collection_counts=dict(
            Counter(name for row in rows for name in collection_names(row))
        ),
        historical_target_comparison=historical_target_comparison(
            rows, target=manifest.get("labels", {}).get("historical_comparison_target")
        ),
    )
    _write_json(result_dir / "progress.json", progress)
    print(
        json.dumps(
            {
                "updated_at_utc": progress["updated_at_utc"],
                "counts": counts,
                "total": len(rows),
            }
        ),
        flush=True,
    )
    if final:
        _write_long_scores(
            rows,
            keys,
            result_dir / "all_label_scores.parquet",
            manifest["scoring_fingerprint"],
        )
        _write_report(rows, result_dir / "report.html", progress)
        progress["artifacts_finalized"] = True
        progress["artifact_sha256"] = {}
        for name in (
            "image_suggestions.parquet",
            "all_label_scores.parquet",
            "report.html",
            "text_prototypes.npy",
            "prompts.json",
        ):
            path = result_dir / name
            if path.is_file():
                with path.open("rb") as handle:
                    progress["artifact_sha256"][name] = hashlib.file_digest(
                        handle, "sha256"
                    ).hexdigest()
        _write_json(result_dir / "progress.json", progress)
    return progress
