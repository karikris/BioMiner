from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.assertions import assertion_table, build_assertion


BIOLOGICAL_EXTRACTION_VERSION = "biominer-gbif-biological-candidates/v1"
BIOLOGICAL_RULE_VERSION = "controlled-english-biological-terms/v1.0.0"
TEXT_FIELDS = ("occurrenceRemarks", "dynamicProperties")
CANDIDATE_SCHEMA = pa.schema(
    [
        ("candidate_version", pa.string()),
        ("candidate_id", pa.string()),
        ("source_snapshot_id", pa.string()),
        ("source_row_id", pa.string()),
        ("gbifID", pa.string()),
        ("affected_media_rows", pa.int64()),
        ("target_field", pa.string()),
        ("original_value", pa.string()),
        ("derived_value", pa.string()),
        ("source_text_field", pa.string()),
        ("source_text", pa.string()),
        ("matched_text_spans", pa.list_(pa.string())),
        ("extraction_rules", pa.list_(pa.string())),
        ("rule_version", pa.string()),
        ("language", pa.string()),
        ("confidence", pa.string()),
        ("candidate_status", pa.string()),
        ("candidate_reason", pa.string()),
        ("review_status", pa.string()),
    ]
)
_LIFE_TERMS = {
    "adult": (r"\badults?\b", r"\bimago(?:es)?\b"),
    "larva": (r"\blarvae?\b", r"\blarval\b"),
    "caterpillar": (r"\bcaterpillars?\b",),
    "pupa": (r"\bpupae?\b", r"\bchrysal(?:is|ides)\b"),
    "egg": (r"\beggs?\b", r"\bova\b"),
    "juvenile": (r"\bjuveniles?\b",),
    "nymph": (r"\bnymphs?\b",),
}
_SEX_TERMS = {
    "male": (r"\bmales?\b", r"♂"),
    "female": (r"\bfemales?\b", r"♀"),
    "indeterminate": (r"\bindeterminate sex\b", r"\bsex unknown\b"),
}
_NEGATION = re.compile(r"\b(?:not|no|without|non)\b(?:\W+\w+){0,3}\W*$", re.I)
_CONTEXT_RISK = re.compile(r"\b(?:host|prey|predator|nearby|flower|plant|parasitoid)\b", re.I)


@dataclass(frozen=True, slots=True)
class TextExtraction:
    value: str | None
    spans: tuple[str, ...]
    rules: tuple[str, ...]
    status: str
    reason: str
    confidence: str


@dataclass(frozen=True, slots=True)
class BiologicalCandidateResult:
    output_directory: Path
    candidate_path: Path
    assertion_path: Path
    manifest: dict[str, object]


def extract_controlled_value(text: object | None, *, target: str) -> TextExtraction:
    source = "" if text is None else str(text)
    terms = _LIFE_TERMS if target == "lifeStage" else _SEX_TERMS if target == "sex" else None
    if terms is None:
        raise ValueError(f"unsupported biological extraction target: {target}")
    positives: list[tuple[str, str, str]] = []
    negated: list[str] = []
    for value, patterns in terms.items():
        for index, pattern in enumerate(patterns):
            for match in re.finditer(pattern, source, re.I):
                prefix = source[max(0, match.start() - 40):match.start()]
                if _NEGATION.search(prefix):
                    negated.append(match.group(0))
                    continue
                positives.append((value, match.group(0), f"{target}:{value}:{index + 1}"))
    if not positives:
        return TextExtraction(
            None, tuple(negated), (), "NEGATED_ONLY" if negated else "NO_MATCH",
            "all_matches_negated" if negated else "no_controlled_term", "NONE"
        )
    values = sorted({value for value, _, _ in positives})
    derived = values[0] if len(values) == 1 else "mixed"
    context_risk = bool(_CONTEXT_RISK.search(source))
    return TextExtraction(
        derived,
        tuple(span for _, span, _ in positives),
        tuple(rule for _, _, rule in positives),
        "CONFLICT" if len(values) > 1 else "CANDIDATE",
        "multiple_controlled_values" if len(values) > 1 else "context_risk" if context_risk else "controlled_term_match",
        "LOW" if context_risk or len(values) > 1 else "MEDIUM",
    )


def publish_biological_candidates(
    *,
    v3_parquet: str | Path,
    output_directory: str | Path,
    source_snapshot_id: str,
    expected_media_rows: int,
    expected_occurrences: int,
    code_commit: str,
    memory_limit: str = "4GB",
    threads: int = 4,
) -> BiologicalCandidateResult:
    source = Path(v3_parquet).resolve()
    destination = Path(output_directory).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    parquet = pq.ParquetFile(source)
    required = {"gbifID", "lifeStage", "sex", *TEXT_FIELDS}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"v3 lacks biological extraction fields: {sorted(missing)}")
    if parquet.metadata.num_rows != expected_media_rows:
        raise ValueError("biological candidate source row count mismatch")
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    rows = _candidate_source_rows(source, memory_limit=memory_limit, threads=threads)
    candidates = []
    assertions = []
    occurrence_ids = set()
    candidate_occurrences: dict[str, set[str]] = {"lifeStage": set(), "sex": set()}
    candidate_media: dict[str, int] = {"lifeStage": 0, "sex": 0}
    for row in rows:
        occurrence_ids.add(str(row["gbifID"]))
        for target, original in (("lifeStage", row["lifeStage"]), ("sex", row["sex"])):
            if _trimmed(original) is not None:
                continue
            for field in TEXT_FIELDS:
                extraction = extract_controlled_value(row[field], target=target)
                if extraction.status == "NO_MATCH":
                    continue
                source_row_id = _source_row_id(source_snapshot_id, str(row["gbifID"]))
                identity = "|".join((source_snapshot_id, str(row["gbifID"]), target, field, extraction.value or "", BIOLOGICAL_RULE_VERSION))
                candidate = {
                    "candidate_version": BIOLOGICAL_EXTRACTION_VERSION,
                    "candidate_id": "sha256:" + hashlib.sha256(identity.encode()).hexdigest(),
                    "source_snapshot_id": source_snapshot_id,
                    "source_row_id": source_row_id,
                    "gbifID": str(row["gbifID"]),
                    "affected_media_rows": int(row["affected_media_rows"]),
                    "target_field": f"derived_{target}",
                    "original_value": None,
                    "derived_value": extraction.value,
                    "source_text_field": field,
                    "source_text": row[field],
                    "matched_text_spans": list(extraction.spans),
                    "extraction_rules": list(extraction.rules),
                    "rule_version": BIOLOGICAL_RULE_VERSION,
                    "language": "und",
                    "confidence": extraction.confidence,
                    "candidate_status": extraction.status,
                    "candidate_reason": extraction.reason,
                    "review_status": "PENDING",
                }
                candidates.append(candidate)
                if extraction.value is None:
                    continue
                if str(row["gbifID"]) not in candidate_occurrences[target]:
                    candidate_media[target] += int(row["affected_media_rows"])
                candidate_occurrences[target].add(str(row["gbifID"]))
                assertions.append(build_assertion(
                    source_snapshot_version=source_snapshot_id,
                    source_row_id=source_row_id,
                    gbif_id=str(row["gbifID"]),
                    target_field=f"derived_{target}",
                    original_value=None,
                    derived_value=extraction.value,
                    evidence_source=field,
                    source_url_or_record_identifier=f"gbifID:{row['gbifID']}",
                    retrieval_timestamp=generated_at,
                    derivation_method="controlled_text_candidate",
                    derivation_rule_version=BIOLOGICAL_RULE_VERSION,
                    confidence_class="CONTROLLED_TEXT_EXTRACTION",
                    validation_status="UNKNOWN",
                    conflict_status="CONFLICT" if extraction.status == "CONFLICT" else "PASS",
                    reviewer_status="PENDING",
                ))
    # More than one source field may produce the same semantic assertion. Keep one
    # assertion deterministically while retaining every source-field candidate.
    unique_assertions = {item.assertion_id: item for item in assertions}
    assertions = [unique_assertions[key] for key in sorted(unique_assertions)]
    candidates.sort(key=lambda row: (str(row["gbifID"]), str(row["target_field"]), str(row["source_text_field"])))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    candidate_path = staging / "biological_candidates.parquet"
    assertion_path = staging / "biological_assertions.parquet"
    try:
        pq.write_table(pa.Table.from_pylist(candidates, schema=CANDIDATE_SCHEMA), candidate_path, compression="zstd")
        pq.write_table(assertion_table(assertions), assertion_path, compression="zstd")
        source_occurrences = _distinct_occurrences(source, memory_limit, threads)
        counts = {
            "source_media_rows": parquet.metadata.num_rows,
            "source_occurrences": source_occurrences,
            "text_prefilter_occurrences": len(occurrence_ids),
            "candidate_rows": len(candidates),
            "assertion_rows": len(assertions),
            "life_stage_candidate_occurrences": len(candidate_occurrences["lifeStage"]),
            "life_stage_candidate_media_rows": candidate_media["lifeStage"],
            "sex_candidate_occurrences": len(candidate_occurrences["sex"]),
            "sex_candidate_media_rows": candidate_media["sex"],
            "negated_only_rows": sum(row["candidate_status"] == "NEGATED_ONLY" for row in candidates),
            "conflicting_candidate_rows": sum(row["candidate_status"] == "CONFLICT" for row in candidates),
        }
        validation = {
            "source_media_rows_match": counts["source_media_rows"] == expected_media_rows,
            "source_occurrences_match": source_occurrences == expected_occurrences,
            "all_candidates_require_review": all(row["review_status"] == "PENDING" for row in candidates),
            "original_values_unchanged": all(row["original_value"] is None for row in candidates),
            "negated_terms_not_asserted": all(row["derived_value"] is None for row in candidates if row["candidate_status"] == "NEGATED_ONLY"),
            "candidate_schema_matches": pq.ParquetFile(candidate_path).schema_arrow.equals(CANDIDATE_SCHEMA),
        }
        if not all(validation.values()):
            raise ValueError(f"biological candidate validation failed: {validation}")
        artifacts = [_artifact(candidate_path), _artifact(assertion_path)]
        manifest = {
            "schema_version": BIOLOGICAL_EXTRACTION_VERSION,
            "rule_version": BIOLOGICAL_RULE_VERSION,
            "generated_at": generated_at,
            "code_commit": code_commit,
            "source_snapshot_id": source_snapshot_id,
            "input": str(source),
            "searched_text_fields": list(TEXT_FIELDS),
            "counts": counts,
            "validation": validation,
            "artifacts": artifacts,
            "policy": {"source_fields_unchanged": True, "candidate_only": True, "human_review_required": True, "language": "und"},
            "network_requests": 0,
            "manifest_policy": {"written_last": True},
        }
        _write_json(staging / "manifest.json", manifest)
        for artifact in artifacts: _verify(staging, artifact)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return BiologicalCandidateResult(destination, destination / candidate_path.name, destination / assertion_path.name, manifest)


def _candidate_source_rows(source: Path, *, memory_limit: str, threads: int) -> list[dict[str, object]]:
    c = duckdb.connect()
    try:
        c.execute(f"SET threads={threads}"); c.execute(f"SET memory_limit={_literal(memory_limit)}")
        pattern = r"\b(adult|imago|larva|larvae|larval|caterpillar|pupa|pupae|chrysalis|egg|juvenile|nymph|male|female)\b|[♂♀]"
        cursor = c.execute(f"""
          SELECT trim(cast(gbifID AS VARCHAR)) gbifID, count(*)::BIGINT affected_media_rows,
                 min(lifeStage) lifeStage, min(sex) sex,
                 min(occurrenceRemarks) occurrenceRemarks, min(dynamicProperties) dynamicProperties
          FROM read_parquet({_literal(str(source))})
          WHERE (lifeStage IS NULL OR sex IS NULL)
            AND regexp_matches(lower(coalesce(occurrenceRemarks,'') || ' ' || coalesce(dynamicProperties,'')), {_literal(pattern)})
          GROUP BY gbifID ORDER BY gbifID
        """)
        names = [item[0] for item in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
    finally:
        c.close()


def _distinct_occurrences(source: Path, memory_limit: str, threads: int) -> int:
    c=duckdb.connect()
    try:
        c.execute(f"SET threads={threads}"); c.execute(f"SET memory_limit={_literal(memory_limit)}")
        return int(c.execute(f"SELECT count(distinct gbifID) FROM read_parquet({_literal(str(source))})").fetchone()[0])
    finally: c.close()


def _source_row_id(snapshot: str, gbif_id: str) -> str:
    return "sha256:" + hashlib.sha256(f"{snapshot}|occurrence.txt|gbifID={gbif_id}".encode()).hexdigest()


def _trimmed(value: object | None) -> str | None:
    if value is None: return None
    text=str(value).strip(); return text or None


def _literal(value: object) -> str: return "'" + str(value).replace("'", "''") + "'"


def _artifact(path: Path) -> dict[str, object]:
    p=pq.ParquetFile(path); return {"path":path.name,"physical_bytes":path.stat().st_size,"sha256":_sha256(path),"row_count":p.metadata.num_rows,"column_count":len(p.schema_arrow),"row_group_count":p.metadata.num_row_groups}


def _verify(root: Path, artifact: dict[str, object]) -> None:
    if _sha256(root / str(artifact["path"])) != artifact["sha256"]: raise ValueError(f"biological checksum mismatch: {artifact['path']}")


def _write_json(path: Path, value: object) -> None:
    temporary=path.with_suffix(".json.tmp"); temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(temporary,path)


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(16*1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


__all__ = ["BIOLOGICAL_EXTRACTION_VERSION", "BIOLOGICAL_RULE_VERSION", "BiologicalCandidateResult", "TextExtraction", "extract_controlled_value", "publish_biological_candidates"]
