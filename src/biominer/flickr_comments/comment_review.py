from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Iterable

import polars as pl

from biominer.filter.category_model import infer_life_stage_from_text
from biominer.filter.extractor import SCIENTIFIC_NAME_PATTERN
from biominer.flickr_comments.comments_enrichment import fetch_flickr_comments, mine_comment_terms
from biominer.flickr_fetch.metadata_poller import PENDING


FetchComments = Callable[[str], list[dict[str, Any]]]

COMMENT_REVIEW_METRICS = (
    "comment_review_queue_created",
    "comment_calls_used",
    "comments_fetched",
    "species_conflicts_reviewed",
    "species_conflicts_resolved",
    "records_moved_to_gold",
    "records_moved_to_silver",
    "records_kept_in_review_no_geo",
    "missing_geo_requests_created",
    "missing_date_requests_created",
    "comment_derived_terms_created",
    "comment_review_failures",
)

GOLD_SPECIES_CONFIDENCE_THRESHOLD = 0.70
SILVER_SPECIES_CONFIDENCE_THRESHOLD = 0.35
MAX_COMMENT_CALLS_PER_HOUR = 300
TARGET_TERMS = (
    "papilio demoleus",
    "lime butterfly",
    "chequered swallowtail",
    "checkered swallowtail",
    "citrus swallowtail",
)
GENERIC_LEPIDOPTERA_TERMS = ("butterfly", "swallowtail")
HARD_NEGATIVE_CATEGORIES = {"museum_specimen", "artwork", "tattoo", "ai_generated", "other_insect", "not_lepidoptera", "object_or_product", "logo_or_brand", "textile_or_pattern"}


@dataclass(frozen=True)
class CommentReviewResult:
    flickr_photo_id: str
    comments_fetched: bool
    comment_count: int
    species_match_from_comments: bool
    species_name_from_comments: str | None
    common_name_from_comments: str | None
    life_stage_from_comments: str | None
    date_evidence_from_comments: str | None
    geo_evidence_from_comments: str | None
    location_text_from_comments: str | None
    comment_review_decision: str
    comment_review_reason: str
    flickr_text_species_candidate: str | None
    bioclip_species_candidate: str | None
    bioclip_tag_conflict: bool
    comment_species_candidate: str | None
    comment_resolves_conflict: bool


class CommentReviewState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def enqueue_records(self, records: Iterable[dict[str, Any]]) -> int:
        return sum(self.enqueue_record(record) for record in records)

    def enqueue_record(self, record: dict[str, Any]) -> int:
        if _is_duplicate_record(record):
            return 0
        photo_id = str(record.get("flickr_photo_id") or record.get("id") or "")
        if not photo_id:
            return 0
        reasons = comment_review_reasons(record)
        if not reasons:
            return 0
        source_record_id = str(record.get("source_record_id") or photo_id)
        source_record_hash = str(record.get("source_record_hash") or _record_hash(record))
        queue_id = _queue_id(photo_id=photo_id, source_record_hash=source_record_hash)
        with self._connect() as conn:
            if self._already_reviewed(conn, photo_id=photo_id, source_record_hash=source_record_hash):
                return 0
            result = conn.execute(
                """
                INSERT OR IGNORE INTO comment_review_queue (
                    queue_id, flickr_photo_id, source_record_id, source_record_hash,
                    photo_page_url, image_url, reason, status, attempts,
                    record_json, created_at, claimed_at, completed_at, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL, NULL, NULL)
                """,
                (
                    queue_id,
                    photo_id,
                    source_record_id,
                    source_record_hash,
                    _optional_string(record.get("photo_page_url")),
                    _optional_string(record.get("image_url")),
                    json.dumps(reasons, sort_keys=True),
                    PENDING,
                    json.dumps(record, sort_keys=True, default=str),
                    _timestamp(),
                ),
            )
        return int(result.rowcount)

    def pending_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT count(*) FROM comment_review_queue WHERE status = ?", (PENDING,)).fetchone()[0])

    def process_pending(self, *, fetch_comments: FetchComments, max_api_calls: int = MAX_COMMENT_CALLS_PER_HOUR) -> dict[str, int]:
        limit = min(max(0, max_api_calls), MAX_COMMENT_CALLS_PER_HOUR)
        summary = {metric: 0 for metric in COMMENT_REVIEW_METRICS}
        if limit <= 0:
            return summary
        for row in self._claim_pending(limit=limit):
            summary["comment_calls_used"] += 1
            try:
                comments = fetch_comments(str(row["flickr_photo_id"]))
                result = review_comments_for_record(_record_from_row(row), comments)
                terms_created = self.record_review_result(queue_row=row, review=result, comments=comments)
                summary["comments_fetched"] += 1 if result.comments_fetched else 0
                summary["comment_derived_terms_created"] += terms_created
                if bool(result.bioclip_tag_conflict):
                    summary["species_conflicts_reviewed"] += 1
                if result.comment_resolves_conflict:
                    summary["species_conflicts_resolved"] += 1
                if result.comment_review_decision == "move_to_gold":
                    summary["records_moved_to_gold"] += 1
                if result.comment_review_decision == "move_to_silver":
                    summary["records_moved_to_silver"] += 1
                if result.comment_review_decision == "keep_in_review_no_geo":
                    summary["records_kept_in_review_no_geo"] += 1
                if result.comment_review_decision == "request_missing_geo":
                    summary["missing_geo_requests_created"] += 1
                if result.comment_review_decision == "request_missing_date":
                    summary["missing_date_requests_created"] += 1
            except Exception as exc:  # noqa: BLE001 - bounded review records failures.
                summary["comment_review_failures"] += 1
                self._mark_failed(queue_id=str(row["queue_id"]), error=str(exc))
        return summary

    def record_review_result(self, *, queue_row: sqlite3.Row, review: CommentReviewResult, comments: Iterable[dict[str, Any]]) -> int:
        terms_created = 0
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO comment_review_results (
                    flickr_photo_id, source_record_hash, comments_fetched, comment_count,
                    species_match_from_comments, species_name_from_comments,
                    common_name_from_comments, life_stage_from_comments,
                    date_evidence_from_comments, geo_evidence_from_comments,
                    location_text_from_comments, comment_review_decision,
                    comment_review_reason, flickr_text_species_candidate,
                    bioclip_species_candidate, bioclip_tag_conflict,
                    comment_species_candidate, comment_resolves_conflict, reviewed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review.flickr_photo_id,
                    str(queue_row["source_record_hash"]),
                    1 if review.comments_fetched else 0,
                    review.comment_count,
                    1 if review.species_match_from_comments else 0,
                    review.species_name_from_comments,
                    review.common_name_from_comments,
                    review.life_stage_from_comments,
                    review.date_evidence_from_comments,
                    review.geo_evidence_from_comments,
                    review.location_text_from_comments,
                    review.comment_review_decision,
                    review.comment_review_reason,
                    review.flickr_text_species_candidate,
                    review.bioclip_species_candidate,
                    1 if review.bioclip_tag_conflict else 0,
                    review.comment_species_candidate,
                    1 if review.comment_resolves_conflict else 0,
                    _timestamp(),
                ),
            )
            for comment in comments:
                text = _comment_text(comment)
                if not text:
                    continue
                for term in mine_comment_terms(text):
                    result = conn.execute(
                        """
                        INSERT OR IGNORE INTO comment_derived_terms (
                            flickr_photo_id, term, term_kind, evidence_text, created_at
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (review.flickr_photo_id, term.term, term.term_kind, text, _timestamp()),
                    )
                    terms_created += int(result.rowcount)
            if review.comment_review_decision in {"request_missing_geo", "request_missing_date", "request_species_review", "request_life_stage_review"}:
                self._insert_missing_data_request(conn, review=review)
            conn.execute(
                """
                UPDATE comment_review_queue
                SET status = ?, completed_at = ?, error = NULL
                WHERE queue_id = ?
                """,
                ("completed", _timestamp(), str(queue_row["queue_id"])),
            )
        return terms_created

    def apply_decisions_to_records(self, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        with self._connect() as conn:
            reviews = {
                str(row["flickr_photo_id"]): row
                for row in conn.execute("SELECT * FROM comment_review_results").fetchall()
            }
        updated: list[dict[str, Any]] = []
        for record in records:
            photo_id = str(record.get("flickr_photo_id") or "")
            review = reviews.get(photo_id)
            if not review:
                updated.append(record)
                continue
            if str(review["comment_review_decision"]) == "move_to_gold":
                updated.append(
                    {
                        **record,
                        "occurrence_bin": "gold",
                        "triage_bin": "gold",
                        "bin_reason": "comment_review_resolved",
                        "triage_reason": "comment_review_resolved",
                        "comment_review_decision": "move_to_gold",
                        "comment_review_reason": review["comment_review_reason"],
                        "comment_species_candidate": review["comment_species_candidate"],
                    }
                )
            elif str(review["comment_review_decision"]) == "move_to_silver":
                updated.append(
                    {
                        **record,
                        "occurrence_bin": "silver",
                        "triage_bin": "silver",
                        "bin_reason": "comment_review_species_support",
                        "triage_reason": "comment_review_species_support",
                        "comment_review_decision": "move_to_silver",
                        "comment_review_reason": review["comment_review_reason"],
                        "comment_species_candidate": review["comment_species_candidate"],
                    }
                )
            else:
                updated.append(
                    {
                        **record,
                        "comment_review_decision": review["comment_review_decision"],
                        "comment_review_reason": review["comment_review_reason"],
                        "comment_species_candidate": review["comment_species_candidate"],
                    }
                )
        return updated

    def summary(self) -> dict[str, int]:
        with self._connect() as conn:
            return {
                "comment_review_queue_created": int(conn.execute("SELECT count(*) FROM comment_review_queue").fetchone()[0]),
                "comment_calls_used": int(conn.execute("SELECT count(*) FROM comment_review_results WHERE comments_fetched = 1").fetchone()[0]),
                "comments_fetched": int(conn.execute("SELECT coalesce(sum(comment_count), 0) FROM comment_review_results").fetchone()[0]),
                "species_conflicts_reviewed": int(conn.execute("SELECT count(*) FROM comment_review_results WHERE bioclip_tag_conflict = 1").fetchone()[0]),
                "species_conflicts_resolved": int(conn.execute("SELECT count(*) FROM comment_review_results WHERE comment_resolves_conflict = 1").fetchone()[0]),
                "records_moved_to_gold": int(conn.execute("SELECT count(*) FROM comment_review_results WHERE comment_review_decision = 'move_to_gold'").fetchone()[0]),
                "records_moved_to_silver": int(conn.execute("SELECT count(*) FROM comment_review_results WHERE comment_review_decision = 'move_to_silver'").fetchone()[0]),
                "records_kept_in_review_no_geo": int(conn.execute("SELECT count(*) FROM comment_review_results WHERE comment_review_decision = 'keep_in_review_no_geo'").fetchone()[0]),
                "missing_geo_requests_created": int(conn.execute("SELECT count(*) FROM missing_data_requests WHERE request_type = 'missing_geo'").fetchone()[0]),
                "missing_date_requests_created": int(conn.execute("SELECT count(*) FROM missing_data_requests WHERE request_type = 'missing_date'").fetchone()[0]),
                "comment_derived_terms_created": int(conn.execute("SELECT count(*) FROM comment_derived_terms").fetchone()[0]),
                "comment_review_failures": int(conn.execute("SELECT count(*) FROM comment_review_queue WHERE status = 'failed'").fetchone()[0]),
            }

    def _claim_pending(self, *, limit: int) -> list[sqlite3.Row]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT *
                FROM comment_review_queue
                WHERE status = ?
                ORDER BY
                    CASE
                        WHEN reason LIKE '%species_conflict%' THEN 1
                        WHEN reason LIKE '%no_geo%' THEN 2
                        WHEN reason LIKE '%missing_event_date%' THEN 3
                        WHEN reason LIKE '%low_bioclip_score%' THEN 4
                        ELSE 5
                    END,
                    created_at,
                    queue_id
                LIMIT ?
                """,
                (PENDING, limit),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE comment_review_queue SET status = ?, attempts = attempts + 1, claimed_at = ? WHERE queue_id = ?",
                    ("claimed", _timestamp(), row["queue_id"]),
                )
            conn.execute("COMMIT")
        return rows

    def _insert_missing_data_request(self, conn: sqlite3.Connection, *, review: CommentReviewResult) -> None:
        if review.comment_review_decision == "request_missing_geo":
            request_type = "missing_geo"
            evidence_text = review.location_text_from_comments or review.geo_evidence_from_comments
        elif review.comment_review_decision == "request_missing_date":
            request_type = "missing_date"
            evidence_text = review.date_evidence_from_comments
        elif review.comment_review_decision == "request_species_review":
            request_type = "ambiguous_species"
            evidence_text = review.comment_species_candidate or review.comment_review_reason
        elif review.comment_review_decision == "request_life_stage_review":
            request_type = "ambiguous_life_stage"
            evidence_text = review.life_stage_from_comments or review.comment_review_reason
        else:
            return
        conn.execute(
            """
            INSERT OR IGNORE INTO missing_data_requests (
                request_id, flickr_photo_id, request_type, evidence_source,
                evidence_text, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _request_id(review.flickr_photo_id, request_type, evidence_text),
                review.flickr_photo_id,
                request_type,
                "comments",
                evidence_text,
                PENDING,
                _timestamp(),
            ),
        )

    def _mark_failed(self, *, queue_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE comment_review_queue SET status = ?, completed_at = ?, error = ? WHERE queue_id = ?",
                ("failed", _timestamp(), error, queue_id),
            )

    def _already_reviewed(self, conn: sqlite3.Connection, *, photo_id: str, source_record_hash: str) -> bool:
        existing_queue = conn.execute(
            "SELECT 1 FROM comment_review_queue WHERE flickr_photo_id = ? AND source_record_hash = ?",
            (photo_id, source_record_hash),
        ).fetchone()
        existing_result = conn.execute(
            "SELECT 1 FROM comment_review_results WHERE flickr_photo_id = ? AND source_record_hash = ?",
            (photo_id, source_record_hash),
        ).fetchone()
        return bool(existing_queue or existing_result)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS comment_review_queue (
                    queue_id TEXT PRIMARY KEY,
                    flickr_photo_id TEXT NOT NULL,
                    source_record_id TEXT,
                    source_record_hash TEXT NOT NULL,
                    photo_page_url TEXT,
                    image_url TEXT,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    claimed_at TEXT,
                    completed_at TEXT,
                    error TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS comment_review_results (
                    flickr_photo_id TEXT NOT NULL,
                    source_record_hash TEXT NOT NULL,
                    comments_fetched INTEGER NOT NULL,
                    comment_count INTEGER NOT NULL,
                    species_match_from_comments INTEGER NOT NULL,
                    species_name_from_comments TEXT,
                    common_name_from_comments TEXT,
                    life_stage_from_comments TEXT,
                    date_evidence_from_comments TEXT,
                    geo_evidence_from_comments TEXT,
                    location_text_from_comments TEXT,
                    comment_review_decision TEXT NOT NULL,
                    comment_review_reason TEXT NOT NULL,
                    flickr_text_species_candidate TEXT,
                    bioclip_species_candidate TEXT,
                    bioclip_tag_conflict INTEGER NOT NULL,
                    comment_species_candidate TEXT,
                    comment_resolves_conflict INTEGER NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    PRIMARY KEY (flickr_photo_id, source_record_hash)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS comment_derived_terms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    flickr_photo_id TEXT NOT NULL,
                    term TEXT NOT NULL,
                    term_kind TEXT NOT NULL,
                    evidence_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (flickr_photo_id, term, term_kind, evidence_text)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS missing_data_requests (
                    request_id TEXT PRIMARY KEY,
                    flickr_photo_id TEXT NOT NULL,
                    request_type TEXT NOT NULL,
                    evidence_source TEXT NOT NULL,
                    evidence_text TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


def build_comment_review_queue_from_parquet(*, input_path: str | Path, state_db: str | Path) -> dict[str, int]:
    frame = pl.read_parquet(input_path)
    state = CommentReviewState(state_db)
    created = state.enqueue_records(frame.to_dicts())
    return {**state.summary(), "comment_review_queue_created": created}


def review_comments_once(*, state_db: str | Path, max_api_calls: int, api_key: str | None = None, fetch_comments: FetchComments | None = None) -> dict[str, int]:
    state = CommentReviewState(state_db)
    fetcher = fetch_comments
    if fetcher is None:
        if not api_key:
            raise RuntimeError("Flickr API key is required for comment review")
        fetcher = fetch_flickr_comments(api_key=api_key)
    result = state.process_pending(fetch_comments=fetcher, max_api_calls=max_api_calls)
    return {**state.summary(), **result}


def apply_comment_review_decisions_to_parquet(*, input_path: str | Path, output_path: str | Path, state_db: str | Path) -> dict[str, int | str]:
    frame = pl.read_parquet(input_path)
    state = CommentReviewState(state_db)
    rows = state.apply_decisions_to_records(frame.to_dicts())
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(output)
    moved = sum(1 for row in rows if row.get("comment_review_decision") == "move_to_gold")
    return {"output": str(output), "rows": len(rows), "records_moved_to_gold": moved}


def comment_review_reasons(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(record.get("occurrence_bin") or record.get("triage_bin") or "") != "bronze":
        return reasons
    reasons.append("bronze_comment_review")
    flickr_candidate = flickr_text_species_candidate(record)
    bioclip_candidate = bioclip_species_candidate(record)
    conflict = _truthy(record.get("bioclip_tag_conflict")) or _truthy(record.get("bioclip_species_conflict")) or _species_conflict(
        flickr_candidate=flickr_candidate,
        bioclip_candidate=bioclip_candidate,
    )
    if conflict:
        reasons.append("species_conflict")
    if str(record.get("occurrence_bin") or record.get("triage_bin") or "") == "in_review/no_geo":
        reasons.append("in_review_no_geo")
    if not _has_event_date(record):
        reasons.append("missing_event_date")
    if not _has_geo(record):
        reasons.append("missing_geo")
    if str(record.get("image_category") or "") == "unknown":
        reasons.append("unknown_image_category")
    if str(record.get("life_stage") or "") == "unknown":
        reasons.append("unknown_life_stage")
    score = _optional_float(record.get("bioclip_top1_score", record.get("top1_score")))
    if score is not None and score < SILVER_SPECIES_CONFIDENCE_THRESHOLD:
        reasons.append("low_bioclip_score")
    return _ordered_unique(reasons)


def review_comments_for_record(record: dict[str, Any], comments: Iterable[dict[str, Any]]) -> CommentReviewResult:
    comment_list = [comment for comment in comments if isinstance(comment, dict)]
    text = " ".join(value for comment in comment_list if (value := _comment_text(comment)))
    species_name = _species_name_from_text(text)
    common_name = _common_name_from_text(text)
    comment_candidate = species_name or _target_species_from_common_name(common_name)
    life_stage = _life_stage_from_text(text)
    date_evidence = _date_evidence_from_text(text)
    geo_evidence = _structured_geo_from_text(text)
    location_text = _location_text_from_text(text)
    flickr_candidate = flickr_text_species_candidate(record)
    bioclip_candidate = bioclip_species_candidate(record)
    conflict = _truthy(record.get("bioclip_tag_conflict")) or _truthy(record.get("bioclip_species_conflict")) or _species_conflict(
        flickr_candidate=flickr_candidate,
        bioclip_candidate=bioclip_candidate,
    )
    species_match = bool(comment_candidate and bioclip_candidate and _normalize(comment_candidate) == _normalize(bioclip_candidate))
    resolves_conflict = bool(conflict and comment_candidate and bioclip_candidate and _normalize(comment_candidate) == _normalize(bioclip_candidate))
    decision, reason = _comment_review_decision(
        record=record,
        species_match=species_match,
        comment_candidate=comment_candidate,
        life_stage=life_stage,
        date_evidence=date_evidence,
        geo_evidence=geo_evidence,
        location_text=location_text,
        conflict=conflict,
        resolves_conflict=resolves_conflict,
    )
    return CommentReviewResult(
        flickr_photo_id=str(record.get("flickr_photo_id") or record.get("id") or ""),
        comments_fetched=True,
        comment_count=len(comment_list),
        species_match_from_comments=species_match,
        species_name_from_comments=species_name,
        common_name_from_comments=common_name,
        life_stage_from_comments=life_stage,
        date_evidence_from_comments=date_evidence,
        geo_evidence_from_comments=geo_evidence,
        location_text_from_comments=location_text,
        comment_review_decision=decision,
        comment_review_reason=reason,
        flickr_text_species_candidate=flickr_candidate,
        bioclip_species_candidate=bioclip_candidate,
        bioclip_tag_conflict=conflict,
        comment_species_candidate=comment_candidate,
        comment_resolves_conflict=resolves_conflict,
    )


def flickr_text_species_candidate(record: dict[str, Any]) -> str | None:
    text = " ".join(str(record.get(key) or "") for key in ("raw_title", "title", "raw_description", "description", "raw_tags", "tags", "machine_tags"))
    return _species_candidate_from_text(text)


def bioclip_species_candidate(record: dict[str, Any]) -> str | None:
    label = str(record.get("bioclip_top1_label", record.get("top1_label", "")) or "")
    return _species_candidate_from_text(label)


def _comment_review_decision(
    *,
    record: dict[str, Any],
    species_match: bool,
    comment_candidate: str | None,
    life_stage: str | None,
    date_evidence: str | None,
    geo_evidence: str | None,
    location_text: str | None,
    conflict: bool,
    resolves_conflict: bool,
) -> tuple[str, str]:
    if _has_hard_negative_flag(record):
        return "keep_bronze", "hard_negative_visual_material"
    missing_geo = not _has_geo(record)
    missing_date = not _has_event_date(record)
    if _gold_eligible(
        record=record,
        species_match=species_match,
        life_stage=life_stage,
        date_recovered=bool(date_evidence),
        structured_geo_recovered=bool(geo_evidence),
        conflict=conflict,
        resolves_conflict=resolves_conflict,
    ):
        return "move_to_gold", "comments_resolve_review_with_date_geo_and_species_support"
    if missing_geo and location_text and not geo_evidence:
        return "request_missing_geo", "comments_contain_place_name_without_structured_geo"
    if missing_date and not date_evidence:
        return "request_missing_date", "comments_do_not_contain_normalized_event_date"
    if _silver_eligible(record=record, species_match=species_match, date_recovered=bool(date_evidence), structured_geo_recovered=bool(geo_evidence)):
        return "move_to_silver", "comments_support_bioclip_species_but_gold_rules_not_met"
    if conflict and not resolves_conflict:
        return "request_species_review", "comments_do_not_resolve_bioclip_flickr_species_conflict"
    if str(record.get("life_stage") or "") == "unknown" and not life_stage:
        return "request_life_stage_review", "comments_do_not_resolve_unknown_life_stage"
    if str(record.get("occurrence_bin") or record.get("triage_bin") or "") == "in_review/no_geo":
        return "keep_in_review_no_geo", "structured_geo_not_recovered"
    if str(record.get("occurrence_bin") or record.get("triage_bin") or "") == "silver":
        return "keep_silver", "comment_review_did_not_meet_gold_rules"
    if str(record.get("occurrence_bin") or record.get("triage_bin") or "") == "bronze":
        return "keep_bronze", "comment_review_did_not_override_bronze"
    return "no_action", "comment_review_no_change"


def _gold_eligible(
    *,
    record: dict[str, Any],
    species_match: bool,
    life_stage: str | None,
    date_recovered: bool,
    structured_geo_recovered: bool,
    conflict: bool,
    resolves_conflict: bool,
) -> bool:
    return bool(
        _bioclip_score(record) > GOLD_SPECIES_CONFIDENCE_THRESHOLD
        and _has_image_url(record)
        and (_has_event_date(record) or date_recovered)
        and (_has_geo(record) or structured_geo_recovered)
        and (species_match or (conflict and resolves_conflict))
        and not _has_hard_negative_flag(record)
    )


def _silver_eligible(
    *,
    record: dict[str, Any],
    species_match: bool,
    date_recovered: bool,
    structured_geo_recovered: bool,
) -> bool:
    if _has_hard_negative_flag(record) or not species_match or not _has_image_url(record):
        return False
    score = _bioclip_score(record)
    if score < SILVER_SPECIES_CONFIDENCE_THRESHOLD:
        return False
    return not (_has_event_date(record) or date_recovered) or not (_has_geo(record) or structured_geo_recovered) or score <= GOLD_SPECIES_CONFIDENCE_THRESHOLD


def _species_candidate_from_text(text: str) -> str | None:
    normalized = _normalize(text)
    if any(term in normalized for term in TARGET_TERMS):
        return "Papilio demoleus"
    names = SCIENTIFIC_NAME_PATTERN.findall(text)
    if names:
        return names[0]
    if any(term in normalized for term in GENERIC_LEPIDOPTERA_TERMS):
        return "butterfly"
    if any(term in normalized for term in ("moth", "beetle", "wasp", "fly")):
        return "non_target_insect"
    return None


def _species_name_from_text(text: str) -> str | None:
    for name in SCIENTIFIC_NAME_PATTERN.findall(text):
        return name
    return None


def _common_name_from_text(text: str) -> str | None:
    normalized = _normalize(text)
    for term in TARGET_TERMS[1:]:
        if term in normalized:
            return term
    return None


def _target_species_from_common_name(common_name: str | None) -> str | None:
    return "Papilio demoleus" if common_name else None


def _life_stage_from_text(text: str) -> str | None:
    value = infer_life_stage_from_text(text)
    return None if value == "adult_butterfly" else value


def _date_evidence_from_text(text: str) -> str | None:
    match = re.search(r"\b(19|20)\d{2}-\d{2}-\d{2}\b", text)
    return match.group(0) if match else None


def _structured_geo_from_text(text: str) -> str | None:
    match = re.search(r"(?<!\d)(-?\d{1,2}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)(?!\d)", text)
    if not match:
        return None
    latitude = float(match.group(1))
    longitude = float(match.group(2))
    if -90 <= latitude <= 90 and -180 <= longitude <= 180:
        return f"{latitude},{longitude}"
    return None


def _location_text_from_text(text: str) -> str | None:
    match = re.search(r"\b(?:seen|found|spotted|photographed)\s+(?:in|near|at)\s+([A-Z][A-Za-z .'-]{2,60})", text)
    if not match:
        return None
    return " ".join(match.group(1).split()).strip(" .,")


def _species_conflict(*, flickr_candidate: str | None, bioclip_candidate: str | None) -> bool:
    if not flickr_candidate or not bioclip_candidate:
        return False
    if flickr_candidate == bioclip_candidate:
        return False
    if bioclip_candidate == "butterfly" and flickr_candidate == TARGET_SPECIES:
        return False
    return True


def _is_bioclip_lepidoptera_positive(record: dict[str, Any]) -> bool:
    if _truthy(record.get("is_target_positive")):
        return True
    candidate = bioclip_species_candidate(record)
    return bool(candidate and candidate not in {"non_target_insect"})


def _bioclip_score(record: dict[str, Any]) -> float:
    value = _optional_float(record.get("species_top1_score", record.get("bioclip_top1_score", record.get("top1_score"))))
    return -1.0 if value is None else value


def _has_hard_negative_flag(record: dict[str, Any]) -> bool:
    if _truthy(record.get("is_negative_material")):
        return True
    return str(record.get("image_category") or "") in HARD_NEGATIVE_CATEGORIES


def _has_image_url(record: dict[str, Any]) -> bool:
    return bool(record.get("image_url") or record.get("image_url_used"))


def _has_event_date(record: dict[str, Any]) -> bool:
    return bool(record.get("captured_at") or record.get("date_taken") or record.get("datetaken") or record.get("eventDate"))


def _has_geo(record: dict[str, Any]) -> bool:
    latitude = record.get("latitude", record.get("decimalLatitude"))
    longitude = record.get("longitude", record.get("decimalLongitude"))
    return latitude not in (None, "") and longitude not in (None, "")


def _is_duplicate_record(record: dict[str, Any]) -> bool:
    return str(record.get("classification_status") or "") == "skipped_existing" or "duplicate" in str(record.get("bin_reason") or record.get("triage_reason") or "")


def _record_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return json.loads(str(row["record_json"]))


def _comment_text(value: dict[str, Any]) -> str | None:
    return _optional_string(value.get("_content") or value.get("text") or value.get("body"))


def _queue_id(*, photo_id: str, source_record_hash: str) -> str:
    return hashlib.sha256(f"{photo_id}|{source_record_hash}".encode("utf-8")).hexdigest()


def _request_id(photo_id: str, request_type: str, evidence_text: str | None) -> str:
    return hashlib.sha256(f"{photo_id}|{request_type}|{evidence_text or ''}".encode("utf-8")).hexdigest()


def _record_hash(record: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.casefold() in {"1", "true", "yes"}
    return bool(value)


def _normalize(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
