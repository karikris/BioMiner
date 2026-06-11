from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Iterable

import httpx

from biominer.filter.category_model import infer_life_stage_from_text
from biominer.filter.extractor import SCIENTIFIC_NAME_PATTERN
from biominer.flickr_fetch.endpoints import FLICKR_REST_BASE_URL
from biominer.flickr_fetch.query_planner import COUNT_PROBE_PAGE_SIZE, FlickrQuery
from biominer.flickr_fetch.metadata_poller import PENDING, MetadataPollState


COMMENTS_METHOD = "flickr.photos.comments.getList"
COMPLETED = "completed"
FAILED = "failed"

COMMON_NAME_TERMS: tuple[str, ...] = (
    "lime butterfly",
    "chequered swallowtail",
    "checkered swallowtail",
    "citrus swallowtail",
    "swallowtail",
    "butterfly",
)
HARD_NEGATIVE_BRONZE_TERMS: tuple[str, ...] = (
    "museum",
    "specimen",
    "pinned",
    "artwork",
    "illustration",
    "ai generated",
    "ai-generated",
    "other insect",
    "not a butterfly",
    "not_butterfly",
    "moth",
    "object",
    "background",
    "non-organism",
)

FetchComments = Callable[[str], list[dict[str, Any]]]


@dataclass(frozen=True)
class CommentTerm:
    term: str
    term_kind: str


@dataclass(frozen=True)
class PromotedCommentTerm:
    term: str
    term_kind: str
    photo_support_count: int
    user_support_count: int
    work_item_inserted: bool


class CommentsEnrichmentState:
    """SQLite state for selected-photo comment enrichment.

    Removal condition: this state wrapper can be merged into a broader poller
    state object once comments, metadata, and image triage share one stable
    schema migration layer.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        MetadataPollState(self.path)
        self._init_db()

    def queue_candidate(self, record: dict[str, Any], *, selected_for_qa: bool = False) -> int:
        photo_id = str(record.get("flickr_photo_id") or record.get("id") or "")
        if not photo_id:
            return 0
        triage_bin = str(record.get("triage_bin") or record.get("occurrence_bin") or "")
        triage_reason = str(record.get("triage_reason") or record.get("bin_reason") or "")
        if _is_obvious_hard_negative_bronze(triage_bin=triage_bin, triage_reason=triage_reason) and not selected_for_qa:
            return 0
        with self._connect() as conn:
            result = conn.execute(
                """
                INSERT OR IGNORE INTO comments_enrichment_queue (
                    source, flickr_photo_id, photo_page_url, image_url,
                    image_url_kind, source_record_hash, triage_bin,
                    triage_reason, selected_for_qa, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record.get("source") or "flickr"),
                    photo_id,
                    _optional_string(record.get("photo_page_url")),
                    _optional_string(record.get("image_url")),
                    _optional_string(record.get("image_url_kind")),
                    _optional_string(record.get("source_record_hash")),
                    triage_bin or None,
                    triage_reason or None,
                    1 if selected_for_qa else 0,
                    PENDING,
                    _timestamp(),
                ),
            )
        return int(result.rowcount)

    def queue_candidates(self, records: Iterable[dict[str, Any]], *, selected_for_qa: bool = False) -> int:
        return sum(self.queue_candidate(record, selected_for_qa=selected_for_qa) for record in records)

    def pending_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT count(*) FROM comments_enrichment_queue WHERE status = ?", (PENDING,)).fetchone()[0])

    def record_comments(self, *, flickr_photo_id: str, comments: Iterable[dict[str, Any]]) -> int:
        inserted = 0
        with self._connect() as conn:
            for comment in comments:
                text = _comment_text(comment)
                if not text:
                    continue
                author_id = str(comment.get("author") or comment.get("author_id") or comment.get("user") or "")
                for term in mine_comment_terms(text):
                    result = conn.execute(
                        """
                        INSERT OR IGNORE INTO comments_term_observations (
                            flickr_photo_id, author_id, term, term_kind, comment_text, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (flickr_photo_id, author_id, term.term, term.term_kind, text, _timestamp()),
                    )
                    inserted += int(result.rowcount)
            conn.execute(
                "UPDATE comments_enrichment_queue SET status = ?, completed_at = ?, error = NULL WHERE flickr_photo_id = ?",
                (COMPLETED, _timestamp(), flickr_photo_id),
            )
        return inserted

    def process_pending(self, *, fetch_comments: FetchComments, limit: int) -> dict[str, int]:
        if limit <= 0:
            return {"comment_records_processed": 0, "comment_records_failed": 0, "term_observations_inserted": 0}
        processed = 0
        failed = 0
        inserted = 0
        for photo_id in self._pending_photo_ids(limit=limit):
            try:
                inserted += self.record_comments(flickr_photo_id=photo_id, comments=fetch_comments(photo_id))
                processed += 1
            except Exception as exc:  # noqa: BLE001 - enrichment records bounded failures.
                failed += 1
                self._mark_failed(photo_id, str(exc))
        return {
            "comment_records_processed": processed,
            "comment_records_failed": failed,
            "term_observations_inserted": inserted,
        }

    def promote_supported_terms(self, *, min_photos: int = 2, min_users: int = 2) -> list[PromotedCommentTerm]:
        promoted: list[PromotedCommentTerm] = []
        poll_state = MetadataPollState(self.path)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT term, term_kind,
                       count(DISTINCT flickr_photo_id) AS photo_support_count,
                       count(DISTINCT NULLIF(author_id, '')) AS user_support_count
                FROM comments_term_observations
                GROUP BY term, term_kind
                HAVING photo_support_count >= ? AND user_support_count >= ?
                ORDER BY term_kind, term
                """,
                (min_photos, min_users),
            ).fetchall()
            for row in rows:
                existing = conn.execute(
                    "SELECT 1 FROM comment_promoted_terms WHERE term = ? AND term_kind = ?",
                    (row["term"], row["term_kind"]),
                ).fetchone()
                if existing:
                    continue
                query = FlickrQuery(
                    term=str(row["term"]),
                    language="comment",
                    search_field="text",
                    lane="count_probe",
                    per_page=COUNT_PROBE_PAGE_SIZE,
                    split_reason="comment_promoted_term",
                )
                inserted = bool(poll_state.enqueue_work_item(query))
                conn.execute(
                    """
                    INSERT INTO comment_promoted_terms (
                        term, term_kind, photo_support_count, user_support_count,
                        work_item_inserted, promoted_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["term"],
                        row["term_kind"],
                        int(row["photo_support_count"]),
                        int(row["user_support_count"]),
                        1 if inserted else 0,
                        _timestamp(),
                    ),
                )
                promoted.append(
                    PromotedCommentTerm(
                        term=str(row["term"]),
                        term_kind=str(row["term_kind"]),
                        photo_support_count=int(row["photo_support_count"]),
                        user_support_count=int(row["user_support_count"]),
                        work_item_inserted=inserted,
                    )
                )
        return promoted

    def summary(self) -> dict[str, int]:
        with self._connect() as conn:
            return {
                "queued_comment_candidates": int(conn.execute("SELECT count(*) FROM comments_enrichment_queue").fetchone()[0]),
                "pending_comment_candidates": int(
                    conn.execute("SELECT count(*) FROM comments_enrichment_queue WHERE status = ?", (PENDING,)).fetchone()[0]
                ),
                "term_observations": int(conn.execute("SELECT count(*) FROM comments_term_observations").fetchone()[0]),
                "promoted_terms": int(conn.execute("SELECT count(*) FROM comment_promoted_terms").fetchone()[0]),
            }

    def _pending_photo_ids(self, *, limit: int) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT flickr_photo_id
                FROM comments_enrichment_queue
                WHERE status = ?
                ORDER BY created_at, flickr_photo_id
                LIMIT ?
                """,
                (PENDING, limit),
            ).fetchall()
        return [str(row["flickr_photo_id"]) for row in rows]

    def _mark_failed(self, flickr_photo_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE comments_enrichment_queue SET status = ?, completed_at = ?, error = ? WHERE flickr_photo_id = ?",
                (FAILED, _timestamp(), error, flickr_photo_id),
            )

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS comments_enrichment_queue (
                    source TEXT NOT NULL,
                    flickr_photo_id TEXT NOT NULL,
                    photo_page_url TEXT,
                    image_url TEXT,
                    image_url_kind TEXT,
                    source_record_hash TEXT,
                    triage_bin TEXT,
                    triage_reason TEXT,
                    selected_for_qa INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT,
                    PRIMARY KEY (source, flickr_photo_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS comments_term_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    flickr_photo_id TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    term TEXT NOT NULL,
                    term_kind TEXT NOT NULL,
                    comment_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (flickr_photo_id, author_id, term, term_kind)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS comment_promoted_terms (
                    term TEXT NOT NULL,
                    term_kind TEXT NOT NULL,
                    photo_support_count INTEGER NOT NULL,
                    user_support_count INTEGER NOT NULL,
                    work_item_inserted INTEGER NOT NULL,
                    promoted_at TEXT NOT NULL,
                    PRIMARY KEY (term, term_kind)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


def mine_comment_terms(text: str) -> tuple[CommentTerm, ...]:
    normalized = _normalize(text)
    terms: set[CommentTerm] = set()
    for name in SCIENTIFIC_NAME_PATTERN.findall(text):
        terms.add(CommentTerm(term=name, term_kind="scientific_name"))
    occupied_spans: list[tuple[int, int]] = []
    for common_name in sorted(COMMON_NAME_TERMS, key=len, reverse=True):
        match = _phrase_match(normalized, common_name)
        if match and not _span_overlaps(match.span(), occupied_spans):
            terms.add(CommentTerm(term=common_name, term_kind="common_name"))
            occupied_spans.append(match.span())
    life_stage = infer_life_stage_from_text(text)
    if life_stage != "adult_butterfly":
        terms.add(CommentTerm(term=life_stage, term_kind="life_stage"))
    return tuple(sorted(terms, key=lambda item: (item.term_kind, item.term.casefold())))


def fetch_flickr_comments(*, api_key: str) -> FetchComments:
    def fetch(photo_id: str) -> list[dict[str, Any]]:
        params = {
            "method": COMMENTS_METHOD,
            "api_key": api_key,
            "photo_id": photo_id,
            "format": "json",
            "nojsoncallback": 1,
        }
        with httpx.Client(timeout=30) as client:
            response = client.get(FLICKR_REST_BASE_URL, params=params)
        response.raise_for_status()
        payload = response.json()
        comments = payload.get("comments", {})
        rows = comments.get("comment", []) if isinstance(comments, dict) else []
        return [row for row in rows if isinstance(row, dict)]

    return fetch


def _is_obvious_hard_negative_bronze(*, triage_bin: str, triage_reason: str) -> bool:
    if triage_bin.casefold() != "bronze":
        return False
    normalized = _normalize(triage_reason)
    return any(term in normalized for term in HARD_NEGATIVE_BRONZE_TERMS)


def _comment_text(value: dict[str, Any]) -> str | None:
    return _optional_string(value.get("_content") or value.get("text") or value.get("body"))


def _phrase_match(text: str, phrase: str) -> re.Match[str] | None:
    return re.search(rf"(?<!\w){re.escape(phrase.casefold())}(?!\w)", text)


def _span_overlaps(span: tuple[int, int], occupied_spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied_spans)


def _normalize(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
