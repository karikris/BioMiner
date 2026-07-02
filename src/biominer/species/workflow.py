from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from biominer.bioclip.bioclip import BioClipClassifier
from biominer.bioclip.register_runner import RegisterRunnerResult, process_records_with_registers
from biominer.bioclip.species_candidates import SpeciesCandidate, load_species_candidates
from biominer.flickr_comments.comment_review import CommentReviewState, apply_comment_review_decisions_to_parquet
from biominer.flickr_comments.comments_enrichment import fetch_flickr_comments
from biominer.flickr_fetch.metadata_poller import PollOnceResult, MetadataPollState, poll_once
from biominer.flickr_fetch.query_planner import load_registry_flickr_queries
from biominer.species.context import SpeciesContext
from biominer.species.query_compile import write_species_flickr_queries
from biominer.species.registry_refresh import resolve_species_context, write_species_registry_outputs


@dataclass(frozen=True)
class SpeciesRunResult:
    context: SpeciesContext
    output_root: Path
    query_definitions: Path
    state_db: Path
    evidence_output: Path
    poll_result: PollOnceResult | None


def resolve_and_write_context(
    *,
    scientific_name: str | None,
    accepted_taxon_key: str | None,
    registry_dir: str | Path,
    output_root: str | Path,
) -> SpeciesContext:
    context = resolve_species_context(
        scientific_name=scientific_name,
        accepted_taxon_key=accepted_taxon_key,
        registry_dir=registry_dir,
    )
    write_species_registry_outputs(context=context, registry_dir=registry_dir, output_root=output_root)
    return context


def compile_and_write_queries(*, context: SpeciesContext, output_root: str | Path) -> Path:
    output = Path(output_root) / "flickr_query_definitions.parquet"
    write_species_flickr_queries(context, output)
    return output


def seed_species_flickr_work(
    *,
    query_definitions: str | Path,
    state_db: str | Path,
    start_date: str = "2004-02-10",
    end_date: str | None = None,
    slice_days: int = 5,
) -> dict[str, int | str]:
    queries = load_registry_flickr_queries(
        query_definitions,
        start_date=start_date,
        end_date=end_date or __import__("datetime").date.today().isoformat(),
        slice_days=slice_days,
    )
    state = MetadataPollState(state_db)
    inserted = sum(state.enqueue_work_item(query) for query in queries)
    return {"query_definitions": str(query_definitions), "state_db": str(state_db), "work_items_seen": len(queries), "work_items_inserted": inserted}


def fetch_species_flickr(
    *,
    state_db: str | Path,
    output_root: str | Path,
    max_api_calls: int,
    api_key: str | None,
    workers: int,
) -> PollOnceResult:
    output = Path(output_root)
    return poll_once(
        state_db=state_db,
        raw_root=output / "raw",
        evidence_output=output / "evidence" / "poll_once_evidence.parquet",
        max_api_calls=max_api_calls,
        api_key=api_key,
        workers=workers,
    )


def run_species_workflow(
    *,
    scientific_name: str,
    registry_dir: str | Path,
    output_root: str | Path,
    workers: int,
    max_api_calls: int,
    api_key: str | None = None,
    fetch: bool = False,
) -> SpeciesRunResult:
    output = Path(output_root)
    context = resolve_and_write_context(
        scientific_name=scientific_name,
        accepted_taxon_key=None,
        registry_dir=registry_dir,
        output_root=output,
    )
    query_definitions = compile_and_write_queries(context=context, output_root=output)
    state_db = output / "state" / "flickr_poller.sqlite"
    seed_species_flickr_work(query_definitions=query_definitions, state_db=state_db)
    poll_result = (
        fetch_species_flickr(
            state_db=state_db,
            output_root=output,
            max_api_calls=max_api_calls,
            api_key=api_key,
            workers=workers,
        )
        if fetch
        else None
    )
    return SpeciesRunResult(
        context=context,
        output_root=output,
        query_definitions=query_definitions,
        state_db=state_db,
        evidence_output=output / "evidence" / "poll_once_evidence.parquet",
        poll_result=poll_result,
    )


def species_candidates_from_context(context: SpeciesContext) -> list[SpeciesCandidate]:
    return [
        SpeciesCandidate(
            scientific_name=context.scientific_name,
            canonical_name=context.canonical_name,
            rank="species",
            family=context.family,
            genus=context.genus,
            source="species_context",
            source_taxon_id=context.accepted_taxon_key,
            is_target_species=True,
            common_names=tuple(name.name for name in context.common_names),
        )
    ]


def run_species_bioclip_funnel(
    *,
    context: SpeciesContext,
    input_path: str | Path,
    output_path: str | Path,
    classifier: BioClipClassifier,
    cache_root: str | Path,
    register_count: int,
    register_size: int,
    download_workers: int,
    candidate_path: str | Path | None = None,
    candidate_limit: int = 2000,
) -> RegisterRunnerResult:
    records = pl.read_parquet(input_path).to_dicts()
    candidates = (
        load_species_candidates(candidate_path, limit=candidate_limit, target_species=context.scientific_name)
        if candidate_path
        else species_candidates_from_context(context)
    )
    return process_records_with_registers(
        records,
        classifier=classifier,
        species_candidates=candidates,
        output_path=output_path,
        cache_root=cache_root,
        register_count=register_count,
        register_size=register_size,
        download_workers=download_workers,
    )


def build_species_comment_queue(*, context: SpeciesContext, input_path: str | Path, state_db: str | Path) -> dict[str, Any]:
    state = CommentReviewState(state_db, species_context=context)
    frame = pl.read_parquet(input_path)
    created = state.enqueue_records(frame.to_dicts())
    return {**state.summary(), "comment_review_queue_created": created}


def review_species_comments_once(*, context: SpeciesContext, state_db: str | Path, max_api_calls: int, api_key: str | None) -> dict[str, int]:
    state = CommentReviewState(state_db, species_context=context)
    if not api_key:
        raise RuntimeError("Flickr API key is required for comment review")
    result = state.process_pending(fetch_comments=fetch_flickr_comments(api_key=api_key), max_api_calls=max_api_calls)
    return {**state.summary(), **result}


def apply_species_comment_decisions(*, input_path: str | Path, output_path: str | Path, state_db: str | Path) -> dict[str, int | str]:
    return apply_comment_review_decisions_to_parquet(input_path=input_path, output_path=output_path, state_db=state_db)
