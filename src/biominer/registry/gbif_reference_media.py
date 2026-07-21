"""Build an immutable GBIF reference-media manifest from a Darwin Core Archive."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZipFile

import duckdb


REFERENCE_MEDIA_MANIFEST_SCHEMA_VERSION = "gbif-reference-media-manifest-v1.0.0"
REFERENCE_MEDIA_RECEIPT_SCHEMA_VERSION = "gbif-reference-media-receipt-v1.0.0"
DEFAULT_CSV_BLOCK_SIZE = 32 * 1024 * 1024
DEFAULT_PROGRESS_INTERVAL = 1_000_000

_NULL_MARKERS = ["", "NULL", "null", "NA", "na", "N/A", "n/a"]
_OCCURRENCE_MEMBER = "occurrence.txt"
_MULTIMEDIA_MEMBER = "multimedia.txt"
_OCCURRENCE_COLUMNS = (
    "gbifID",
    "occurrenceID",
    "datasetKey",
    "datasetName",
    "publisher",
    "basisOfRecord",
    "lifeStage",
    "taxonKey",
    "acceptedTaxonKey",
    "speciesKey",
    "scientificName",
    "acceptedScientificName",
    "species",
    "genus",
    "family",
    "taxonRank",
    "taxonomicStatus",
)
_MULTIMEDIA_COLUMNS = (
    "gbifID",
    "type",
    "format",
    "identifier",
    "references",
    "title",
    "description",
    "source",
    "audience",
    "created",
    "creator",
    "contributor",
    "publisher",
    "license",
    "rightsHolder",
)
_MEMORY_LIMIT_PATTERN = re.compile(r"[1-9][0-9]*(?:KB|MB|GB|TB)\Z", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class GBIFReferenceMediaManifestConfig:
    archive: Path
    output: Path
    receipt: Path
    download_key: str
    download_doi: str
    download_url: str
    citation: str
    source_snapshot_version: str
    report_dir: Path = Path("reports")
    expected_occurrence_rows: int | None = None
    expected_multimedia_rows: int | None = None
    csv_block_size: int = DEFAULT_CSV_BLOCK_SIZE
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL
    duckdb_threads: int = 4
    duckdb_memory_limit: str = "8GB"
    temp_dir: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "archive", Path(self.archive))
        object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "receipt", Path(self.receipt))
        object.__setattr__(self, "report_dir", Path(self.report_dir))
        if self.temp_dir is not None:
            object.__setattr__(self, "temp_dir", Path(self.temp_dir))
        for field in (
            "download_key",
            "download_doi",
            "download_url",
            "citation",
            "source_snapshot_version",
        ):
            value = str(getattr(self, field)).strip()
            if not value:
                raise ValueError(f"{field} must be non-empty")
            object.__setattr__(self, field, value)
        for field in ("expected_occurrence_rows", "expected_multimedia_rows"):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{field} must be a non-negative integer or null")
        for field in ("csv_block_size", "progress_interval", "duckdb_threads"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        memory_limit = str(self.duckdb_memory_limit).strip().upper()
        if _MEMORY_LIMIT_PATTERN.fullmatch(memory_limit) is None:
            raise ValueError("duckdb_memory_limit must use units such as 8GB")
        object.__setattr__(self, "duckdb_memory_limit", memory_limit)
        if self.output == self.receipt:
            raise ValueError("output and receipt paths must differ")


def build_gbif_reference_media_manifest(
    config: GBIFReferenceMediaManifestConfig,
) -> dict[str, object]:
    """Join every multimedia row to its GBIF occurrence without loading either member."""

    if not isinstance(config, GBIFReferenceMediaManifestConfig):
        raise TypeError("config must be GBIFReferenceMediaManifestConfig")
    started_at = datetime.now(UTC)
    run_id = f"gbif-reference-media-{started_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    report_json = config.report_dir / f"{run_id}.json"
    report_markdown = config.report_dir / f"{run_id}.md"
    if not config.archive.is_file():
        raise FileNotFoundError(config.archive)
    existing = [path for path in (config.output, config.receipt) if path.exists()]
    if existing:
        raise FileExistsError(
            "immutable GBIF reference-media output already exists: "
            + ", ".join(str(path) for path in existing)
        )

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.receipt.parent.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    if config.temp_dir is not None:
        config.temp_dir.mkdir(parents=True, exist_ok=True)

    archive_sha256 = _sha256_file(config.archive)
    archive_bytes = config.archive.stat().st_size
    output_staging = config.output.with_name(
        f".{config.output.name}.{uuid4().hex}.staging"
    )
    receipt_staging = config.receipt.with_name(
        f".{config.receipt.name}.{uuid4().hex}.staging"
    )
    report_json_staging = report_json.with_name(
        f".{report_json.name}.{uuid4().hex}.staging"
    )
    report_markdown_staging = report_markdown.with_name(
        f".{report_markdown.name}.{uuid4().hex}.staging"
    )
    try:
        temporary_parent = str(config.temp_dir) if config.temp_dir is not None else None
        with TemporaryDirectory(
            prefix="biominer-gbif-reference-media-",
            dir=temporary_parent,
        ) as temporary_directory:
            temporary = Path(temporary_directory)
            occurrence_projection = temporary / "occurrence_projection.parquet"
            multimedia_projection = temporary / "multimedia_projection.parquet"
            occurrence_rows = _project_member(
                config.archive,
                _OCCURRENCE_MEMBER,
                _OCCURRENCE_COLUMNS,
                occurrence_projection,
                block_size=config.csv_block_size,
                progress_interval=config.progress_interval,
                include_source_row_number=False,
            )
            multimedia_rows = _project_member(
                config.archive,
                _MULTIMEDIA_MEMBER,
                _MULTIMEDIA_COLUMNS,
                multimedia_projection,
                block_size=config.csv_block_size,
                progress_interval=config.progress_interval,
                include_source_row_number=True,
            )
            _validate_expected_rows(
                member=_OCCURRENCE_MEMBER,
                actual=occurrence_rows,
                expected=config.expected_occurrence_rows,
            )
            _validate_expected_rows(
                member=_MULTIMEDIA_MEMBER,
                actual=multimedia_rows,
                expected=config.expected_multimedia_rows,
            )
            join_metrics = _join_projections(
                occurrence_projection=occurrence_projection,
                multimedia_projection=multimedia_projection,
                output=output_staging,
                config=config,
                archive_sha256=archive_sha256,
                archive_bytes=archive_bytes,
            )

        output_sha256 = _sha256_file(output_staging)
        ended_at = datetime.now(UTC)
        git_sha, working_tree_dirty = _git_identity()
        receipt = {
            "schema_version": REFERENCE_MEDIA_RECEIPT_SCHEMA_VERSION,
            "generated_at": ended_at.isoformat().replace("+00:00", "Z"),
            "run": {
                "run_id": run_id,
                "git_sha": git_sha,
                "working_tree_dirty": working_tree_dirty,
                "command": "biominer dev registry build-gbif-reference-media",
                "started_at": started_at.isoformat().replace("+00:00", "Z"),
                "ended_at": ended_at.isoformat().replace("+00:00", "Z"),
                "duration_seconds": (ended_at - started_at).total_seconds(),
                "status": "completed",
            },
            "scientific_scope": (
                "GBIF provider metadata and media relationships are provisional source "
                "evidence, not verified organism identity or life-stage labels."
            ),
            "source": {
                "provider": "GBIF",
                "download_key": config.download_key,
                "download_doi": config.download_doi,
                "download_url": config.download_url,
                "citation": config.citation,
                "source_snapshot_version": config.source_snapshot_version,
                "archive_path": str(config.archive),
                "archive_bytes": archive_bytes,
                "archive_sha256": f"sha256:{archive_sha256}",
            },
            "input_rows": {
                "occurrence": occurrence_rows,
                "multimedia": multimedia_rows,
            },
            "join": join_metrics,
            "output": {
                "path": str(config.output),
                "row_count": join_metrics["output_rows"],
                "physical_bytes": output_staging.stat().st_size,
                "sha256": f"sha256:{output_sha256}",
                "compression": "zstd",
                "schema_version": REFERENCE_MEDIA_MANIFEST_SCHEMA_VERSION,
            },
            "metrics": {
                "rows_in": occurrence_rows + multimedia_rows,
                "rows_out": join_metrics["output_rows"],
                "records_seen": multimedia_rows,
                "records_classified": None,
                "records_skipped_existing": 0,
                "download_failures": None,
                "bioclip_failures": None,
                "bucket_counts": None,
                "category_counts": None,
                "life_stage_counts": None,
                "candidate_set_count": None,
                "avg_records_per_candidate_set": None,
                "max_records_per_candidate_set": None,
                "geo_candidate_coverage": None,
                "geo_candidate_fallback_counts": None,
                "rss_peak_memory": None,
                "gpu_memory_peak": None,
                "mps_current_allocated_memory": None,
                "mps_driver_allocated_memory": None,
                "mps_recommended_max_memory": None,
                "images_per_second": None,
                "seconds_per_image": None,
            },
            "reports": {
                "json": str(report_json),
                "markdown": str(report_markdown),
            },
        }
        receipt_staging.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_json_staging.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_markdown_staging.write_text(
            _report_markdown(receipt),
            encoding="utf-8",
        )
        output_staging.replace(config.output)
        receipt_staging.replace(config.receipt)
        report_json_staging.replace(report_json)
        report_markdown_staging.replace(report_markdown)
        return receipt
    finally:
        output_staging.unlink(missing_ok=True)
        receipt_staging.unlink(missing_ok=True)
        report_json_staging.unlink(missing_ok=True)
        report_markdown_staging.unlink(missing_ok=True)


def _project_member(
    archive_path: Path,
    member_name: str,
    columns: tuple[str, ...],
    output: Path,
    *,
    block_size: int,
    progress_interval: int,
    include_source_row_number: bool,
) -> int:
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    with ZipFile(archive_path) as archive:
        try:
            raw_member = archive.open(member_name)
        except KeyError as error:
            raise ValueError(f"DWCA archive is missing {member_name}") from error
        with raw_member:
            header = raw_member.readline().decode("utf-8-sig").rstrip("\r\n").split("\t")
            missing = [column for column in columns if column not in header]
            if missing:
                raise ValueError(
                    f"{member_name} is missing required columns: {', '.join(missing)}"
                )
            schema_fields = [(column, pa.string()) for column in columns]
            if include_source_row_number:
                schema_fields.append(("source_media_row_number", pa.int64()))
            output_schema = pa.schema(schema_fields)
            reader = pacsv.open_csv(
                raw_member,
                read_options=pacsv.ReadOptions(
                    block_size=block_size,
                    column_names=header,
                    use_threads=True,
                ),
                parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                convert_options=pacsv.ConvertOptions(
                    column_types={column: pa.string() for column in header},
                    include_columns=list(columns),
                    null_values=_NULL_MARKERS,
                    strings_can_be_null=True,
                ),
            )
            rows = 0
            reported = 0
            with pq.ParquetWriter(output, output_schema, compression="zstd") as writer:
                for batch in reader:
                    table = pa.Table.from_batches([batch])
                    if include_source_row_number:
                        table = table.append_column(
                            "source_media_row_number",
                            pa.array(
                                range(rows + 1, rows + table.num_rows + 1),
                                type=pa.int64(),
                            ),
                        )
                    writer.write_table(table)
                    rows += table.num_rows
                    if rows // progress_interval > reported // progress_interval:
                        reported = rows
                        print(
                            json.dumps(
                                {
                                    "event": "gbif_reference_media_projection_progress",
                                    "member": member_name,
                                    "rows_written": rows,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
            return rows


def _join_projections(
    *,
    occurrence_projection: Path,
    multimedia_projection: Path,
    output: Path,
    config: GBIFReferenceMediaManifestConfig,
    archive_sha256: str,
    archive_bytes: int,
) -> dict[str, int]:
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={config.duckdb_threads}")
        connection.execute(f"SET memory_limit='{config.duckdb_memory_limit}'")
        connection.execute(
            f"SET temp_directory='{_sql_text(str(config.temp_dir or output.parent))}'"
        )
        connection.execute(
            f"CREATE VIEW occurrence_source AS SELECT * FROM read_parquet('{_sql_text(str(occurrence_projection))}')"
        )
        connection.execute(
            f"CREATE VIEW multimedia_source AS SELECT * FROM read_parquet('{_sql_text(str(multimedia_projection))}')"
        )
        null_occurrence_ids = int(
            connection.execute(
                "SELECT count(*) FROM occurrence_source WHERE gbifID IS NULL OR trim(gbifID) = ''"
            ).fetchone()[0]
        )
        if null_occurrence_ids:
            raise ValueError(
                f"occurrence.txt contains {null_occurrence_ids} rows without gbifID"
            )
        duplicate_occurrence_ids = int(
            connection.execute(
                "SELECT count(*) FROM (SELECT gbifID FROM occurrence_source GROUP BY gbifID HAVING count(*) > 1)"
            ).fetchone()[0]
        )
        if duplicate_occurrence_ids:
            raise ValueError(
                "occurrence.txt contains duplicate gbifID values: "
                f"{duplicate_occurrence_ids} duplicated keys"
            )

        source_constants = {
            "schema": REFERENCE_MEDIA_MANIFEST_SCHEMA_VERSION,
            "snapshot": config.source_snapshot_version,
            "download_key": config.download_key,
            "download_doi": config.download_doi,
            "download_url": config.download_url,
            "citation": config.citation,
            "archive_sha256": f"sha256:{archive_sha256}",
        }
        copy_sql = f"""
            COPY (
                SELECT
                    '{_sql_text(source_constants['schema'])}' AS schema_version,
                    'GBIF' AS source,
                    '{_sql_text(source_constants['snapshot'])}' AS source_snapshot_version,
                    '{_sql_text(source_constants['download_key'])}' AS source_download_key,
                    '{_sql_text(source_constants['download_doi'])}' AS source_download_doi,
                    '{_sql_text(source_constants['download_url'])}' AS source_download_url,
                    '{_sql_text(source_constants['citation'])}' AS source_citation,
                    '{_sql_text(source_constants['archive_sha256'])}' AS source_archive_sha256,
                    {archive_bytes}::UBIGINT AS source_archive_bytes,
                    m.source_media_row_number,
                    m.gbifID AS gbif_id,
                    o.gbifID IS NOT NULL AS occurrence_joined,
                    o.occurrenceID AS occurrence_id,
                    o.datasetKey AS dataset_key,
                    o.datasetName AS dataset_name,
                    o.publisher AS occurrence_publisher,
                    o.basisOfRecord AS basis_of_record,
                    o.lifeStage AS provider_life_stage,
                    o.taxonKey AS taxon_key,
                    o.acceptedTaxonKey AS accepted_taxon_key,
                    o.speciesKey AS species_key,
                    o.scientificName AS scientific_name,
                    o.acceptedScientificName AS accepted_scientific_name,
                    o.species AS species,
                    o.genus AS genus,
                    o.family AS family,
                    o.taxonRank AS taxon_rank,
                    o.taxonomicStatus AS taxonomic_status,
                    m.type AS media_type,
                    m.format AS media_format,
                    m.identifier AS image_url,
                    m.references AS media_references,
                    m.title AS media_title,
                    m.description AS media_description,
                    m.source AS media_source,
                    m.audience AS media_audience,
                    m.created AS media_created,
                    m.creator AS media_creator,
                    m.contributor AS media_contributor,
                    m.publisher AS media_publisher,
                    m.license AS media_license,
                    m.rightsHolder AS media_rights_holder,
                    lower(coalesce(m.type, '')) IN ('stillimage', 'still image')
                        OR lower(coalesce(m.format, '')) LIKE 'image/%'
                        AS media_is_still_image,
                    lower(coalesce(m.identifier, '')) LIKE 'http://%'
                        OR lower(coalesce(m.identifier, '')) LIKE 'https://%'
                        AS image_url_is_http,
                    'sha256:' || sha256(concat_ws(
                        chr(31),
                        coalesce(m.gbifID, ''),
                        cast(m.source_media_row_number AS VARCHAR),
                        coalesce(m.identifier, ''),
                        coalesce(m.type, ''),
                        coalesce(m.format, ''),
                        coalesce(m.license, ''),
                        coalesce(m.rightsHolder, ''),
                        '{_sql_text(source_constants['archive_sha256'])}'
                    )) AS source_row_fingerprint
                FROM multimedia_source AS m
                LEFT JOIN occurrence_source AS o ON o.gbifID = m.gbifID
                ORDER BY m.source_media_row_number
            ) TO '{_sql_text(str(output))}' (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000
            )
        """
        connection.execute(copy_sql)
        output_rows = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet('{_sql_text(str(output))}')"
            ).fetchone()[0]
        )
        if output_rows == 0:
            raise ValueError("GBIF reference-media manifest is empty")
        if output_rows != int(
            connection.execute("SELECT count(*) FROM multimedia_source").fetchone()[0]
        ):
            raise ValueError("GBIF reference-media join changed the multimedia row count")
        joined_rows, unjoined_rows, still_image_rows, http_url_rows = (
            int(value)
            for value in connection.execute(
                f"""
                SELECT
                    count(*) FILTER (WHERE occurrence_joined),
                    count(*) FILTER (WHERE NOT occurrence_joined),
                    count(*) FILTER (WHERE media_is_still_image),
                    count(*) FILTER (WHERE image_url_is_http)
                FROM read_parquet('{_sql_text(str(output))}')
                """
            ).fetchone()
        )
        return {
            "output_rows": output_rows,
            "joined_rows": joined_rows,
            "unjoined_rows": unjoined_rows,
            "still_image_rows": still_image_rows,
            "http_url_rows": http_url_rows,
            "duplicate_occurrence_ids": duplicate_occurrence_ids,
            "null_occurrence_ids": null_occurrence_ids,
        }
    finally:
        connection.close()


def _validate_expected_rows(*, member: str, actual: int, expected: int | None) -> None:
    if actual == 0:
        raise ValueError(f"{member} is empty")
    if expected is not None and actual != expected:
        raise ValueError(f"{member} row count {actual} does not match expected {expected}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_text(value: str) -> str:
    return str(value).replace("'", "''")


def _git_identity() -> tuple[str | None, bool | None]:
    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return sha_result.stdout.strip() or None, bool(status_result.stdout.strip())


def _report_markdown(receipt: dict[str, object]) -> str:
    run = receipt["run"]
    source = receipt["source"]
    inputs = receipt["input_rows"]
    join = receipt["join"]
    output = receipt["output"]
    assert isinstance(run, dict)
    assert isinstance(source, dict)
    assert isinstance(inputs, dict)
    assert isinstance(join, dict)
    assert isinstance(output, dict)
    return "\n".join(
        (
            "# GBIF reference-media manifest build",
            "",
            f"- Run ID: `{run['run_id']}`",
            f"- Status: `{run['status']}`",
            f"- Started: `{run['started_at']}`",
            f"- Ended: `{run['ended_at']}`",
            f"- Git SHA: `{run['git_sha']}`",
            f"- Working tree dirty: `{run['working_tree_dirty']}`",
            "",
            "## Source",
            "",
            f"- Download key: `{source['download_key']}`",
            f"- DOI: {source['download_doi']}",
            f"- Download URL: {source['download_url']}",
            f"- Archive SHA-256: `{source['archive_sha256']}`",
            "",
            "## Result",
            "",
            f"- Occurrence rows: {inputs['occurrence']:,}",
            f"- Multimedia rows: {inputs['multimedia']:,}",
            f"- Output rows: {join['output_rows']:,}",
            f"- Joined rows: {join['joined_rows']:,}",
            f"- Unjoined rows: {join['unjoined_rows']:,}",
            f"- Still-image rows: {join['still_image_rows']:,}",
            f"- HTTP(S) URL rows: {join['http_url_rows']:,}",
            f"- Parquet: `{output['path']}`",
            f"- Parquet bytes: {output['physical_bytes']:,}",
            f"- Parquet SHA-256: `{output['sha256']}`",
            "",
            "GBIF provider metadata and relationships are source evidence, not "
            "verified organism identity or life-stage labels.",
            "",
        )
    )


__all__ = [
    "GBIFReferenceMediaManifestConfig",
    "REFERENCE_MEDIA_MANIFEST_SCHEMA_VERSION",
    "REFERENCE_MEDIA_RECEIPT_SCHEMA_VERSION",
    "build_gbif_reference_media_manifest",
]
