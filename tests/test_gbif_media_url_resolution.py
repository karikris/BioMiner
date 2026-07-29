from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import time
import zipfile

import duckdb
import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from PIL import Image

from biominer.cli import build_parser
from biominer.gbif_media_resolution.cli import (
    COMMAND,
    _receipt_count,
    run_gbif_media_resolution_command,
)

from biominer.gbif_media_resolution.archive_circuit import (
    ARCHIVE_CIRCUIT_VERSION,
    TERMINAL_REASON,
    complete_archive_reference_only_rows,
)
from biominer.gbif_media_resolution.models import (
    ATTEMPT_SCHEMA,
    JOB_NAME,
    RESULT_SCHEMA,
    STAGE,
    ResolutionInput,
    ResolutionStatus,
    source_row_id,
)
from biominer.gbif_media_resolution.pipeline import (
    finalize_resolution,
    host_fair_schedule,
    import_pilot_cache,
    prepare_resolution,
    publish_v4,
    rebalance_resolution_queue,
    pilot_selection_table,
    run_targeted_provider_batch,
    run_worker,
    select_pilot_inputs,
)
from biominer.gbif_media_resolution.resolver import (
    SAME_HOST_MIXED_CONTENT_POLICY_VERSION,
    MediaURLResolver,
    ResolverConfig,
    extract_structured_image_candidates,
    sniff_image_content_type,
    validate_public_http_url,
    redact_url_for_audit,
    upgrade_same_host_mixed_content_candidate,
)
from biominer.workstore.sqlite import SQLiteWorkStore


PUBLIC_DNS = lambda _host: ("93.184.216.34",)  # noqa: E731


def _image_bytes(image_format: str) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(buffer, format=image_format)
    return buffer.getvalue()


def _input(**overrides: object) -> ResolutionInput:
    values: dict[str, object] = {
        "source_row_id": source_row_id(
            "sha256:source", "123", "https://example.org/record/123"
        ),
        "source_artifact_sha256": "sha256:source",
        "gbif_id": "123",
        "media_references": "https://example.org/record/123",
        "media_type": "StillImage",
        "media_format": None,
        "media_license": "CC BY 4.0",
        "occurrence_license": "CC BY 4.0",
    }
    values.update(overrides)
    return ResolutionInput(**values)  # type: ignore[arg-type]


def _write_source(path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "gbifID": ["1", "2", "3", "4"],
                "license": ["CC0", "CC BY", "CC BY", "CC BY"],
                "occurrenceID": ["o1", "o2", "o3", "o4"],
                "media_type": ["StillImage"] * 4,
                "media_format": [None] * 4,
                "media_identifier": [None, "https://example.org/existing.jpg", None, None],
                "media_references": [
                    "https://example.org/record/1",
                    "https://example.org/record/2",
                    "https://youtube.com/watch?v=abc",
                    "https://example.org/record/4",
                ],
                "media_license": [
                    "CC0",
                    "CC BY",
                    "CC BY",
                    "All rights reserved",
                ],
            }
        ),
        path,
    )


def _write_pilot_acceptance(directory: Path, source: Path) -> Path:
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    source_snapshot = "sha256:" + source_sha
    source_table = pq.read_table(source)
    pilot_results: list[dict[str, object]] = []
    pilot_attempts: list[dict[str, object]] = []
    for row in source_table.to_pylist():
        if row["media_identifier"] is not None or row["media_references"] is None:
            continue
        identity = source_row_id(
            source_snapshot,
            str(row["gbifID"]),
            str(row["media_references"]),
        )
        rights_blocked = row["media_license"] == "All rights reserved"
        pilot_results.append(
            {
                "source_row_id": identity,
                "source_artifact_sha256": source_snapshot,
                "gbif_id": str(row["gbifID"]),
                "media_references": str(row["media_references"]),
                "status": (
                    ResolutionStatus.RIGHTS_BLOCKED.value
                    if rights_blocked
                    else ResolutionStatus.UNRESOLVED_NOT_FOUND.value
                ),
                "terminal_reason": (
                    "fixture_rights_blocked"
                    if rights_blocked
                    else "fixture_not_found"
                ),
            }
        )
        if not rights_blocked:
            pilot_attempts.append(
                {
                    "attempt_id": f"attempt-{identity}",
                    "source_row_id": identity,
                    "sequence": 1,
                    "phase": "fixture",
                    "method": "fixture",
                    "outcome": "not_found",
                }
            )
    resolution_directory = directory / "pilot-resolution"
    resolution_directory.mkdir()
    result_path = resolution_directory / "resolution_results.parquet"
    attempt_path = resolution_directory / "resolution_attempts.parquet"
    pq.write_table(
        pa.Table.from_pylist(pilot_results, schema=RESULT_SCHEMA),
        result_path,
    )
    pq.write_table(
        pa.Table.from_pylist(pilot_attempts, schema=ATTEMPT_SCHEMA),
        attempt_path,
    )
    result_inventory = _test_parquet_inventory(result_path)
    attempt_inventory = _test_parquet_inventory(attempt_path)
    resolution_manifest = resolution_directory / "manifest.json"
    resolution_manifest.write_text(
        json.dumps(
            {
                "schema_version": "biominer-gbif-media-url-resolution/v1",
                "run_id": "fixture-pilot",
                "input": {
                    "mode": "pilot",
                    "source_artifact_sha256": source_snapshot,
                    "work_rows": len(pilot_results),
                },
                "counts": {
                    "result_rows": len(pilot_results),
                    "attempt_rows": len(pilot_attempts),
                },
                "artifacts": {
                    result_path.name: result_inventory,
                    attempt_path.name: attempt_inventory,
                },
                "validation": {
                    "one_result_per_input": True,
                    "unique_source_row_ids": True,
                    "every_work_item_completed": True,
                    "rights_blocked_zero_attempts": True,
                    "all_parquet_row_groups_complete": True,
                },
                "manifest_policy": {"written_last": True},
            }
        ),
        encoding="utf-8",
    )
    artifact = directory / "pilot-gates.parquet"
    pq.write_table(
        pa.table(
            {
                "gate_id": ["PILOT_001"],
                "status": ["PASS"],
            }
        ),
        artifact,
    )
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = directory / "pilot-acceptance.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "biominer-gbif-media-url-pilot-execution-audit/v1",
                "source_snapshot_id": source_snapshot,
                "overall_acceptance_status": "PASS",
                "input": {
                    "resolution_manifest": str(resolution_manifest),
                },
                "counts": {
                    "pilot_rows": len(pilot_results),
                    "result_rows": len(pilot_results),
                    "attempt_rows": len(pilot_attempts),
                },
                "artifacts": [
                    {
                        "path": artifact.name,
                        "sha256": artifact_sha,
                    }
                ],
                "validation": {
                    "all_resolved_rows_reviewed": True,
                    "rights_blocked_zero_attempts": True,
                    "unresolved_reasons_complete": True,
                    "manifest_written_last": True,
                    "resolution_checksums_match": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _test_parquet_inventory(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    row_groups = [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ]
    return {
        "path": path.name,
        "physical_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": parquet.metadata.num_rows,
        "row_groups_complete": (
            sum(row_groups) == parquet.metadata.num_rows
            and (
                parquet.metadata.num_rows == 0
                or all(row_count > 0 for row_count in row_groups)
            )
        ),
    }


def test_source_row_id_is_trimmed_and_source_bound() -> None:
    first = source_row_id("sha256:a", "12", " https://example.org/a ")
    assert first == source_row_id("sha256:a", "12", "https://example.org/a")
    assert first != source_row_id("sha256:b", "12", "https://example.org/a")


def test_pilot_selection_uses_host_size_bands_deterministically() -> None:
    rows: list[ResolutionInput] = []
    for host, count in (("large.test", 1_001), ("medium.test", 26), ("small.test", 4)):
        for index in range(count):
            reference = f"https://{host}/{index}"
            rows.append(
                _input(
                    source_row_id=source_row_id("sha256:source", str(index), reference),
                    gbif_id=str(index),
                    media_references=reference,
                )
            )
    selected = select_pilot_inputs(list(reversed(rows)))
    counts: dict[str, int] = {}
    for item in selected:
        counts[item.host] = counts.get(item.host, 0) + 1
    assert counts == {"large.test": 100, "medium.test": 25, "small.test": 4}
    assert [item.source_row_id for item in selected] == sorted(
        item.source_row_id for item in selected
    )


def test_pilot_selection_round_robins_context_strata() -> None:
    rows = []
    for index in range(1_000):
        reference = f"https://large.test/{index}"
        rows.append(_input(
            source_row_id=source_row_id("sha256:source", str(index), reference),
            gbif_id=str(index), media_references=reference,
            provider="provider-a" if index < 500 else "provider-b",
            publisher="publisher-a" if index < 500 else "publisher-b",
            taxon_rank="SPECIES", country_code="AU",
        ))

    selected = select_pilot_inputs(rows)
    providers = [item.provider for item in selected]
    table = pilot_selection_table(selected, population=rows)

    assert len(selected) == 100
    assert providers.count("provider-a") == 50
    assert providers.count("provider-b") == 50
    assert table.num_rows == 100
    assert set(table.column("host_size_band").to_pylist()) == {"large"}
    assert all(table.column("selection_stratum").to_pylist())


def test_resolution_input_payload_preserves_pilot_context() -> None:
    item = _input(
        provider="provider", publisher="publisher", dataset_name="dataset",
        taxon_rank="SPECIES", country_code="AU",
    )

    assert ResolutionInput.from_payload(item.to_payload()) == item


def test_public_url_validation_rejects_credentials_and_private_dns() -> None:
    assert (
        validate_public_http_url(
            "https://example.org/image.jpg", resolve_host=PUBLIC_DNS
        ).host
        == "example.org"
    )
    for url, addresses in (
        ("ftp://example.org/x", PUBLIC_DNS),
        ("https://user:secret@example.org/x", PUBLIC_DNS),
        ("https://example.org/bad%2", PUBLIC_DNS),
        ("https://example.org/bad\npath", PUBLIC_DNS),
        ("https://example.org/x", lambda _host: ("127.0.0.1",)),
        (
            "https://example.org/x",
            lambda _host: ("93.184.216.34", "10.0.0.1"),
        ),
    ):
        try:
            validate_public_http_url(url, resolve_host=addresses)
        except ValueError:
            pass
        else:  # pragma: no cover - assertion spelling keeps the failing URL visible.
            raise AssertionError(f"unsafe URL accepted: {url}")


def test_audit_url_redacts_signed_query_values() -> None:
    value = redact_url_for_audit(
        "https://example.org/image.jpg?X-Amz-Signature=secret&size=large"
    )

    assert "secret" not in value
    assert "size=large" in value


def test_cli_requires_explicit_network_and_full_queue_opt_in() -> None:
    parser = build_parser()
    prepare = parser.parse_args([
        COMMAND, "prepare", "--source", "x", "--source-manifest", "m",
        "--output-root", "o", "--run-id", "r",
    ])
    assert prepare.mode == "pilot"
    full = parser.parse_args([
        COMMAND, "prepare", "--source", "x", "--source-manifest", "m",
        "--output-root", "o", "--run-id", "r", "--mode", "full",
    ])
    with pytest.raises(ValueError, match="--allow-full-queue"):
        run_gbif_media_resolution_command(full)
    full_without_accepted_pilot = parser.parse_args([
        COMMAND, "prepare", "--source", "x", "--source-manifest", "m",
        "--output-root", "o", "--run-id", "r", "--mode", "full",
        "--allow-full-queue",
    ])
    with pytest.raises(ValueError, match="pilot acceptance manifest"):
        run_gbif_media_resolution_command(full_without_accepted_pilot)
    work = parser.parse_args([
        COMMAND, "work", "--output-root", "o", "--run-id", "r",
        "--worker-id", "w",
    ])
    with pytest.raises(ValueError, match="--execute-network"):
        run_gbif_media_resolution_command(work)
    review = parser.parse_args(
        [
            COMMAND,
            "prepare-review",
            "--pilot-selection",
            "selection.parquet",
            "--resolution-directory",
            "resolution",
            "--output",
            "review.parquet",
        ]
    )
    assert review.gbif_media_url_command == "prepare-review"
    audit = parser.parse_args(
        [
            COMMAND,
            "audit-pilot",
            "--prepare-receipt",
            "prepare.json",
            "--pilot-selection",
            "selection.parquet",
            "--resolution-directory",
            "resolution",
            "--reviewed-pilot",
            "reviewed.parquet",
            "--output-directory",
            "audit",
            "--code-commit",
            "deadbeef",
            "--adapter-test-receipt",
            "tests.json",
        ]
    )
    assert audit.gbif_media_url_command == "audit-pilot"
    import_cache = parser.parse_args(
        [
            COMMAND,
            "import-pilot-cache",
            "--output-root",
            "runtime",
            "--run-id",
            "full-run",
        ]
    )
    assert import_cache.gbif_media_url_command == "import-pilot-cache"
    rebalance = parser.parse_args(
        [
            COMMAND,
            "rebalance",
            "--output-root",
            "runtime",
            "--run-id",
            "full-run",
            "--chunk-rows",
            "25",
        ]
    )
    assert rebalance.gbif_media_url_command == "rebalance"
    archive_circuit = parser.parse_args(
        [
            COMMAND,
            "archive-circuit",
            "--output-root",
            "runtime",
            "--run-id",
            "full-run",
            "--archive-manifest",
            "archives/manifest.json",
            "--provider",
            "Provider",
            "--dataset-key",
            "dataset-key",
            "--expected-pending-rows",
            "2",
        ]
    )
    assert archive_circuit.gbif_media_url_command == "archive-circuit"
    assert archive_circuit.expected_pending_rows == 2
    targeted = parser.parse_args(
        [
            COMMAND,
            "targeted-provider",
            "--output-root",
            "runtime",
            "--run-id",
            "full-run",
            "--provider",
            "Provider",
            "--policy",
            "same-host-http-to-https",
        ]
    )
    with pytest.raises(ValueError, match="--execute-network"):
        run_gbif_media_resolution_command(targeted)


def test_structured_html_extraction_does_not_scrape_generic_images() -> None:
    html = b"""
      <html><head>
      <meta property="og:image" content="/media/specimen.jpg">
      </head><body><img src="/logo.png"><img src="/unrelated.jpg"></body></html>
    """
    assert extract_structured_image_candidates(
        html, base_url="https://example.org/record/1"
    ) == ("https://example.org/media/specimen.jpg",)


def test_mixed_content_upgrade_is_same_host_and_default_port_only() -> None:
    source = "https://images.example.org/record/1"
    assert upgrade_same_host_mixed_content_candidate(
        "http://images.example.org/media/1.jpg",
        source,
    ) == "https://images.example.org/media/1.jpg"
    assert upgrade_same_host_mixed_content_candidate(
        "http://images.example.org:80/media/1.jpg?size=large",
        source,
    ) == "https://images.example.org/media/1.jpg?size=large"
    for candidate in (
        "http://other.example.org/media/1.jpg",
        "http://images.example.org:8080/media/1.jpg",
        "https://images.example.org/media/1.jpg",
        "http://user@images.example.org/media/1.jpg",
    ):
        assert (
            upgrade_same_host_mixed_content_candidate(candidate, source)
            == candidate
        )


def test_receipt_count_supports_manifest_and_empty_batch_shapes() -> None:
    assert _receipt_count(
        {"counts": {"selected_rows": 2}, "selected_rows": 99},
        "selected_rows",
    ) == 2
    assert _receipt_count({"selected_rows": 0}, "selected_rows") == 0
    assert _receipt_count({}, "selected_rows") == 0


def test_targeted_provider_batch_records_mixed_content_rewrite_and_image(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    state = SQLiteWorkStore(tmp_path / "state.sqlite")
    state.get_or_create_run(
        job_name=JOB_NAME,
        stage=STAGE,
        run_id="targeted-run",
        registry_version="sha256:source",
        config={
            "output_root": str(runtime.resolve()),
            "resolver_fingerprint": "sha256:prepared-resolver",
        },
    )
    provider = "Mixed Content Provider"
    inputs = [
        _input(
            source_row_id=f"targeted-{index}",
            gbif_id=str(index),
            media_references=f"https://images.test/record/{index}",
            provider=provider,
        )
        for index in range(1, 3)
    ]
    state.enqueue_work(
        JOB_NAME,
        "sha256:source",
        [
            {"work_key": item.source_row_id, **item.to_payload()}
            for item in inputs
        ],
        stage=STAGE,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/record/"):
            image_id = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html"},
                content=(
                    "<meta property='og:image' "
                    f"content='http://images.test/media/{image_id}.png'>"
                ).encode(),
            )
        if request.url.scheme == "https" and request.url.path.startswith(
            "/media/"
        ):
            return httpx.Response(
                206,
                headers={"Content-Type": "image/png"},
                content=_image_bytes("PNG"),
            )
        raise AssertionError(f"unexpected request: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with MediaURLResolver(
            config=ResolverConfig(max_attempts=1),
            http_client=client,
            resolve_host=PUBLIC_DNS,
            structured_candidate_rewriter=(
                upgrade_same_host_mixed_content_candidate
            ),
            structured_candidate_rewriter_version=(
                SAME_HOST_MIXED_CONTENT_POLICY_VERSION
            ),
        ) as resolver:
            receipt = run_targeted_provider_batch(
                workstore=state,
                output_root=runtime,
                run_id="targeted-run",
                provider=provider,
                policy_version=SAME_HOST_MIXED_CONTENT_POLICY_VERSION,
                resolver=resolver,
                batch_rows=2,
                expected_pending_rows=2,
            )

    assert receipt["counts"]["completed_rows"] == 2
    assert receipt["counts"]["status_counts"] == {"resolved": 2}
    assert receipt["counts"]["attempt_rows"] == 6
    assert receipt["counts"]["network_attempt_rows"] == 4
    assert all(receipt["validation"].values())
    results = pq.read_table(
        receipt["artifacts"]["result_shard"]["path"]
    ).to_pylist()
    assert [row["stable_candidate_url"] for row in results] == [
        "https://images.test/media/1.png",
        "https://images.test/media/2.png",
    ]
    assert {
        row["adapter_version"] for row in results
    } == {SAME_HOST_MIXED_CONTENT_POLICY_VERSION}
    attempts = pq.read_table(
        receipt["artifacts"]["attempt_shard"]["path"]
    ).to_pylist()
    rewrites = [
        row for row in attempts if row["phase"] == "candidate_normalization"
    ]
    assert len(rewrites) == 2
    assert all(row["outcome"] == "rewritten" for row in rewrites)
    assert all(
        row["requested_url"].startswith("http://")
        and row["response_url"].startswith("https://")
        for row in rewrites
    )


def test_image_signature_sniffing_requires_recognised_bytes() -> None:
    assert sniff_image_content_type(b"\xff\xd8\xff\xe0payload") == "image/jpeg"
    assert sniff_image_content_type(b"\x89PNG\r\n\x1a\npayload") == "image/png"
    assert sniff_image_content_type(b"<html>not an image</html>") is None


def test_resolver_rejects_decompression_bomb_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=_image_bytes("PNG"),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result, _attempts = MediaURLResolver(
            config=ResolverConfig(max_attempts=1),
            http_client=client,
            resolve_host=PUBLIC_DNS,
        ).resolve(_input())

    assert result.status is ResolutionStatus.UNRESOLVED_INVALID_IMAGE
    assert result.terminal_reason == "image_probe_decoder_rejected"


def test_resolver_follows_redirect_and_validates_structured_candidate() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/record/123":
            return httpx.Response(302, headers={"Location": "/page/123"})
        if request.url.path == "/page/123":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html"},
                content=b'<meta property="og:image" content="/images/123.jpg">',
            )
        if request.url.path == "/images/123.jpg":
            return httpx.Response(
                200,
                headers={"Content-Type": "image/jpeg", "ETag": '"abc"'},
                content=_image_bytes("JPEG"),
            )
        raise AssertionError(request.url)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resolver = MediaURLResolver(
            config=ResolverConfig(max_attempts=1),
            http_client=client,
            resolve_host=PUBLIC_DNS,
        )
        result, attempts = resolver.resolve(_input())

    assert result.status is ResolutionStatus.RESOLVED
    assert result.stable_candidate_url == "https://example.org/images/123.jpg"
    assert result.validated_final_url == "https://example.org/images/123.jpg"
    assert result.detected_content_type == "image/jpeg"
    assert result.content_sha256 is None
    assert result.content_hash_status == "deferred"
    assert result.probe_prefix_sha256.startswith("sha256:")
    assert result.redirect_count == 1
    assert len(attempts) == 3
    assert attempts[-1].response_prefix_sha256 is not None
    assert seen == [
        "https://example.org/record/123",
        "https://example.org/page/123",
        "https://example.org/images/123.jpg",
    ]


def test_resolver_detects_redirect_loop_before_hop_budget() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.gbif.org":
            return httpx.Response(200, json={"key": 123, "media": []})
        if request.url.path == "/record/123":
            return httpx.Response(302, headers={"Location": "/loop"})
        if request.url.path == "/loop":
            return httpx.Response(302, headers={"Location": "/record/123"})
        raise AssertionError(request.url)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result, attempts = MediaURLResolver(
            config=ResolverConfig(max_attempts=1, max_redirects=10),
            http_client=client,
            resolve_host=PUBLIC_DNS,
        ).resolve(_input())

    assert result.terminal_reason == "redirect_loop_detected"
    assert len(attempts) == 3


def test_rights_blocked_and_youtube_make_zero_requests() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be used")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resolver = MediaURLResolver(http_client=client, resolve_host=PUBLIC_DNS)
        blocked, blocked_attempts = resolver.resolve(
            _input(media_license="Copyright")
        )
        youtube, youtube_attempts = resolver.resolve(
            _input(media_references="https://youtube.com/watch?v=abc")
        )

    assert blocked.status is ResolutionStatus.RIGHTS_BLOCKED
    assert youtube.status is ResolutionStatus.NON_IMAGE_MEDIA
    assert blocked_attempts == ()
    assert youtube_attempts == ()
    assert calls == 0


def test_retry_is_recorded_before_success() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            headers={"Content-Type": "image/gif"},
            content=_image_bytes("GIF"),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result, attempts = MediaURLResolver(
            config=ResolverConfig(max_attempts=2),
            http_client=client,
            resolve_host=PUBLIC_DNS,
            sleep=lambda _seconds: None,
        ).resolve(_input())
    assert result.status is ResolutionStatus.RESOLVED
    assert [attempt.outcome for attempt in attempts] == ["retry", "received"]


def test_ambiguous_or_spoofed_structured_candidates_fail_closed() -> None:
    for page, candidate_response, expected in (
        (
            b'<meta property="og:image" content="/a.jpg"><meta property="og:image" content="/b.jpg">',
            None,
            ResolutionStatus.UNRESOLVED_AMBIGUOUS_CANDIDATES,
        ),
        (
            b'<meta property="og:image" content="/a.jpg?X-Amz-Signature=secret">',
            None,
            ResolutionStatus.UNRESOLVED_INVALID_IMAGE,
        ),
        (
            b'<meta property="og:image" content="/a.jpg">',
            httpx.Response(
                200,
                headers={"Content-Type": "image/png"},
                content=_image_bytes("JPEG"),
            ),
            ResolutionStatus.UNRESOLVED_INVALID_IMAGE,
        ),
    ):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "api.gbif.org":
                return httpx.Response(200, json={"key": 123, "media": []})
            if request.url.path == "/record/123":
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/html"},
                    content=page,
                )
            if candidate_response is not None and request.url.path == "/a.jpg":
                return candidate_response
            raise AssertionError(request.url)

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result, _ = MediaURLResolver(
                config=ResolverConfig(max_attempts=1),
                http_client=client,
                resolve_host=PUBLIC_DNS,
            ).resolve(_input())
        assert result.status is expected


def test_flickr_adapter_recovers_after_reference_access_denied() -> None:
    reference = "https://www.flickr.com/photos/example/987654321"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/photos/example/987654321":
            return httpx.Response(403)
        if request.url.path == "/services/oembed/":
            return httpx.Response(
                200,
                json={"type": "photo", "url": "https://live.staticflickr.com/1/987654321_abcd_b.jpg"},
            )
        if request.url.host == "live.staticflickr.com":
            return httpx.Response(
                200,
                headers={"Content-Type": "image/jpeg"},
                content=_image_bytes("JPEG"),
            )
        raise AssertionError(request.url)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result, attempts = MediaURLResolver(
            config=ResolverConfig(max_attempts=1),
            http_client=client,
            resolve_host=PUBLIC_DNS,
        ).resolve(_input(media_references=reference))
    assert result.status is ResolutionStatus.RESOLVED
    assert result.method == "flickr_oembed"
    assert result.adapter_version == "flickr-oembed-adapter/v1"
    assert len(attempts) == 3


def test_prepare_worker_finalize_and_publish_v4(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    source_manifest = tmp_path / "source_manifest.json"
    runtime = tmp_path / "runtime"
    sidecars = tmp_path / "resolution-v1"
    v4 = tmp_path / "v4"
    state = SQLiteWorkStore(tmp_path / "state.sqlite")
    _write_source(source)
    source_manifest.write_text("{}\n", encoding="utf-8")
    pilot_acceptance = _write_pilot_acceptance(tmp_path, source)

    prepared = prepare_resolution(
        source=source,
        source_manifest=source_manifest,
        output_root=runtime,
        workstore=state,
        run_id="fixture-run",
        expected_missing_rows=3,
        enqueue_batch_rows=2,
        mode="full",
        pilot_acceptance_manifest=pilot_acceptance,
        resolver_config=ResolverConfig(max_attempts=1),
    )
    assert prepared["input_rows"] == 3
    assert prepared["rights_blocked_rows"] == 1
    assert prepared["pilot_acceptance_manifest_sha256"]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.gbif.org":
            return httpx.Response(200, json={"key": 1, "media": []})
        if request.url.path == "/record/1":
            return httpx.Response(
                200,
                headers={"Content-Type": "image/png"},
                content=_image_bytes("PNG"),
            )
        raise AssertionError(request.url)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        worker = run_worker(
            workstore=state,
            output_root=runtime,
            run_id="fixture-run",
            worker_id="worker-1",
            batch_rows=10,
            resolver=MediaURLResolver(
                config=ResolverConfig(max_attempts=1),
                http_client=client,
                resolve_host=PUBLIC_DNS,
            ),
        )
    assert worker["completed_rows"] == 3

    manifest = finalize_resolution(
        workstore=state,
        run_id="fixture-run",
        output_root=runtime,
        output_directory=sidecars,
        expected_rows=3,
    )
    assert manifest["counts"]["result_rows"] == 3
    assert manifest["counts"]["rights_blocked_rows"] == 1
    assert (sidecars / "manifest.json").is_file()
    results = pq.read_table(sidecars / "resolution_results.parquet")
    assert results.num_rows == 3

    v4_manifest = publish_v4(
        source=source,
        source_manifest=source_manifest,
        resolution_directory=sidecars,
        output_directory=v4,
        batch_rows=2,
    )
    assert v4_manifest["counts"]["source_rows"] == 4
    assert v4_manifest["counts"]["rights_blocked_rows_excluded"] == 1
    assert v4_manifest["counts"]["output_rows"] == 3
    output = pq.read_table(v4 / "gbif_media.parquet")
    assert output.column("effective_media_identifier").to_pylist() == [
        "https://example.org/record/1",
        "https://example.org/existing.jpg",
        None,
    ]
    con = duckdb.connect(str(v4 / "gbif_media.duckdb"), read_only=True)
    try:
        indexes = {row[0] for row in con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
    finally:
        con.close()
    assert {
        "idx_gbif_media_effective_identifier",
        "idx_gbif_media_media_identifier",
        "idx_gbif_media_gbif_id",
    } <= indexes
    assert json.loads((v4 / "manifest.json").read_text())["manifest_policy"][
        "written_last"
    ]

    result_shard = Path(str(worker["result_shard"]))
    original_result_bytes = result_shard.read_bytes()
    with result_shard.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(
        RuntimeError,
        match="result shard checksum mismatch",
    ):
        finalize_resolution(
            workstore=state,
            run_id="fixture-run",
            output_root=runtime,
            output_directory=tmp_path / "tampered-resolution",
            expected_rows=3,
        )
    result_shard.write_bytes(original_result_bytes)

    attempt_shard = Path(str(worker["attempt_shard"]))
    original_attempt_bytes = attempt_shard.read_bytes()
    with attempt_shard.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(
        RuntimeError,
        match="attempt shard checksum mismatch",
    ):
        finalize_resolution(
            workstore=state,
            run_id="fixture-run",
            output_root=runtime,
            output_directory=tmp_path / "tampered-attempt-resolution",
            expected_rows=3,
        )
    attempt_shard.write_bytes(original_attempt_bytes)

    orphan_shard = runtime / "shards" / "results" / "orphan.parquet"
    orphan_shard.write_bytes(original_result_bytes)
    state.register_shard(
        shard_id="orphan",
        job_name=JOB_NAME,
        registry_version=prepared["source_artifact_sha256"],
        stage=STAGE,
        run_id="fixture-run",
        worker_id="expired-worker",
        uri=str(orphan_shard),
        checksum=str(worker["result_sha256"]),
        row_count=3,
        byte_count=orphan_shard.stat().st_size,
        metadata={
            "attempt_uri": str(attempt_shard),
            "attempt_sha256": str(worker["attempt_sha256"]),
            "attempt_rows": pq.ParquetFile(attempt_shard).metadata.num_rows,
        },
    )
    with pytest.raises(
        RuntimeError,
        match="unreferenced or unregistered result shards",
    ):
        finalize_resolution(
            workstore=state,
            run_id="fixture-run",
            output_root=runtime,
            output_directory=tmp_path / "unreferenced-resolution",
            expected_rows=3,
        )


def test_import_pilot_cache_is_checksum_bound_idempotent_and_mixes_prior_results(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    source_manifest = tmp_path / "source_manifest.json"
    runtime = tmp_path / "runtime"
    sidecars = tmp_path / "resolution-full"
    state = SQLiteWorkStore(tmp_path / "state.sqlite")
    _write_source(source)
    source_manifest.write_text("{}\n", encoding="utf-8")
    pilot_acceptance = _write_pilot_acceptance(tmp_path, source)
    prepared = prepare_resolution(
        source=source,
        source_manifest=source_manifest,
        output_root=runtime,
        workstore=state,
        run_id="cache-full-run",
        expected_missing_rows=3,
        mode="full",
        pilot_acceptance_manifest=pilot_acceptance,
        resolver_config=ResolverConfig(max_attempts=1),
    )

    pilot_results = pq.read_table(
        tmp_path / "pilot-resolution" / "resolution_results.parquet"
    )
    pilot_attempts = pq.read_table(
        tmp_path / "pilot-resolution" / "resolution_attempts.parquet"
    )
    preexisting_id = str(pilot_results["source_row_id"][0].as_py())
    preexisting_result = runtime / "shards" / "results" / "preexisting.parquet"
    preexisting_attempt = runtime / "shards" / "attempts" / "preexisting.parquet"
    preexisting_result.parent.mkdir(parents=True)
    preexisting_attempt.parent.mkdir(parents=True)
    pq.write_table(
        pilot_results.filter(
            pc.equal(pilot_results["source_row_id"], preexisting_id)
        ),
        preexisting_result,
    )
    pq.write_table(
        pilot_attempts.filter(
            pc.equal(pilot_attempts["source_row_id"], preexisting_id)
        ),
        preexisting_attempt,
    )
    result_sha = "sha256:" + hashlib.sha256(
        preexisting_result.read_bytes()
    ).hexdigest()
    attempt_sha = "sha256:" + hashlib.sha256(
        preexisting_attempt.read_bytes()
    ).hexdigest()
    state.register_shard(
        shard_id="preexisting",
        job_name=JOB_NAME,
        registry_version=prepared["source_artifact_sha256"],
        stage=STAGE,
        run_id="cache-full-run",
        worker_id="network-worker-before-cache-fix",
        uri=str(preexisting_result),
        checksum=result_sha,
        row_count=1,
        byte_count=preexisting_result.stat().st_size,
        metadata={
            "attempt_uri": str(preexisting_attempt),
            "attempt_sha256": attempt_sha,
            "attempt_rows": pq.ParquetFile(preexisting_attempt).metadata.num_rows,
        },
    )
    assert state.complete_pending(
        preexisting_id,
        output_uri=str(preexisting_result),
        checksum=result_sha,
        row_count=1,
    )

    receipt = import_pilot_cache(
        workstore=state,
        run_id="cache-full-run",
        output_root=runtime,
    )
    assert receipt["network_requests"] == 0
    assert receipt["counts"]["pilot_result_rows"] == 3
    assert receipt["counts"]["cache_completed_rows"] == 2
    assert receipt["counts"]["preexisting_full_completions"] == 1
    assert receipt["counts"]["duplicate_network_probe_rows"] == 1
    assert pq.read_table(
        receipt["artifacts"]["preexisting_full_completions"]["path"]
    ).num_rows == 1
    assert (
        import_pilot_cache(
            workstore=state,
            run_id="cache-full-run",
            output_root=runtime,
        )
        == receipt
    )

    manifest = finalize_resolution(
        workstore=state,
        run_id="cache-full-run",
        output_root=runtime,
        output_directory=sidecars,
        expected_rows=3,
    )
    assert manifest["counts"]["result_rows"] == 3
    assert manifest["counts"]["attempt_rows"] == 2
    assert manifest["counts"]["registered_result_rows"] == 4
    assert manifest["counts"]["selected_result_rows"] == 3
    assert manifest["counts"]["unselected_registered_result_rows"] == 1
    assert manifest["counts"]["registered_attempt_rows"] == 3
    assert manifest["counts"]["selected_attempt_rows"] == 2
    assert manifest["counts"]["unselected_registered_attempt_rows"] == 1


def test_import_pilot_cache_rejects_active_claims_before_publication(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    source_manifest = tmp_path / "source_manifest.json"
    runtime = tmp_path / "runtime"
    state = SQLiteWorkStore(tmp_path / "state.sqlite")
    _write_source(source)
    source_manifest.write_text("{}\n", encoding="utf-8")
    pilot_acceptance = _write_pilot_acceptance(tmp_path, source)
    prepared = prepare_resolution(
        source=source,
        source_manifest=source_manifest,
        output_root=runtime,
        workstore=state,
        run_id="cache-claimed-run",
        expected_missing_rows=3,
        mode="full",
        pilot_acceptance_manifest=pilot_acceptance,
        resolver_config=ResolverConfig(max_attempts=1),
    )
    assert state.claim_next_batch(
        "active-worker",
        1,
        job_name=JOB_NAME,
        stage=STAGE,
        registry_version=prepared["source_artifact_sha256"],
    )

    with pytest.raises(RuntimeError, match="zero active claims"):
        import_pilot_cache(
            workstore=state,
            run_id="cache-claimed-run",
            output_root=runtime,
        )
    assert not (runtime / "shards").exists()


def test_import_pilot_cache_rejects_resolution_checksum_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    source_manifest = tmp_path / "source_manifest.json"
    runtime = tmp_path / "runtime"
    state = SQLiteWorkStore(tmp_path / "state.sqlite")
    _write_source(source)
    source_manifest.write_text("{}\n", encoding="utf-8")
    pilot_acceptance = _write_pilot_acceptance(tmp_path, source)
    prepare_resolution(
        source=source,
        source_manifest=source_manifest,
        output_root=runtime,
        workstore=state,
        run_id="cache-checksum-run",
        expected_missing_rows=3,
        mode="full",
        pilot_acceptance_manifest=pilot_acceptance,
        resolver_config=ResolverConfig(max_attempts=1),
    )
    result_path = tmp_path / "pilot-resolution" / "resolution_results.parquet"
    result_path.write_bytes(result_path.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="checksum mismatch"):
        import_pilot_cache(
            workstore=state,
            run_id="cache-checksum-run",
            output_root=runtime,
        )
    assert not (runtime / "shards").exists()


def test_host_fair_schedule_emits_full_origin_chunks_before_remainders() -> None:
    inputs = [
        _input(
            source_row_id=f"{host}-{offset}",
            media_references=f"https://{host}.example/record/{offset}",
        )
        for host, count in (("a", 5), ("b", 4), ("c", 2))
        for offset in range(count)
    ]

    scheduled = host_fair_schedule(inputs, chunk_rows=2)
    hosts = [item.host for item in scheduled]

    assert hosts == [
        "a.example",
        "a.example",
        "b.example",
        "b.example",
        "c.example",
        "c.example",
        "a.example",
        "a.example",
        "b.example",
        "b.example",
        "a.example",
    ]
    assert [item.source_row_id for item in scheduled] == [
        "a-0",
        "a-1",
        "b-0",
        "b-1",
        "c-0",
        "c-1",
        "a-2",
        "a-3",
        "b-2",
        "b-3",
        "a-4",
    ]


def test_rebalance_resolution_queue_is_idempotent_and_auditable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    source_manifest = tmp_path / "source_manifest.json"
    runtime = tmp_path / "runtime"
    state = SQLiteWorkStore(tmp_path / "state.sqlite")
    _write_source(source)
    source_manifest.write_text("{}\n", encoding="utf-8")
    pilot_acceptance = _write_pilot_acceptance(tmp_path, source)
    prepared = prepare_resolution(
        source=source,
        source_manifest=source_manifest,
        output_root=runtime,
        workstore=state,
        run_id="rebalance-run",
        expected_missing_rows=3,
        mode="full",
        pilot_acceptance_manifest=pilot_acceptance,
        resolver_config=ResolverConfig(max_attempts=1),
    )

    receipt = rebalance_resolution_queue(
        workstore=state,
        run_id="rebalance-run",
        output_root=runtime,
        chunk_rows=1,
    )
    assert receipt["network_requests"] == 0
    assert receipt["counts"] == {
        "origin_hosts": 2,
        "pending_rows": 3,
        "schedule_chunks": 3,
    }
    assignments = pq.read_table(
        receipt["artifacts"]["schedule_assignments"]["path"]
    ).sort_by("schedule_rank")
    assert assignments["origin_host"].to_pylist() == [
        "example.org",
        "youtube.com",
        "example.org",
    ]
    assert (
        rebalance_resolution_queue(
            workstore=state,
            run_id="rebalance-run",
            output_root=runtime,
            chunk_rows=1,
        )
        == receipt
    )

    first = state.claim_next_batch(
        "worker-1",
        1,
        job_name=JOB_NAME,
        stage=STAGE,
        registry_version=prepared["source_artifact_sha256"],
    )
    second = state.claim_next_batch(
        "worker-2",
        1,
        job_name=JOB_NAME,
        stage=STAGE,
        registry_version=prepared["source_artifact_sha256"],
    )
    assert ResolutionInput.from_payload(first[0]["payload"]).host == "example.org"
    assert ResolutionInput.from_payload(second[0]["payload"]).host == "youtube.com"
    with pytest.raises(RuntimeError, match="zero active claims"):
        rebalance_resolution_queue(
            workstore=state,
            run_id="rebalance-run",
            output_root=runtime,
            chunk_rows=1,
        )


@pytest.mark.parametrize("occurrence_as_extension", [False, True])
def test_archive_circuit_completes_exact_reference_bound_rows_without_network(
    tmp_path: Path,
    occurrence_as_extension: bool,
) -> None:
    runtime = tmp_path / "runtime"
    state = SQLiteWorkStore(tmp_path / "state.sqlite")
    state.get_or_create_run(
        job_name=JOB_NAME,
        stage=STAGE,
        run_id="archive-run",
        registry_version="sha256:source",
        config={"output_root": str(runtime.resolve())},
    )
    provider = "Reference-only Provider"
    inputs = [
        _input(
            source_row_id=f"source-{index}",
            gbif_id=str(index),
            media_references=f"https://provider.test/record/{index}.html",
            provider=provider if index < 3 else "Other Provider",
        )
        for index in range(1, 4)
    ]
    state.enqueue_work(
        JOB_NAME,
        "sha256:source",
        [
            {"work_key": item.source_row_id, **item.to_payload()}
            for item in inputs
        ],
        stage=STAGE,
    )

    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive_path = archive_root / "provider.zip"
    if occurrence_as_extension:
        meta = """\
<archive xmlns="http://rs.tdwg.org/dwc/text/">
  <core encoding="UTF-8" fieldsTerminatedBy="\\t" ignoreHeaderLines="1"
        rowType="http://rs.tdwg.org/dwc/terms/Event">
    <files><location>event.txt</location></files>
    <id index="0"/>
  </core>
  <extension encoding="UTF-8" fieldsTerminatedBy="\\t" ignoreHeaderLines="1"
             rowType="http://rs.tdwg.org/dwc/terms/Occurrence">
    <files><location>occurrence.txt</location></files>
    <coreid index="0"/>
    <field index="1" term="http://rs.tdwg.org/dwc/terms/occurrenceID"/>
    <field index="2" term="http://rs.tdwg.org/dwc/terms/associatedMedia"/>
  </extension>
</archive>
"""
        occurrence_text = (
            "coreid\toccurrenceID\tassociatedMedia\n"
            "event-1\tarchive-1\thttps://provider.test/record/1.html\n"
            "event-2\tarchive-2\thttps://provider.test/record/2.html\n"
        )
    else:
        meta = """\
<archive xmlns="http://rs.tdwg.org/dwc/text/">
  <core encoding="UTF-8" fieldsTerminatedBy="\\t" ignoreHeaderLines="1"
        rowType="http://rs.tdwg.org/dwc/terms/Occurrence">
    <files><location>occurrence.txt</location></files>
    <id index="0"/>
    <field index="1" term="http://rs.tdwg.org/dwc/terms/associatedMedia"/>
  </core>
</archive>
"""
        occurrence_text = (
            "id\tassociatedMedia\n"
            "archive-1\thttps://provider.test/record/1.html\n"
            "archive-2\thttps://provider.test/record/2.html\n"
        )
    with zipfile.ZipFile(archive_path, "w") as bundle:
        bundle.writestr("meta.xml", meta)
        if occurrence_as_extension:
            bundle.writestr("event.txt", "id\nevent-1\nevent-2\n")
        bundle.writestr("occurrence.txt", occurrence_text)
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest_path = archive_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "archives": [
                    {
                        "provider": provider,
                        "dataset_key": "dataset-1",
                        "source_url": "https://provider.test/archive.zip",
                        "path": archive_path.name,
                        "physical_bytes": archive_path.stat().st_size,
                        "sha256": archive_sha,
                        "intake_status": "PASS",
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = complete_archive_reference_only_rows(
        workstore=state,
        run_id="archive-run",
        output_root=runtime,
        archive_manifest=manifest_path,
        provider=provider,
        dataset_key="dataset-1",
        expected_pending_rows=2,
    )

    assert receipt["schema_version"] == ARCHIVE_CIRCUIT_VERSION
    assert receipt["network_requests"] == 0
    assert receipt["terminal_reason"] == TERMINAL_REASON
    assert receipt["counts"]["completed_rows"] == 2
    assert all(receipt["validation"].values())
    result_path = Path(receipt["artifacts"]["result_shard"]["path"])
    results = pq.read_table(result_path)
    assert results["source_row_id"].to_pylist() == ["source-1", "source-2"]
    assert set(results["status"].to_pylist()) == {
        ResolutionStatus.UNRESOLVED_ARCHIVE_REFERENCE_ONLY.value
    }
    assert set(results["attempt_count"].to_pylist()) == {0}
    attempts = pq.read_table(receipt["artifacts"]["attempt_shard"]["path"])
    assert attempts.num_rows == 0
    bindings = pq.read_table(
        Path(receipt["artifacts"]["archive_reference_bindings"]["path"])
        if Path(
            receipt["artifacts"]["archive_reference_bindings"]["path"]
        ).is_absolute()
        else (
            runtime
            / "archive_circuits"
            / Path(receipt["artifacts"]["result_shard"]["path"]).stem.removeprefix(
                "archive-"
            )
            / receipt["artifacts"]["archive_reference_bindings"]["path"]
        )
    )
    assert bindings["archive_occurrence_id"].to_pylist() == [
        "archive-1",
        "archive-2",
    ]
    statuses = {
        item["work_key"]: item["status"]
        for item in state.list_work_items(
            job_name=JOB_NAME,
            stage=STAGE,
            registry_version="sha256:source",
        )
    }
    assert statuses == {
        "source-1": "completed",
        "source-2": "completed",
        "source-3": "pending",
    }


def test_worker_renews_every_claim_before_shard_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.parquet"
    source_manifest = tmp_path / "source_manifest.json"
    runtime = tmp_path / "runtime"
    state = SQLiteWorkStore(tmp_path / "state.sqlite")
    _write_source(source)
    source_manifest.write_text("{}\n", encoding="utf-8")
    pilot_acceptance = _write_pilot_acceptance(tmp_path, source)
    prepare_resolution(
        source=source,
        source_manifest=source_manifest,
        output_root=runtime,
        workstore=state,
        run_id="lease-renewal-run",
        expected_missing_rows=3,
        mode="full",
        pilot_acceptance_manifest=pilot_acceptance,
        resolver_config=ResolverConfig(max_attempts=1),
    )

    renewed: list[str] = []
    real_renew_claim = state.renew_claim

    def recording_renew_claim(work_key: str, **kwargs: object) -> bool:
        renewed.append(work_key)
        return real_renew_claim(work_key, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(state, "renew_claim", recording_renew_claim)
    def delayed_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/record/1":
            time.sleep(1.1)
            return httpx.Response(
                200,
                headers={"Content-Type": "image/png"},
                content=_image_bytes("PNG"),
            )
        return httpx.Response(200, json={"key": 1, "media": []})

    with httpx.Client(transport=httpx.MockTransport(delayed_handler)) as client:
        worker = run_worker(
            workstore=state,
            output_root=runtime,
            run_id="lease-renewal-run",
            worker_id="worker-lease",
            batch_rows=10,
            stale_after_seconds=1,
            resolver=MediaURLResolver(
                config=ResolverConfig(max_attempts=1),
                http_client=client,
                resolve_host=PUBLIC_DNS,
            ),
        )

    assert worker["completed_rows"] == 3
    assert len(renewed) >= 6


def test_worker_does_not_publish_shard_after_lease_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.parquet"
    source_manifest = tmp_path / "source_manifest.json"
    runtime = tmp_path / "runtime"
    state = SQLiteWorkStore(tmp_path / "state.sqlite")
    _write_source(source)
    source_manifest.write_text("{}\n", encoding="utf-8")
    pilot_acceptance = _write_pilot_acceptance(tmp_path, source)
    prepare_resolution(
        source=source,
        source_manifest=source_manifest,
        output_root=runtime,
        workstore=state,
        run_id="lease-loss-run",
        expected_missing_rows=3,
        mode="full",
        pilot_acceptance_manifest=pilot_acceptance,
        resolver_config=ResolverConfig(max_attempts=1),
    )
    monkeypatch.setattr(state, "renew_claim", lambda *_args, **_kwargs: False)

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"Content-Type": "image/png"},
                content=_image_bytes("PNG"),
            )
            if request.url.path == "/record/1"
            else httpx.Response(200, json={"key": 1, "media": []})
        )
    ) as client:
        with pytest.raises(
            RuntimeError, match="claim lease was lost before shard publication"
        ):
            run_worker(
                workstore=state,
                output_root=runtime,
                run_id="lease-loss-run",
                worker_id="worker-expired",
                batch_rows=10,
                resolver=MediaURLResolver(
                    config=ResolverConfig(max_attempts=1),
                    http_client=client,
                    resolve_host=PUBLIC_DNS,
                ),
            )

    assert not (runtime / "shards").exists()
