from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import time

import duckdb
import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from biominer.cli import build_parser
from biominer.gbif_media_resolution.cli import COMMAND, run_gbif_media_resolution_command

from biominer.gbif_media_resolution.models import (
    ResolutionInput,
    ResolutionStatus,
    source_row_id,
)
from biominer.gbif_media_resolution.pipeline import (
    finalize_resolution,
    prepare_resolution,
    publish_v4,
    pilot_selection_table,
    run_worker,
    select_pilot_inputs,
)
from biominer.gbif_media_resolution.resolver import (
    MediaURLResolver,
    ResolverConfig,
    extract_structured_image_candidates,
    sniff_image_content_type,
    validate_public_http_url,
    redact_url_for_audit,
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
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = directory / "pilot-acceptance.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "biominer-gbif-media-url-pilot-execution-audit/v1",
                "source_snapshot_id": "sha256:" + source_sha,
                "overall_acceptance_status": "PASS",
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
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


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


def test_structured_html_extraction_does_not_scrape_generic_images() -> None:
    html = b"""
      <html><head>
      <meta property="og:image" content="/media/specimen.jpg">
      </head><body><img src="/logo.png"><img src="/unrelated.jpg"></body></html>
    """
    assert extract_structured_image_candidates(
        html, base_url="https://example.org/record/1"
    ) == ("https://example.org/media/specimen.jpg",)


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
