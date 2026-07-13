from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from io import BytesIO
from pathlib import Path
import socket
import ssl
import threading
import time
import tomllib

import httpcore
import httpx
from PIL import Image
import polars as pl
import pytest

from biominer.references.downloader import (
    _HostValidator,
    _OriginLimiterRegistry,
    _PermanentResponse,
    _PinnedAddressHTTPTransport,
    _PinnedAddressNetworkBackend,
    _decode_image_isolated,
    ProviderMediaDownloadPolicy,
    ReferenceMediaDownloadConfig,
    download_reference_media,
)
from biominer.references.licensing import (
    ReferenceLicencePolicy,
    canonicalise_creative_commons_licence,
)
from biominer.references.schemas import (
    REFERENCE_ACQUISITION_SELECTIONS_SCHEMA_VERSION,
    REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
    make_reference_media_id,
    make_reference_observation_id,
    make_reference_selection_id,
    reference_acquisition_selections_frame,
    reference_media_candidates_frame,
)
from biominer.storage.local import LocalStorageBackend


_NOW = datetime(2026, 7, 14, 3, 0, tzinfo=timezone.utc)
_PLAN_FINGERPRINT = "sha256:" + "a" * 64


def _image_bytes(
    image_format: str,
    *,
    size: tuple[int, int] = (2, 2),
    color: tuple[int, int, int] = (10, 20, 30),
) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color=color).save(output, format=image_format)
    return output.getvalue()


def _animated_gif_bytes() -> bytes:
    output = BytesIO()
    first = Image.new("RGB", (2, 2), color=(10, 20, 30))
    second = Image.new("RGB", (2, 2), color=(30, 20, 10))
    first.save(
        output,
        format="GIF",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )
    return output.getvalue()


class _MemoryStorage:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.json: dict[str, dict[str, object]] = {}
        self.parquet: dict[str, pl.DataFrame] = {}
        self.text: dict[str, str] = {}
        self.operations: list[tuple[str, str]] = []
        self.fail_checkpoint = False
        self.fail_summary = False

    def write_file(
        self,
        uri: str,
        source: str | Path,
        *,
        content_type: str | None = None,
        overwrite: bool = True,
    ) -> str:
        self.operations.append(("file", uri))
        if not overwrite and uri in self.files:
            raise FileExistsError(uri)
        self.files[uri] = Path(source).read_bytes()
        return uri

    def write_json(self, uri: str, payload: dict[str, object]) -> str:
        self.operations.append(("json", uri))
        if self.fail_checkpoint and "/checkpoints/" in uri:
            raise OSError("checkpoint unavailable")
        self.json[uri] = deepcopy(payload)
        return uri

    def read_json(self, uri: str) -> dict[str, object]:
        return deepcopy(self.json[uri])

    def write_parquet_shard(
        self,
        uri: str,
        frame: pl.DataFrame,
        *,
        compression: str | None = "zstd",
        overwrite: bool = True,
    ) -> str:
        self.operations.append(("parquet", uri))
        if not overwrite and uri in self.parquet:
            raise FileExistsError(uri)
        self.parquet[uri] = frame.clone()
        return uri

    def read_parquet(self, uri: str) -> pl.DataFrame:
        return self.parquet[uri].clone()

    def write_text(self, uri: str, text: str, *, encoding: str = "utf-8") -> str:
        self.operations.append(("text", uri))
        if self.fail_summary and uri.endswith("reference_media_download_summary.md"):
            raise OSError("summary unavailable")
        self.text[uri] = text
        return uri

    def exists(self, uri: str) -> bool:
        return (
            uri in self.files
            or uri in self.json
            or uri in self.parquet
            or uri in self.text
        )

    def file_size(self, uri: str) -> int:
        if uri not in self.files:
            raise FileNotFoundError(uri)
        return len(self.files[uri])

    def file_sha256(self, uri: str) -> str:
        if uri not in self.files:
            raise FileNotFoundError(uri)
        return "sha256:" + hashlib.sha256(self.files[uri]).hexdigest()


def test_pillow_is_a_direct_runtime_dependency_for_reference_decoding() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependency_names = {
        value.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0].casefold()
        for value in pyproject["project"]["dependencies"]
    }

    assert "pillow" in dependency_names


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"allowed_hosts": "media.example.test"}, "allowed_hosts"),
        ({"allowed_schemes": "https"}, "allowed_schemes"),
    ],
)
def test_provider_policy_requires_explicit_string_tuples(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "source": "TestProvider",
        "allowed_hosts": ("media.example.test",),
    }
    values.update(kwargs)
    with pytest.raises(TypeError, match=message):
        ProviderMediaDownloadPolicy(**values)


@pytest.mark.parametrize(
    ("retry_statuses", "error", "message"),
    [
        (("429",), TypeError, "tuple of integers"),
        ([429], TypeError, "tuple of integers"),
        ((200,), ValueError, "non-transient or unsupported"),
        ((404,), ValueError, "non-transient or unsupported"),
    ],
)
def test_download_config_rejects_invalid_retry_statuses(
    retry_statuses: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _config(retry_statuses=retry_statuses)


@pytest.mark.parametrize(
    ("value", "canonical"),
    [
        ("CC0 1.0", "cc0"),
        ("cc-by", "cc-by"),
        ("CC BY-NC 4.0", "cc-by-nc"),
        ("https://creativecommons.org/licenses/by-sa/3.0/legalcode", "cc-by-sa"),
        ("https://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
        ("ftp://creativecommons.org/licenses/by/4.0/", None),
        ("https://creativecommons.org/licenses/by/evil", None),
        ("https://creativecommons.org/publicdomain/zero/not-a-version", None),
        ("all rights reserved", None),
        ("https://example.org/licenses/by/4.0", None),
    ],
)
def test_canonicalises_only_known_creative_commons_licences(
    value: str,
    canonical: str | None,
) -> None:
    assert canonicalise_creative_commons_licence(value) == canonical


@pytest.mark.parametrize(
    ("licence", "licence_uri", "attribution", "status", "reason"),
    [
        ("cc0", None, None, "allowed", None),
        (
            "cc-by",
            "https://creativecommons.org/licenses/by/4.0/",
            "A. Observer / CC BY",
            "allowed",
            None,
        ),
        ("cc-by-nc", None, "A. Observer", "research_only", None),
        (
            "cc-by-nd",
            None,
            "A. Observer",
            "denied",
            "media_licence_not_allowed:cc-by-nd",
        ),
        (None, None, "A. Observer", "quarantined", "missing_media_licence"),
        ("custom", None, "A. Observer", "quarantined", "unrecognised_media_licence"),
        (
            "cc-by",
            "https://creativecommons.org/licenses/by-nc/4.0/",
            "A. Observer",
            "quarantined",
            "conflicting_media_licence",
        ),
        ("cc-by", None, None, "quarantined", "missing_required_attribution"),
    ],
)
def test_reference_licence_policy_is_fail_closed_and_separates_reuse(
    licence: str | None,
    licence_uri: str | None,
    attribution: str | None,
    status: str,
    reason: str | None,
) -> None:
    decision = ReferenceLicencePolicy().evaluate(
        media_licence=licence,
        licence_uri=licence_uri,
        attribution=attribution,
    )

    assert decision.status == status
    assert decision.reason == reason


def test_reference_licence_policy_rejects_overlapping_allowlists() -> None:
    with pytest.raises(ValueError, match="allowlists overlap"):
        ReferenceLicencePolicy(
            broadly_reusable=("cc-by",),
            research_only=("CC BY 4.0",),
            attribution_required=("cc-by",),
        )


def test_reference_licence_policy_supports_explicit_provider_aliases() -> None:
    policy = ReferenceLicencePolicy(
        broadly_reusable=("provider-open",),
        research_only=("provider-research",),
        attribution_required=("provider-open", "provider-research"),
        licence_aliases=(
            ("Provider Open Licence 2.0", "provider-open"),
            ("https://licences.example/research/1", "provider-research"),
        ),
    )

    assert (
        policy.evaluate(
            media_licence="Provider Open Licence 2.0",
            licence_uri=None,
            attribution="Observer",
        ).status
        == "allowed"
    )
    assert (
        policy.evaluate(
            media_licence=None,
            licence_uri="https://licences.example/research/1",
            attribution="Observer",
        ).status
        == "research_only"
    )
    assert (
        policy.evaluate(
            media_licence="Provider Unknown",
            licence_uri=None,
            attribution="Observer",
        ).status
        == "quarantined"
    )


def test_download_commits_exact_validated_source_then_checkpoint() -> None:
    payload = _image_bytes("PNG", size=(3, 2))
    candidates, selections = _frames(
        licence="cc-by-nc",
        licence_uri="https://creativecommons.org/licenses/by-nc/4.0/",
        attribution="A. Observer / CC BY-NC",
    )
    original_candidates = candidates.clone()
    original_selections = selections.clone()
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=payload,
        )

    storage = _MemoryStorage()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            run_id="run-1",
            now=lambda: _NOW,
        )

    assert candidates.equals(original_candidates)
    assert selections.equals(original_selections)
    assert requests == ["https://media.example.test/photos/101/original.png"]
    row = result.media_objects.to_dicts()[0]
    digest = hashlib.sha256(payload).hexdigest()
    assert row["decode_status"] == "valid"
    assert row["licence_policy_status"] == "research_only"
    assert row["content_type"] == "image/png"
    assert row["source_byte_count"] == len(payload)
    assert (row["decoded_width"], row["decoded_height"]) == (3, 2)
    assert row["sha256"] == f"sha256:{digest}"
    assert row["source_object_uri"].endswith(f"/{digest}.png")
    assert storage.files[row["source_object_uri"]] == payload
    object_index = storage.operations.index(("file", row["source_object_uri"]))
    checkpoint_index = next(
        index
        for index, operation in enumerate(storage.operations)
        if operation[0] == "json" and "/checkpoints/" in operation[1]
    )
    assert object_index < checkpoint_index
    checkpoint = storage.json[storage.operations[checkpoint_index][1]]
    evidence = checkpoint["download_evidence"]
    assert evidence["attribution"] == "A. Observer / CC BY-NC"
    assert evidence["licence"] == "cc-by-nc"
    assert result.report["counts"]["committed"] == 1
    assert result.report["counts"]["http_requests"] == 1
    report_run_id = "run-1-" + hashlib.sha256(b"run-1").hexdigest()[:12]
    assert result.report_uri == (
        f"s3://references/bank-v1/reports/run_id={report_run_id}/"
        "reference_media_download_report.json"
    )
    assert result.summary_uri == (
        f"s3://references/bank-v1/reports/run_id={report_run_id}/"
        "reference_media_download_summary.md"
    )
    settings = result.report["settings"]
    assert settings["allowed_content_types"] == [
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
    ]
    assert settings["retry_statuses"] == [408, 425, 429, 500, 502, 503, 504]
    assert settings["max_download_seconds"] == 300.0
    assert settings["licence_policy"]["research_only"] == [
        "cc-by-nc",
        "cc-by-nc-sa",
    ]
    assert settings["provider_policies"][0]["allowed_hosts"] == ["media.example.test"]
    assert settings["provider_policies"][0]["url_strategy"] == "direct"


def test_committed_checkpoint_resumes_without_http_or_object_overwrite() -> None:
    payload = _image_bytes("JPEG", size=(4, 3))
    candidates, selections = _frames(
        url="https://media.example.test/photos/101/original.jpg"
    )
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200, headers={"Content-Type": "image/jpeg"}, content=payload
        )

    storage = _MemoryStorage()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )
        file_writes = len([value for value in storage.operations if value[0] == "file"])
        second = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )

    assert request_count == 1
    assert (
        len([value for value in storage.operations if value[0] == "file"])
        == file_writes
    )
    assert second.report["counts"]["resumed"] == 1
    assert second.report["counts"]["http_requests"] == 0
    assert second.media_objects.equals(first.media_objects)


def test_committed_checkpoint_resume_does_not_require_live_dns() -> None:
    payload = _image_bytes("JPEG", size=(4, 3))
    candidates, selections = _frames(
        url="https://media.example.test/photos/101/original.jpg"
    )
    request_count = 0
    resolution_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200, headers={"Content-Type": "image/jpeg"}, content=payload
        )

    def public_resolution(_host: str) -> tuple[str, ...]:
        nonlocal resolution_count
        resolution_count += 1
        return ("93.184.216.34",)

    def fail_resolution(_host: str) -> tuple[str, ...]:
        nonlocal resolution_count
        resolution_count += 1
        raise socket.gaierror("publisher DNS is unavailable")

    storage = _MemoryStorage()
    policy = ProviderMediaDownloadPolicy(
        source="TestProvider",
        allowed_hosts=("media.example.test",),
        resolve_public_addresses=True,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(provider_policies=(policy,)),
            http_client=client,
            now=lambda: _NOW,
            resolve_host=public_resolution,
        )
        assert resolution_count == 1
        second = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(provider_policies=(policy,)),
            http_client=client,
            now=lambda: _NOW,
            resolve_host=fail_resolution,
        )

    assert resolution_count == 1
    assert request_count == 1
    assert second.report["counts"]["resumed"] == 1
    assert second.media_objects.equals(first.media_objects)


def test_report_paths_are_collision_resistant_for_default_and_sanitized_run_ids() -> (
    None
):
    candidates, selections = _frames()
    storage = _MemoryStorage()
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=_image_bytes("PNG"),
        )

    results = []
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        for explicit_run_id in ("A/B", "a_b", None, None):
            results.append(
                download_reference_media(
                    selections,
                    candidates,
                    storage=storage,
                    output_prefix="s3://references/bank-v1",
                    config=_config(),
                    http_client=client,
                    run_id=explicit_run_id,
                    now=lambda: _NOW,
                )
            )

    assert calls == 1
    report_uris = [result.report_uri for result in results]
    assert len(report_uris) == len(set(report_uris)) == 4
    assert all("/reports/run_id=" in uri for uri in report_uris)
    assert results[0].report["run_id"] == "A/B"
    assert results[1].report["run_id"] == "a_b"
    assert results[2].report["run_id"] != results[3].report["run_id"]


def test_long_run_id_uses_bounded_local_report_path(tmp_path: Path) -> None:
    candidates, selections = _frames()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=_image_bytes("PNG"),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=LocalStorageBackend(),
            output_prefix=str(tmp_path / "bank"),
            config=_config(),
            http_client=client,
            run_id="A" * 300,
            now=lambda: _NOW,
        )

    assert Path(result.report_uri).is_file()
    assert Path(result.summary_uri).is_file()
    assert max(len(part) for part in Path(result.report_uri).parts) <= 255


def test_summary_failure_never_publishes_complete_json_report() -> None:
    candidates, selections = _frames()
    storage = _MemoryStorage()
    storage.fail_summary = True

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=_image_bytes("PNG"),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OSError, match="summary unavailable"):
            download_reference_media(
                selections,
                candidates,
                storage=storage,
                output_prefix="s3://references/bank-v1",
                config=_config(),
                http_client=client,
                run_id="summary-failure",
                now=lambda: _NOW,
            )

    reports = [
        payload
        for payload in storage.json.values()
        if payload.get("schema_version") == "reference-media-download-report-v1"
    ]
    assert reports == []


@pytest.mark.parametrize(
    ("licence", "licence_uri", "attribution", "expected_status", "reason"),
    [
        (None, None, "Observer", "quarantined", "missing_media_licence"),
        ("unknown", None, "Observer", "quarantined", "unrecognised_media_licence"),
        (
            "https://[:::",
            None,
            "Observer",
            "quarantined",
            "unrecognised_media_licence",
        ),
        ("cc-by-nd", None, "Observer", "denied", "media_licence_not_allowed:cc-by-nd"),
        ("cc-by", None, None, "quarantined", "missing_required_attribution"),
    ],
)
def test_unsafe_licence_is_quarantined_without_network(
    licence: str | None,
    licence_uri: str | None,
    attribution: str | None,
    expected_status: str,
    reason: str,
) -> None:
    candidates, selections = _frames(
        licence=licence,
        licence_uri=licence_uri,
        attribution=attribution,
    )

    def fail_network(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("unsafe licences must be rejected before HTTP")

    storage = _MemoryStorage()
    with httpx.Client(transport=httpx.MockTransport(fail_network)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )

    row = result.media_objects.to_dicts()[0]
    assert row["licence_policy_status"] == expected_status
    assert row["decode_status"] == "not_attempted"
    assert row["quarantine_reason"] == reason
    assert not storage.files
    assert not [uri for uri in storage.json if "/checkpoints/" in uri]
    assert result.report["quarantine_reason_counts"] == {reason: 1}
    assert result.report["source_error_counts"] == {"TestProvider": 1}
    assert result.report["performance"]["request_seconds_p95"] == ("not_instrumented")


@pytest.mark.parametrize(
    ("headers", "payload", "decode_status", "reason"),
    [
        (
            {"Content-Type": "image/jpeg"},
            b"<!doctype html><html>upstream error</html>",
            "invalid_content_type",
            "html_payload_masquerading_as_image",
        ),
        (
            {"Content-Type": "text/html"},
            b"<html>error</html>",
            "invalid_content_type",
            "content_type_not_allowed:text/html",
        ),
        (
            {"Content-Type": "image/png"},
            _image_bytes("JPEG"),
            "invalid_content_type",
            "content_type_signature_mismatch:image/png:image/jpeg",
        ),
        (
            {"Content-Type": "image/png"},
            b"\x89PNG\r\n\x1a\ntruncated",
            "decode_failed",
            "image_decode_failed:OSError",
        ),
        ({}, _image_bytes("PNG"), "invalid_content_type", "missing_content_type"),
    ],
)
def test_rejects_mislabeled_html_mime_mismatch_and_undecodable_images(
    headers: dict[str, str],
    payload: bytes,
    decode_status: str,
    reason: str,
) -> None:
    candidates, selections = _frames()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers, content=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )

    row = result.media_objects.to_dicts()[0]
    assert row["decode_status"] == decode_status
    assert row["quarantine_reason"] == reason
    assert row["source_object_uri"] is None


def test_streamed_payload_limit_is_enforced_without_object_commit() -> None:
    payload = _image_bytes("PNG", size=(20, 20))
    candidates, selections = _frames()
    storage = _MemoryStorage()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Type": "image/png"}, content=payload
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(max_source_bytes=len(payload) - 1),
            http_client=client,
            now=lambda: _NOW,
        )

    row = result.media_objects.to_dicts()[0]
    assert row["decode_status"] == "download_failed"
    assert row["quarantine_reason"] == "source_payload_too_large"
    assert not storage.files


def test_multi_frame_image_is_quarantined_without_object_commit() -> None:
    candidates, selections = _frames()
    storage = _MemoryStorage()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "image/gif"},
            content=_animated_gif_bytes(),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )

    row = result.media_objects.to_dicts()[0]
    assert row["decode_status"] == "decode_failed"
    assert row["quarantine_reason"] == "image_decode_failed:ValueError"
    assert not storage.files


def test_retries_429_with_retry_after_then_commits() -> None:
    payload = _image_bytes("PNG")
    candidates, selections = _frames()
    calls = 0
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(
            200, headers={"Content-Type": "image/png"}, content=payload
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
            sleep=sleeps.append,
            monotonic=lambda: 0.0,
        )

    row = result.media_objects.to_dicts()[0]
    assert row["decode_status"] == "valid"
    assert row["download_attempt_count"] == 2
    assert result.report["counts"]["retries"] == 1
    assert calls == 2
    assert sleeps == [2.0]
    assert result.report["counts"]["retries"] == 1
    assert result.report["counts"]["rate_limit_events"] == 1


def test_retry_after_is_not_shortened_by_backoff_cap() -> None:
    candidates, selections = _frames()
    sleeps: list[float] = []
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"Retry-After": "600"})
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=_image_bytes("PNG"),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(
                backoff_cap_seconds=1.0,
                max_download_seconds=1_200.0,
            ),
            http_client=client,
            now=lambda: _NOW,
            sleep=sleeps.append,
            monotonic=lambda: 0.0,
        )

    assert result.media_objects["decode_status"].item() == "valid"
    assert sleeps == [600.0]


def test_naive_retry_after_http_date_uses_backoff_without_losing_request_count() -> (
    None
):
    candidates, selections = _frames()
    calls = 0
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                503,
                headers={"Retry-After": "Wed, 21 Oct 2037 07:28:00"},
            )
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=_image_bytes("PNG"),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(backoff_base_seconds=1.0),
            http_client=client,
            now=lambda: _NOW,
            sleep=sleeps.append,
        )

    assert result.media_objects["decode_status"].item() == "valid"
    assert result.media_objects["download_attempt_count"].item() == 2
    assert result.report["counts"]["http_requests"] == 2
    assert calls == 2
    assert len(sleeps) == 1
    assert 0.9 <= sleeps[0] <= 1.1


def test_retry_after_above_operational_limit_blocks_origin_without_sleeping() -> None:
    candidates, selections = _frames()
    sleeps: list[float] = []
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "61"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(max_retry_after_seconds=60.0),
            http_client=client,
            now=lambda: _NOW,
            sleep=sleeps.append,
            monotonic=lambda: 0.0,
        )

    row = result.media_objects.to_dicts()[0]
    assert row["decode_status"] == "download_failed"
    assert row["quarantine_reason"] == (
        "retry_after_exceeds_operational_limit_http_429"
    )
    assert calls == 1
    assert sleeps == []
    assert result.report["counts"]["rate_limit_events"] == 1


def test_origin_blocked_before_request_is_not_counted_as_http_attempt() -> None:
    first_candidates, first_selections = _frames(
        provider_media_id="101",
        observation_id="observation-101",
        acquisition_plan_id="plan-101",
    )
    second_candidates, second_selections = _frames(
        provider_media_id="202",
        observation_id="observation-202",
        url="https://media.example.test/photos/202/original.png",
        acquisition_plan_id="plan-202",
    )
    candidates = reference_media_candidates_frame(
        first_candidates.to_dicts() + second_candidates.to_dicts()
    )
    selections = reference_acquisition_selections_frame(
        first_selections.to_dicts() + second_selections.to_dicts()
    )
    policy = ProviderMediaDownloadPolicy(
        source="TestProvider",
        allowed_hosts=("media.example.test",),
        max_concurrent_per_origin=1,
        resolve_public_addresses=False,
    )
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, headers={"Retry-After": "61"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(
                workers=2,
                max_inflight=2,
                max_retry_after_seconds=60.0,
                provider_policies=(policy,),
            ),
            http_client=client,
            now=lambda: _NOW,
            sleep=lambda _seconds: None,
            monotonic=lambda: 0.0,
        )

    assert calls == 1
    assert sorted(result.media_objects["download_attempt_count"].to_list()) == [0, 1]
    assert result.report["counts"]["http_requests"] == 1


def test_permanent_http_error_is_not_retried() -> None:
    candidates, selections = _frames()
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )

    assert calls == 1
    row = result.media_objects.to_dicts()[0]
    assert row["download_attempt_count"] == 1
    assert row["quarantine_reason"] == "http_status_404"


def test_inaturalist_square_url_is_strictly_rewritten_to_approved_style() -> None:
    payload = _image_bytes("JPEG")
    candidates, selections = _frames(
        source="iNaturalist",
        provider_media_id="101",
        url="https://static.inaturalist.org/photos/101/square.jpg?123",
    )
    requests: list[str] = []
    policy = ProviderMediaDownloadPolicy(
        source="iNaturalist",
        allowed_hosts=("static.inaturalist.org",),
        resolve_public_addresses=False,
        url_strategy="inaturalist_photo",
        inaturalist_image_style="large",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200, headers={"Content-Type": "image/jpeg"}, content=payload
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(provider_policies=(policy,)),
            http_client=client,
            now=lambda: _NOW,
        )

    assert result.media_objects["decode_status"].item() == "valid"
    assert requests == ["https://static.inaturalist.org/photos/101/large.jpg?123"]


def test_gbif_publisher_host_requires_explicit_reviewed_policy() -> None:
    candidates, selections = _frames(
        source="GBIF",
        url="https://publisher.example.test/specimens/101.png",
    )
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=_image_bytes("PNG"),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        rejected = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=ReferenceMediaDownloadConfig(workers=1, max_inflight=1),
            http_client=client,
            now=lambda: _NOW,
        )
        allowed = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v2",
            config=ReferenceMediaDownloadConfig(
                workers=1,
                max_inflight=1,
                provider_policies=(
                    ProviderMediaDownloadPolicy(
                        source="GBIF",
                        allowed_hosts=("publisher.example.test",),
                        resolve_public_addresses=False,
                    ),
                ),
            ),
            http_client=client,
            now=lambda: _NOW,
        )

    rejected_row = rejected.media_objects.to_dicts()[0]
    assert rejected_row["decode_status"] == "not_attempted"
    assert rejected_row["quarantine_reason"] == "provider_policy_missing"
    assert allowed.media_objects["decode_status"].item() == "valid"
    assert calls == 1


@pytest.mark.parametrize(
    "url",
    [
        "https://static.inaturalist.org/photos/999/square.jpg",
        "https://static.inaturalist.org/assets/default.png",
        "https://unknown.example/photos/101/square.jpg",
    ],
)
def test_inaturalist_mismatched_placeholder_or_unknown_origin_is_not_fetched(
    url: str,
) -> None:
    candidates, selections = _frames(
        source="iNaturalist",
        provider_media_id="101",
        url=url,
    )
    policy = ProviderMediaDownloadPolicy(
        source="iNaturalist",
        allowed_hosts=("static.inaturalist.org",),
        resolve_public_addresses=False,
        url_strategy="inaturalist_photo",
    )

    def fail_network(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid iNaturalist media must not be fetched")

    with httpx.Client(transport=httpx.MockTransport(fail_network)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(provider_policies=(policy,)),
            http_client=client,
            now=lambda: _NOW,
        )

    assert result.media_objects["decode_status"].item() == "not_attempted"
    assert result.media_objects["source_object_uri"].item() is None


def test_redirect_to_unapproved_origin_is_rejected() -> None:
    candidates, selections = _frames()
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            302, headers={"Location": "https://private.example/image.png"}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )

    assert calls == 1
    row = result.media_objects.to_dicts()[0]
    assert row["decode_status"] == "download_failed"
    assert row["quarantine_reason"].startswith("redirect_provider_constraint:")


def test_private_provider_resolution_is_rejected_before_http() -> None:
    candidates, selections = _frames()
    policy = ProviderMediaDownloadPolicy(
        source="TestProvider",
        allowed_hosts=("media.example.test",),
        resolve_public_addresses=True,
    )

    def fail_network(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("a private provider address must not be fetched")

    with httpx.Client(transport=httpx.MockTransport(fail_network)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(provider_policies=(policy,)),
            http_client=client,
            now=lambda: _NOW,
            resolve_host=lambda _host: ("127.0.0.1",),
        )

    row = result.media_objects.to_dicts()[0]
    assert row["decode_status"] == "download_failed"
    assert "non-public address" in row["quarantine_reason"]


def test_provider_dns_failure_is_quarantined_without_aborting_batch() -> None:
    candidates, selections = _frames()
    policy = ProviderMediaDownloadPolicy(
        source="TestProvider",
        allowed_hosts=("media.example.test",),
        resolve_public_addresses=True,
    )

    def fail_resolution(_host: str) -> tuple[str, ...]:
        raise socket.gaierror("publisher DNS is unavailable")

    def fail_network(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("an unresolved provider host must not be fetched")

    with httpx.Client(transport=httpx.MockTransport(fail_network)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(provider_policies=(policy,)),
            http_client=client,
            now=lambda: _NOW,
            resolve_host=fail_resolution,
        )

    row = result.media_objects.to_dicts()[0]
    assert row["decode_status"] == "download_failed"
    assert row["quarantine_reason"] == (
        "provider_constraint:provider host resolution failed"
    )


class _RecordingNetworkBackend(httpcore.NetworkBackend):
    def __init__(self) -> None:
        self.connected_hosts: list[str] = []

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object = None,
    ) -> httpcore.NetworkStream:
        del port, timeout, local_address, socket_options
        self.connected_hosts.append(host)
        return object()  # type: ignore[return-value]

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: object = None,
    ) -> httpcore.NetworkStream:
        del path, timeout, socket_options
        raise AssertionError("UNIX sockets are forbidden")


class _FailingNetworkBackend(_RecordingNetworkBackend):
    def __init__(self) -> None:
        super().__init__()
        self.timeouts: list[float | None] = []

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object = None,
    ) -> httpcore.NetworkStream:
        del port, local_address, socket_options
        self.connected_hosts.append(host)
        self.timeouts.append(timeout)
        raise httpcore.ConnectTimeout("address unavailable")


class _RecordingHTTPStream(httpcore.MockStream):
    def __init__(self) -> None:
        super().__init__([b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"])
        self.writes: list[bytes] = []
        self.server_hostname: str | None = None

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        del timeout
        self.writes.append(buffer)

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        del ssl_context, timeout
        self.server_hostname = server_hostname
        return self


class _RecordingHTTPBackend(httpcore.NetworkBackend):
    def __init__(self) -> None:
        self.connected_hosts: list[str] = []
        self.streams: list[_RecordingHTTPStream] = []

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object = None,
    ) -> httpcore.NetworkStream:
        del port, timeout, local_address, socket_options
        self.connected_hosts.append(host)
        stream = _RecordingHTTPStream()
        self.streams.append(stream)
        return stream

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: object = None,
    ) -> httpcore.NetworkStream:
        del path, timeout, socket_options
        raise AssertionError("UNIX sockets are forbidden")


def test_pinned_network_backend_connects_to_validated_numeric_address() -> None:
    delegate = _RecordingNetworkBackend()
    validator = _HostValidator(lambda _host: ("93.184.216.34",))
    backend = _PinnedAddressNetworkBackend(validator, delegate=delegate)

    backend.connect_tcp("media.example.test", 443)

    assert delegate.connected_hosts == ["93.184.216.34"]


def test_network_backend_honours_explicit_public_resolution_opt_out() -> None:
    delegate = _RecordingNetworkBackend()
    validator = _HostValidator(
        lambda _host: (_ for _ in ()).throw(AssertionError("DNS must be delegated"))
    )
    backend = _PinnedAddressNetworkBackend(
        validator,
        delegate=delegate,
        pinned_hosts=frozenset(),
    )

    backend.connect_tcp("private-media.example.test", 443)

    assert delegate.connected_hosts == ["private-media.example.test"]


def test_pinned_network_backend_re_resolves_and_rejects_rebinding() -> None:
    resolutions = iter((("93.184.216.34",), ("127.0.0.1",)))
    delegate = _RecordingNetworkBackend()
    validator = _HostValidator(lambda _host: next(resolutions))
    backend = _PinnedAddressNetworkBackend(validator, delegate=delegate)

    backend.connect_tcp("media.example.test", 443)
    with pytest.raises(httpcore.ConnectError, match="non-public address"):
        backend.connect_tcp("media.example.test", 443)

    assert delegate.connected_hosts == ["93.184.216.34"]


def test_pinned_network_backend_rejects_mixed_public_private_answers() -> None:
    delegate = _RecordingNetworkBackend()
    validator = _HostValidator(lambda _host: ("93.184.216.34", "169.254.169.254"))
    backend = _PinnedAddressNetworkBackend(validator, delegate=delegate)

    with pytest.raises(httpcore.ConnectError, match="non-public address"):
        backend.connect_tcp("media.example.test", 443)

    assert delegate.connected_hosts == []


def test_pinned_network_backend_shares_one_deadline_across_addresses() -> None:
    times = iter((100.0, 100.0, 104.0))
    delegate = _FailingNetworkBackend()
    validator = _HostValidator(lambda _host: ("93.184.216.34", "93.184.216.35"))
    backend = _PinnedAddressNetworkBackend(
        validator,
        delegate=delegate,
        monotonic=lambda: next(times),
    )

    with pytest.raises(httpcore.ConnectTimeout, match="address unavailable"):
        backend.connect_tcp("media.example.test", 443, timeout=10.0)

    assert delegate.connected_hosts == ["93.184.216.34", "93.184.216.35"]
    assert delegate.timeouts == [10.0, 6.0]


def test_pinned_transport_preserves_origin_sni_and_host_without_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    backend = _RecordingHTTPBackend()
    validator = _HostValidator(lambda _host: ("93.184.216.34",))
    transport = _PinnedAddressHTTPTransport(
        validator,
        max_connections=2,
        network_backend=backend,
    )

    with httpx.Client(transport=transport, trust_env=True) as client:
        first = client.get("https://one.example.test/first")
        second = client.get("https://two.example.test/second")
        trailing_dot = client.get("https://one.example.test./trailing-dot")

    assert first.status_code == 200
    assert str(first.url) == "https://one.example.test/first"
    assert second.status_code == 200
    assert str(second.url) == "https://two.example.test/second"
    assert trailing_dot.status_code == 200
    assert backend.connected_hosts == [
        "93.184.216.34",
        "93.184.216.34",
        "93.184.216.34",
    ]
    assert [stream.server_hostname for stream in backend.streams] == [
        "one.example.test",
        "two.example.test",
        "one.example.test.",
    ]
    written = [b"".join(stream.writes).lower() for stream in backend.streams]
    assert b"host: one.example.test" in written[0]
    assert b"host: two.example.test" in written[1]
    assert b"host: one.example.test." in written[2]


def test_download_rejects_non_mock_injected_http_client() -> None:
    candidates, selections = _frames()
    storage = _MemoryStorage()

    with httpx.Client(transport=httpx.HTTPTransport()) as client:
        with pytest.raises(TypeError, match="restricted to MockTransport"):
            download_reference_media(
                selections,
                candidates,
                storage=storage,
                output_prefix="s3://references/bank-v1",
                config=_config(),
                http_client=client,
                now=lambda: _NOW,
            )

    report = next(iter(storage.json.values()))
    assert report["status"] == "failed"


def test_mock_client_mounts_are_not_inherited_by_downloader() -> None:
    candidates, selections = _frames()
    payload = _image_bytes("PNG")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=payload,
        )

    class FailingMountedTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            del request
            raise AssertionError("injected client mounts must never be used")

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        mounts={"https://": FailingMountedTransport()},
    ) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )

    assert result.media_objects["decode_status"].item() == "valid"


def test_origin_limiter_is_shared_across_source_policies() -> None:
    first = ProviderMediaDownloadPolicy(
        source="GBIF",
        allowed_hosts=("shared.example.test",),
        max_concurrent_per_origin=1,
        resolve_public_addresses=False,
    )
    second = ProviderMediaDownloadPolicy(
        source="iNaturalist",
        allowed_hosts=("shared.example.test",),
        max_concurrent_per_origin=3,
        resolve_public_addresses=False,
    )
    clock = 0.0
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        nonlocal clock
        sleeps.append(seconds)
        clock += seconds

    registry = _OriginLimiterRegistry(
        policies=(first, second),
        sleep=sleep,
        monotonic=lambda: clock,
    )
    first_limiter = registry.get(first, "https://shared.example.test/a.jpg")
    second_limiter = registry.get(second, "https://shared.example.test/b.jpg")

    assert first_limiter is second_limiter
    first_limiter.defer_for(7.0)
    with second_limiter.slot():
        pass
    assert sleeps == [7.0]


def test_origin_limiter_rejects_an_expired_item_before_waiting() -> None:
    policy = ProviderMediaDownloadPolicy(
        source="TestProvider",
        allowed_hosts=("media.example.test",),
        resolve_public_addresses=False,
    )
    registry = _OriginLimiterRegistry(
        policies=(policy,),
        sleep=lambda _seconds: None,
        monotonic=lambda: 5.0,
    )

    with pytest.raises(_PermanentResponse, match="item_deadline_exceeded"):
        with registry.get(policy, "https://media.example.test/image.png").slot(
            deadline=5.0
        ):
            raise AssertionError("expired work must not enter the slot")


def test_origin_limiter_bounds_contended_slot_wait_by_item_deadline() -> None:
    policy = ProviderMediaDownloadPolicy(
        source="TestProvider",
        allowed_hosts=("media.example.test",),
        max_concurrent_per_origin=1,
        resolve_public_addresses=False,
    )
    registry = _OriginLimiterRegistry(
        policies=(policy,),
        sleep=time.sleep,
        monotonic=time.monotonic,
    )
    limiter = registry.get(policy, "https://media.example.test/image.png")

    with limiter.slot():
        with pytest.raises(_PermanentResponse, match="item_deadline_exceeded"):
            with limiter.slot(deadline=time.monotonic() + 0.01):
                raise AssertionError("contended work must not outlive its deadline")


def test_origin_limiter_does_not_hold_schedule_lock_during_throttle_wait() -> None:
    policy = ProviderMediaDownloadPolicy(
        source="TestProvider",
        allowed_hosts=("media.example.test",),
        max_concurrent_per_origin=2,
        resolve_public_addresses=False,
    )
    sleep_started = threading.Event()
    release_sleep = threading.Event()

    def blocking_sleep(_seconds: float) -> None:
        sleep_started.set()
        assert release_sleep.wait(timeout=1.0)

    registry = _OriginLimiterRegistry(
        policies=(policy,),
        sleep=blocking_sleep,
        monotonic=lambda: 0.0,
    )
    limiter = registry.get(policy, "https://media.example.test/image.png")
    limiter.defer_for(10.0)
    first_done = threading.Event()
    rejected = threading.Event()

    def first_request() -> None:
        with limiter.slot():
            first_done.set()

    def short_deadline_request() -> None:
        try:
            with limiter.slot(deadline=1.0):
                raise AssertionError("throttled work must not outlive its deadline")
        except _PermanentResponse:
            rejected.set()

    first = threading.Thread(target=first_request)
    second = threading.Thread(target=short_deadline_request)
    first.start()
    assert sleep_started.wait(timeout=1.0)
    second.start()
    try:
        assert rejected.wait(timeout=0.1)
    finally:
        release_sleep.set()
        first.join(timeout=1.0)
        second.join(timeout=1.0)
    assert first_done.is_set()


def test_transport_error_is_selectively_retried() -> None:
    payload = _image_bytes("PNG")
    candidates, selections = _frames()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporary connection failure", request=request)
        return httpx.Response(
            200, headers={"Content-Type": "image/png"}, content=payload
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
            sleep=lambda _seconds: None,
        )

    row = result.media_objects.to_dicts()[0]
    assert row["decode_status"] == "valid"
    assert row["download_attempt_count"] == 2


def test_redirects_do_not_reset_item_wide_attempt_budget() -> None:
    candidates, selections = _frames()
    payload = _image_bytes("PNG")
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls in {1, 3}:
            return httpx.Response(503)
        if calls == 2:
            return httpx.Response(
                302,
                headers={"Location": "/photos/101/redirected.png"},
            )
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=payload,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(max_attempts=3),
            http_client=client,
            now=lambda: _NOW,
        )

    assert calls == 3
    row = result.media_objects.to_dicts()[0]
    assert row["download_attempt_count"] == 3
    assert row["decode_status"] == "download_failed"
    assert row["quarantine_reason"] == "retry_exhausted_http_503"


def test_item_deadline_stops_slow_streaming_response() -> None:
    candidates, selections = _frames()
    payload = _image_bytes("PNG")
    clock = 0.0

    class SlowStream(httpx.SyncByteStream):
        def __iter__(self):
            nonlocal clock
            clock = 6.0
            yield payload

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            stream=SlowStream(),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(max_download_seconds=5.0),
            http_client=client,
            now=lambda: _NOW,
            monotonic=lambda: clock,
        )

    row = result.media_objects.to_dicts()[0]
    assert row["download_attempt_count"] == 1
    assert row["decode_status"] == "download_failed"
    assert row["quarantine_reason"] == "item_deadline_exceeded"


def test_isolated_image_decode_succeeds_and_honours_hard_timeout(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(_image_bytes("PNG", size=(4, 3)))

    assert _decode_image_isolated(
        image_path,
        max_pixels=100,
        timeout_seconds=5.0,
    ) == (4, 3, "image/png")
    with pytest.raises(_PermanentResponse, match="item_deadline_exceeded"):
        _decode_image_isolated(
            image_path,
            max_pixels=100,
            timeout_seconds=1e-9,
        )


def test_host_resolution_has_a_hard_timeout() -> None:
    release = threading.Event()

    def stalled_resolver(_host: str) -> tuple[str, ...]:
        release.wait(timeout=1.0)
        return ("93.184.216.34",)

    validator = _HostValidator(stalled_resolver)
    try:
        with pytest.raises(ValueError, match="resolution timed out"):
            validator.public_addresses(
                "media.example.test",
                timeout_seconds=0.01,
            )
        started = time.monotonic()
        with pytest.raises(ValueError, match="resolution timed out"):
            validator.public_addresses(
                "media.example.test",
                timeout_seconds=0.5,
            )
        assert time.monotonic() - started < 0.1
    finally:
        release.set()
    recovery_deadline = time.monotonic() + 0.5
    while True:
        try:
            recovered = validator.public_addresses(
                "media.example.test",
                timeout_seconds=0.5,
            )
            break
        except ValueError as exc:
            if "resolution timed out" not in str(exc):
                raise
            if time.monotonic() >= recovery_deadline:
                raise
            time.sleep(0.001)
    assert recovered == ("93.184.216.34",)


def test_source_checksum_mismatch_quarantines_valid_image() -> None:
    payload = _image_bytes("PNG")
    candidates, selections = _frames(
        source_checksum="0" * 64,
        source_checksum_algorithm="SHA-256",
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Type": "image/png"}, content=payload
        )

    storage = _MemoryStorage()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )

    row = result.media_objects.to_dicts()[0]
    assert row["decode_status"] == "decode_failed"
    assert row["quarantine_reason"] == "source_checksum_mismatch"
    assert not storage.files


def test_missing_committed_object_makes_resume_fail_closed() -> None:
    payload = _image_bytes("PNG")
    candidates, selections = _frames()
    storage = _MemoryStorage()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Type": "image/png"}, content=payload
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )
        del storage.files[first.media_objects["source_object_uri"].item()]
        with pytest.raises(ValueError, match="missing object"):
            download_reference_media(
                selections,
                candidates,
                storage=storage,
                output_prefix="s3://references/bank-v1",
                config=_config(),
                http_client=client,
                now=lambda: _NOW,
            )


def test_wrong_committed_object_size_makes_resume_fail_closed() -> None:
    payload = _image_bytes("PNG")
    candidates, selections = _frames()
    storage = _MemoryStorage()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Type": "image/png"}, content=payload
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )
        storage.files[first.media_objects["source_object_uri"].item()] = b"truncated"
        with pytest.raises(ValueError, match="object size is incompatible"):
            download_reference_media(
                selections,
                candidates,
                storage=storage,
                output_prefix="s3://references/bank-v1",
                config=_config(),
                http_client=client,
                now=lambda: _NOW,
            )


def test_same_size_object_corruption_makes_resume_fail_closed() -> None:
    payload = _image_bytes("PNG")
    candidates, selections = _frames()
    storage = _MemoryStorage()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Type": "image/png"}, content=payload
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )
        object_uri = first.media_objects["source_object_uri"].item()
        corrupted = bytearray(storage.files[object_uri])
        corrupted[-1] ^= 1
        storage.files[object_uri] = bytes(corrupted)
        with pytest.raises(ValueError, match="object checksum is incompatible"):
            download_reference_media(
                selections,
                candidates,
                storage=storage,
                output_prefix="s3://references/bank-v1",
                config=_config(),
                http_client=client,
                now=lambda: _NOW,
            )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("missing_attribution", "download evidence is incomplete"),
        ("invalid_commit_marker", "commit marker is invalid"),
        ("object_fingerprint", "object fingerprint is invalid"),
        ("object_identity", "object identity is incompatible"),
        ("final_url", "final URL is invalid"),
    ],
)
def test_checkpoint_tampering_makes_resume_fail_closed(
    tamper: str,
    message: str,
) -> None:
    payload = _image_bytes("PNG")
    candidates, selections = _frames()
    storage = _MemoryStorage()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Type": "image/png"}, content=payload
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )
        checkpoint_uri = next(uri for uri in storage.json if "/checkpoints/" in uri)
        checkpoint = storage.json[checkpoint_uri]
        if tamper == "missing_attribution":
            checkpoint["download_evidence"].pop("attribution")
        elif tamper == "invalid_commit_marker":
            checkpoint["commit"]["object_write_precedes_checkpoint"] = False
        elif tamper == "object_fingerprint":
            checkpoint["object"]["object_fingerprint"] = "sha256:" + "f" * 64
        elif tamper == "object_identity":
            checkpoint["object"]["reference_media_id"] = "reference-media:other"
        else:
            checkpoint["download_evidence"]["final_url"] = (
                "https://unreviewed.example.test/image.png"
            )

        with pytest.raises(ValueError, match=message):
            download_reference_media(
                selections,
                candidates,
                storage=storage,
                output_prefix="s3://references/bank-v1",
                config=_config(),
                http_client=client,
                now=lambda: _NOW,
            )


def test_changed_candidate_rejects_stale_checkpoint_without_redownload() -> None:
    payload = _image_bytes("PNG")
    candidates, selections = _frames()
    storage = _MemoryStorage()
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, headers={"Content-Type": "image/png"}, content=payload
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        successful = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )
        changed_rows = candidates.to_dicts()
        changed_rows[0]["media_identifier"] = (
            "https://media.example.test/photos/101/replacement.png"
        )
        changed_candidates = reference_media_candidates_frame(changed_rows)
        with pytest.raises(ValueError, match="binding is incompatible"):
            download_reference_media(
                selections,
                changed_candidates,
                storage=storage,
                output_prefix="s3://references/bank-v1",
                config=_config(),
                http_client=client,
                now=lambda: _NOW,
            )

    assert calls == 1
    failed_reports = [
        value
        for value in storage.json.values()
        if value.get("schema_version") == "reference-media-download-report-v1"
        and value.get("status") == "failed"
    ]
    assert len(failed_reports) == 1
    assert failed_reports[0]["error"]["type"] == "ValueError"
    assert set(failed_reports[0]) == set(successful.report)
    assert failed_reports[0]["settings"] == successful.report["settings"]


def test_checkpoint_reuses_object_across_lifecycle_state_and_plan_changes() -> None:
    payload = _image_bytes("PNG")
    candidates, selections = _frames()
    storage = _MemoryStorage()
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, headers={"Content-Type": "image/png"}, content=payload
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )
        candidate_rows = candidates.to_dicts()
        candidate_rows[0]["download_status"] = "complete"
        candidate_rows[0]["licence_policy_status"] = "allowed"
        updated_candidates = reference_media_candidates_frame(candidate_rows)
        selection_rows = selections.to_dicts()
        selection_rows[0]["acquisition_plan_id"] = "plan-2"
        selection_rows[0]["reference_selection_id"] = make_reference_selection_id(
            acquisition_plan_id="plan-2",
            reference_media_id=str(selection_rows[0]["reference_media_id"]),
            candidate_accepted_taxon_key="taxon-1",
            geo_cluster_id="cluster-1",
            life_stage="adult",
            visual_domain="unreviewed",
        )
        updated_selections = reference_acquisition_selections_frame(selection_rows)
        second = download_reference_media(
            updated_selections,
            updated_candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )

    assert calls == 1
    assert second.report["counts"]["resumed"] == 1
    assert second.media_objects.equals(first.media_objects)


def test_checkpoint_ignores_operational_retry_and_throttle_tuning() -> None:
    payload = _image_bytes("PNG")
    candidates, selections = _frames()
    storage = _MemoryStorage()
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, headers={"Content-Type": "image/png"}, content=payload
        )

    first_policy = ProviderMediaDownloadPolicy(
        source="TestProvider",
        allowed_hosts=("media.example.test",),
        max_concurrent_per_origin=1,
        min_request_interval_seconds=0.0,
        resolve_public_addresses=False,
    )
    tuned_policy = ProviderMediaDownloadPolicy(
        source="TestProvider",
        allowed_hosts=("media.example.test",),
        max_concurrent_per_origin=4,
        min_request_interval_seconds=1.5,
        resolve_public_addresses=False,
    )
    unrelated_policy = ProviderMediaDownloadPolicy(
        source="GBIF",
        allowed_hosts=("reviewed-publisher.example.test",),
        resolve_public_addresses=False,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(
                max_attempts=2,
                retry_statuses=(429, 503),
                provider_policies=(first_policy,),
            ),
            http_client=client,
            now=lambda: _NOW,
        )
        second = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(
                max_attempts=9,
                max_redirects=7,
                max_retry_after_seconds=120.0,
                retry_statuses=(408, 425, 429, 500, 502, 503, 504),
                provider_policies=(tuned_policy, unrelated_policy),
            ),
            http_client=client,
            now=lambda: _NOW,
        )

    assert calls == 1
    assert second.report["counts"]["resumed"] == 1
    assert second.media_objects.equals(first.media_objects)


def test_report_counts_content_addressed_source_bytes_once() -> None:
    first_candidates, first_selections = _frames(
        provider_media_id="101",
        observation_id="observation-101",
        acquisition_plan_id="plan-101",
    )
    second_candidates, second_selections = _frames(
        provider_media_id="202",
        observation_id="observation-202",
        url="https://media.example.test/photos/202/original.png",
        acquisition_plan_id="plan-202",
    )
    candidates = reference_media_candidates_frame(
        first_candidates.to_dicts() + second_candidates.to_dicts()
    )
    selections = reference_acquisition_selections_frame(
        first_selections.to_dicts() + second_selections.to_dicts()
    )
    payload = _image_bytes("PNG")
    storage = _MemoryStorage()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=payload,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(workers=2, max_inflight=2),
            http_client=client,
            now=lambda: _NOW,
        )

    assert result.media_objects["source_object_uri"].n_unique() == 1
    assert len(storage.files) == 1
    assert result.report["bytes"]["source_objects"] == len(payload)


def test_disjoint_runs_merge_media_object_inventory() -> None:
    first_candidates, first_selections = _frames(
        provider_media_id="101",
        observation_id="observation-101",
        acquisition_plan_id="plan-101",
    )
    second_candidates, second_selections = _frames(
        provider_media_id="202",
        observation_id="observation-202",
        url="https://media.example.test/photos/202/original.png",
        acquisition_plan_id="plan-202",
    )
    storage = _MemoryStorage()

    def handler(request: httpx.Request) -> httpx.Response:
        color = (10, 20, 30) if "/101/" in request.url.path else (30, 20, 10)
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=_image_bytes("PNG", color=color),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = download_reference_media(
            first_selections,
            first_candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )
        second = download_reference_media(
            second_selections,
            second_candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )

    assert first.media_objects.height == 1
    assert second.media_objects.height == 2
    assert second.media_objects["reference_media_id"].to_list() == sorted(
        first_candidates["reference_media_id"].to_list()
        + second_candidates["reference_media_id"].to_list()
    )
    assert second.report["counts"]["selected"] == 1
    assert second.report["counts"]["rows_out"] == 1
    assert second.report["counts"]["inventory_rows"] == 2
    assert first.report["inputs"]["selection_rows"] == 1
    assert second.report["inputs"]["selection_rows"] == 1
    assert (
        first.report["inputs"]["selection_fingerprint"]
        != second.report["inputs"]["selection_fingerprint"]
    )
    assert (
        first.report["inputs"]["selected_candidates_fingerprint"]
        != (second.report["inputs"]["selected_candidates_fingerprint"])
    )
    assert first.report["inputs"]["acquisition_plan_ids"] == ["plan-101"]
    assert second.report["inputs"]["acquisition_plan_ids"] == ["plan-202"]
    assert storage.parquet[second.media_objects_uri].equals(second.media_objects)


def test_checkpoint_failure_never_reports_object_as_committed() -> None:
    payload = _image_bytes("PNG")
    candidates, selections = _frames()
    storage = _MemoryStorage()
    storage.fail_checkpoint = True

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Type": "image/png"}, content=payload
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=storage,
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )

    row = result.media_objects.to_dicts()[0]
    assert row["decode_status"] == "download_failed"
    assert row["source_object_uri"] is None
    assert row["quarantine_reason"] == "object_commit_failed:OSError"
    assert storage.files  # The content-addressed orphan is not a false committed row.
    assert not [uri for uri in storage.json if "/checkpoints/" in uri]


def test_unselected_candidates_are_never_downloaded() -> None:
    first_candidates, selections = _frames()
    second_candidates, _ = _frames(
        provider_media_id="202",
        observation_id="observation-202",
        url="https://media.example.test/photos/202/original.png",
    )
    candidates = reference_media_candidates_frame(
        first_candidates.to_dicts() + second_candidates.to_dicts()
    )
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=_image_bytes("PNG"),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )

    assert result.media_objects.height == 1
    assert requested == ["https://media.example.test/photos/101/original.png"]


def test_same_media_selected_by_multiple_plans_is_downloaded_once() -> None:
    candidates, selections = _frames()
    rows = selections.to_dicts()
    second = dict(rows[0])
    second["acquisition_plan_id"] = "plan-2"
    second["reference_selection_id"] = make_reference_selection_id(
        acquisition_plan_id="plan-2",
        reference_media_id=str(second["reference_media_id"]),
        candidate_accepted_taxon_key=str(second["candidate_accepted_taxon_key"]),
        geo_cluster_id=str(second["geo_cluster_id"]),
        life_stage=str(second["life_stage"]),
        visual_domain=str(second["visual_domain"]),
    )
    multi_plan_selections = reference_acquisition_selections_frame(rows + [second])
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=_image_bytes("PNG"),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            multi_plan_selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(),
            http_client=client,
            now=lambda: _NOW,
        )

    assert calls == 1
    assert result.media_objects.height == 1
    assert result.report["counts"]["selected"] == 1
    assert result.report["inputs"]["selection_rows"] == 2
    assert result.report["inputs"]["acquisition_plan_ids"] == ["plan-1", "plan-2"]


def test_successful_run_removes_staged_files_immediately(tmp_path: Path) -> None:
    candidates, selections = _frames()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=_image_bytes("PNG"),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(temporary_directory=tmp_path),
            http_client=client,
            now=lambda: _NOW,
        )

    assert result.media_objects["decode_status"].item() == "valid"
    assert list(tmp_path.iterdir()) == []


def test_out_of_order_workers_still_emit_deterministic_media_order() -> None:
    first_candidates, first_selections = _frames(
        provider_media_id="202",
        observation_id="observation-202",
        url="https://media.example.test/photos/202/original.png",
        acquisition_plan_id="plan-202",
    )
    second_candidates, second_selections = _frames(
        provider_media_id="101",
        observation_id="observation-101",
        url="https://media.example.test/photos/101/original.png",
        acquisition_plan_id="plan-101",
    )
    candidates = reference_media_candidates_frame(
        first_candidates.to_dicts() + second_candidates.to_dicts()
    )
    selections = reference_acquisition_selections_frame(
        first_selections.to_dicts() + second_selections.to_dicts()
    )
    release = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        if "/101/" in request.url.path:
            release.set()
        else:
            release.wait(timeout=1)
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=_image_bytes(
                "PNG", color=(int(request.url.path.split("/")[2]) % 255, 0, 0)
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(workers=2, max_inflight=2),
            http_client=client,
            now=lambda: _NOW,
        )

    ids = result.media_objects["reference_media_id"].to_list()
    assert ids == sorted(ids)
    assert result.media_objects["decode_status"].to_list() == ["valid", "valid"]


def test_workers_obey_per_origin_concurrency_and_release_slots_on_errors() -> None:
    candidate_frames = []
    selection_frames = []
    for provider_media_id in ("101", "202", "303", "404"):
        candidates, selections = _frames(
            provider_media_id=provider_media_id,
            observation_id=f"observation-{provider_media_id}",
            url=(f"https://media.example.test/photos/{provider_media_id}/original.png"),
            acquisition_plan_id=f"plan-{provider_media_id}",
        )
        candidate_frames.extend(candidates.to_dicts())
        selection_frames.extend(selections.to_dicts())
    candidates = reference_media_candidates_frame(candidate_frames)
    selections = reference_acquisition_selections_frame(selection_frames)
    policy = ProviderMediaDownloadPolicy(
        source="TestProvider",
        allowed_hosts=("media.example.test",),
        max_concurrent_per_origin=2,
        resolve_public_addresses=False,
    )
    lock = threading.Lock()
    two_active = threading.Event()
    active = 0
    max_active = 0
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active, calls
        with lock:
            active += 1
            calls += 1
            max_active = max(max_active, active)
            if active == 2:
                two_active.set()
        try:
            assert two_active.wait(timeout=1)
            time.sleep(0.02)
            if "/101/" in request.url.path:
                return httpx.Response(404)
            return httpx.Response(
                200,
                headers={"Content-Type": "image/png"},
                content=_image_bytes("PNG"),
            )
        finally:
            with lock:
                active -= 1

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_reference_media(
            selections,
            candidates,
            storage=_MemoryStorage(),
            output_prefix="s3://references/bank-v1",
            config=_config(
                workers=4,
                max_inflight=4,
                provider_policies=(policy,),
            ),
            http_client=client,
            now=lambda: _NOW,
        )

    assert calls == 4
    assert max_active == 2
    assert result.media_objects["decode_status"].to_list().count("valid") == 3
    assert result.media_objects["decode_status"].to_list().count("download_failed") == 1


def _config(**overrides: object) -> ReferenceMediaDownloadConfig:
    values: dict[str, object] = {
        "workers": 2,
        "max_inflight": 4,
        "provider_policies": (
            ProviderMediaDownloadPolicy(
                source="TestProvider",
                allowed_hosts=("media.example.test",),
                max_concurrent_per_origin=2,
                resolve_public_addresses=False,
            ),
        ),
        "backoff_base_seconds": 0.0,
        "backoff_cap_seconds": 60.0,
        "git_sha": "test-sha",
    }
    values.update(overrides)
    return ReferenceMediaDownloadConfig(**values)


def _frames(
    *,
    source: str = "TestProvider",
    provider_media_id: str = "101",
    observation_id: str = "observation-101",
    url: str = "https://media.example.test/photos/101/original.png",
    licence: str | None = "cc-by",
    licence_uri: str | None = "https://creativecommons.org/licenses/by/4.0/",
    attribution: str | None = "A. Observer / CC BY",
    source_checksum: str | None = None,
    source_checksum_algorithm: str | None = None,
    acquisition_plan_id: str = "plan-1",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    reference_observation_id = make_reference_observation_id(source, observation_id)
    reference_media_id = make_reference_media_id(
        source,
        provider_media_id,
        reference_observation_id,
    )
    candidates = reference_media_candidates_frame(
        [
            {
                "schema_version": REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
                "reference_media_id": reference_media_id,
                "reference_observation_id": reference_observation_id,
                "provider_media_id": provider_media_id,
                "source": source,
                "media_identifier": url,
                "media_type": "StillImage",
                "width": None,
                "height": None,
                "creator": "A. Observer",
                "rights_holder": "A. Observer",
                "licence": licence,
                "licence_uri": licence_uri,
                "attribution": attribution,
                "occurrence_licence": "cc-by",
                "original_provider": source,
                "media_position": 0,
                "source_checksum": source_checksum,
                "source_checksum_algorithm": source_checksum_algorithm,
                "download_status": "pending",
                "verification_status": "unreviewed",
                "exclusion_reason": None,
                "licence_policy_status": "unreviewed",
                "retrieved_at": _NOW,
                "source_snapshot_version": "snapshot-v1",
            }
        ]
    )
    selection_id = make_reference_selection_id(
        acquisition_plan_id=acquisition_plan_id,
        reference_media_id=reference_media_id,
        candidate_accepted_taxon_key="taxon-1",
        geo_cluster_id="cluster-1",
        life_stage="adult",
        visual_domain="unreviewed",
    )
    selections = reference_acquisition_selections_frame(
        [
            {
                "schema_version": REFERENCE_ACQUISITION_SELECTIONS_SCHEMA_VERSION,
                "reference_selection_id": selection_id,
                "acquisition_plan_id": acquisition_plan_id,
                "target_accepted_taxon_key": "target-1",
                "candidate_set_id": "candidate-set-1",
                "source_candidate_set_id": "source-set-1",
                "candidate_accepted_taxon_key": "taxon-1",
                "scientific_name": "Papilio testus",
                "geo_cluster_id": "cluster-1",
                "life_stage": "adult",
                "visual_domain": "unreviewed",
                "reference_media_id": reference_media_id,
                "reference_observation_id": reference_observation_id,
                "source": source,
                "fallback_level": 0,
                "selection_rank": 1,
                "selection_round": "independent_observation",
                "distance_to_cluster_medoid_km": 1.0,
                "observer_id": "observer-1",
                "observed_date": _NOW.date(),
                "locality": "Test locality",
                "background_group_id": "background-1",
                "licence": licence,
                "source_snapshot_version": "snapshot-v1",
                "selection_strategy": "test-selection-v1",
                "selection_seed": 42,
                "plan_configuration_fingerprint": _PLAN_FINGERPRINT,
                "selected_at": _NOW,
            }
        ]
    )
    return candidates, selections
