from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from biominer.gbif_quality.provider_enrichment import (
    DEFAULT_PROVIDER_METADATA_ADAPTERS,
    ProviderMetadataAdapter,
    publish_provider_enrichment_registry,
)


def test_provider_adapter_accepts_only_direct_structured_fields() -> None:
    adapter = DEFAULT_PROVIDER_METADATA_ADAPTERS[0]
    assert isinstance(adapter, ProviderMetadataAdapter)
    request = adapter.request(
        provider="Vermont Center for Ecostudies", resource_identifier="photo-1"
    )
    assert request.cache_key.startswith("sha256:")
    evidence = adapter.parse(
        {
            "evidence_scope": "item",
            "provider_resource_identifier": "photo-1",
            "media_license": "CC BY 4.0",
            "creator": "A Person",
            "occurrence_license": "CC0",
            "recordedBy": "Someone Else",
        },
        evidence_source_uri="https://provider.example/export.json",
    )
    assert evidence.media_license == "CC BY 4.0"
    assert evidence.creator == "A Person"
    assert evidence.rights_holder is None

    with pytest.raises(ValueError, match="scope"):
        adapter.parse(
            {"evidence_scope": "unknown"},
            evidence_source_uri="https://provider.example/export.json",
        )


def test_provider_registry_withholds_unexecuted_enrichment(tmp_path) -> None:
    output = tmp_path / "registry"
    manifest = publish_provider_enrichment_registry(
        output_directory=output,
        source_snapshot_id="sha256:snapshot",
        code_commit="commit",
    )

    rows = pq.read_table(output / "provider_enrichment_registry.parquet").to_pylist()
    assert len(rows) == 7
    assert {row["execution_status"] for row in rows} == {"NOT_TESTED"}
    assert manifest["counts"] == {"registered_adapters": 7, "executed_adapters": 0}
    assert manifest["network_requests"] == 0
