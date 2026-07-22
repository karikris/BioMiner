from biominer.gbif_quality.performance import BENCHMARK_SCHEMA, _parse_bytes


def test_performance_contract_uses_bytes_and_required_metrics() -> None:
    assert _parse_bytes("8GB") == 8 * 1024**3
    assert "process_peak_rss_bytes" in BENCHMARK_SCHEMA.names
    assert "rows_per_second" in BENCHMARK_SCHEMA.names
    assert "result_fingerprint" in BENCHMARK_SCHEMA.names
