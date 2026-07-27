from __future__ import annotations

from datetime import UTC, datetime
import importlib.util
from pathlib import Path
from types import SimpleNamespace


_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_gbif_flickr_keyword_recovery.py"
_SPEC = importlib.util.spec_from_file_location("run_gbif_flickr_keyword_recovery", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
recovery = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(recovery)


def test_seconds_until_next_utc_day_includes_two_minute_safety_margin() -> None:
    now = datetime(2026, 7, 25, 23, 59, tzinfo=UTC)

    assert recovery._seconds_until_next_utc_day(now) == 180.0


def test_recovery_retries_wikimedia_daily_budget_and_never_enables_mymemory(tmp_path, monkeypatch) -> None:
    translation_calls: list[dict[str, object]] = []
    compile_calls: list[dict[str, object]] = []
    sleeps: list[float] = []

    monkeypatch.setattr(
        recovery,
        "build_enrichment_sources_from_registry",
        lambda **kwargs: {
            "status": "complete",
            "coverage": {"enriched_species": 1},
            "name_assertion_rows": 2,
            "external_taxon_link_rows": 1,
        },
    )

    def fake_translation(**kwargs):
        translation_calls.append(kwargs)
        return {
            "translation_status": "budget_exhausted" if len(translation_calls) == 1 else "complete",
            "wikimedia_assertion_rows": len(translation_calls),
            "translation_request_rows": len(translation_calls),
        }

    def fake_compile(**kwargs):
        compile_calls.append(kwargs)
        return {
            "qa_status": "passed",
            "query_definition_rows": 10,
            "name_rows": 5,
        }

    monkeypatch.setattr(recovery, "build_translation_candidates_from_registry", fake_translation)
    monkeypatch.setattr(recovery, "compile_enriched_registry", fake_compile)
    monkeypatch.setattr(recovery, "_seconds_until_next_utc_day", lambda: 60.0)
    monkeypatch.setattr(recovery.time, "sleep", sleeps.append)

    result = recovery.run(
        SimpleNamespace(
            base_registry_dir=tmp_path / "base",
            enrichment_dir=tmp_path / "enrichment",
            output_dir=tmp_path / "output",
            registry_version="recovery-v2",
            report_dir=tmp_path / "reports",
            target_locales_json=tmp_path / "locales.json",
            workers=4,
            progress_every=10,
            checkpoint_every=10,
            max_retries=2,
            translation_daily_request_limit=100,
            translation_checkpoint_every=10,
            translation_checkpoint_seconds=1.0,
        )
    )

    assert len(translation_calls) == 2
    assert all(call["translation_sources"] == ("wikimedia",) for call in translation_calls)
    assert sleeps == [60.0]
    assert compile_calls[0]["requested_translation_sources"] == ("wikimedia",)
    assert result["machine_translation_enabled"] is False
    assert (tmp_path / "output" / "recovery_manifest.json").exists()
