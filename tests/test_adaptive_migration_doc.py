from pathlib import Path


def test_adaptive_default_migration_guide_covers_safe_transition() -> None:
    text = Path(
        "docs/migrations/adaptive-gbif-reference-default.md"
    ).read_text(encoding="utf-8")
    required = (
        "human_verified_strict",
        "adaptive_gbif_fast_start",
        "reference-support-manifest-v3.0.0",
        "reference-bank-readiness-v3.0.0",
        "v2-to-v3",
        "policy fingerprint",
        "invalidates readiness",
        "Rollback",
        "final release",
        "raw score",
    )
    assert all(term in text for term in required)
    assert "provider assertions alone are ineligible" in text
    assert "never reuse a strict readiness checksum" in text.casefold()
