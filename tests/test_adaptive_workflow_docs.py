from pathlib import Path


def test_fast_start_documentation_covers_complete_production_lifecycle() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    production = Path("docs/production.md").read_text(encoding="utf-8")
    workflow = Path("docs/adaptive_gbif_fast_start.md").read_text(encoding="utf-8")
    normalized_production = " ".join(production.split())
    normalized_workflow = " ".join(workflow.split())
    assert "adaptive_gbif_fast_start" in readme
    assert "human_verified_strict" in readme
    assert "risk-controlled and statistical reference audit" in normalized_production
    assert "family and geography hard pruning are forbidden" in normalized_production
    assert (
        "canonical target-aware model input is the full frame" in normalized_production
    )
    assert "fail closed when their live adapter initializes" in normalized_production
    required = (
        "Reference acquisition and admission",
        "Review and readiness",
        "BioCLIP, orchestration and selective reruns",
        "CLI and local/cloud execution",
        "Evaluation and release",
        "GBIF provider-asserted provisional support",
        "YOLOE supplies route and domain evidence",
        "ready_provisional",
        "provisional_reference_ranking",
        "insufficient_sample",
        "Final export remains fail-closed",
    )
    assert all(term in normalized_workflow for term in required)
