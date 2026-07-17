from pathlib import Path


def test_agent_and_human_decision_record_adaptive_scientific_boundaries() -> None:
    agents = Path("AGENTS.md").read_text(encoding="utf-8")
    decision = Path("docs/governance/adaptive_reference_policy.md").read_text(
        encoding="utf-8"
    )
    required_agents = (
        "GBIF provider-asserted provisional support",
        "not human verification",
        "final occurrence dataset",
        "statistical flag prioritizes review",
        "human_verified_strict",
        "not probabilities or confidence values",
        "Do not weaken unrelated",
    )
    assert all(term in agents for term in required_agents)
    required_decision = (
        "Status: accepted",
        "Human reviewer: Kris Kari",
        "adaptive_gbif_fast_start",
        "source image hash",
        "Pre-review Flickr scores remain candidate evidence",
        "independent calibrator",
        "It does not weaken",
    )
    assert all(term in decision for term in required_decision)
