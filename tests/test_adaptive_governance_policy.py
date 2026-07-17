from pathlib import Path


def test_agent_and_human_decision_record_adaptive_scientific_boundaries() -> None:
    instruction_paths = [Path("AGENTS.md")]
    science_topic = Path("docs/agents/SCIENCE_AND_PIPELINE.md")
    if science_topic.exists():
        instruction_paths.append(science_topic)
    agents = "\n".join(
        path.read_text(encoding="utf-8") for path in instruction_paths
    )
    decision = Path("docs/governance/adaptive_reference_policy.md").read_text(
        encoding="utf-8"
    )
    required_agent_boundaries = (
        ("GBIF provider-asserted provisional support",),
        ("not human verification", "never call it verified"),
        ("final occurrence dataset", "Final inclusion requires"),
        (
            "statistical flag prioritizes review",
            "Statistical findings prioritize human reference review",
        ),
        ("human_verified_strict",),
        (
            "not probabilities or confidence values",
            "Raw similarities, distances, detector scores, margins, and SVM outputs",
        ),
        ("Do not weaken unrelated", "must not weaken scientific"),
    )
    assert all(
        any(term in agents for term in alternatives)
        for alternatives in required_agent_boundaries
    )
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
