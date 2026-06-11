from __future__ import annotations

from pathlib import Path


def test_env_example_exists() -> None:
    assert Path(".env.example").exists()


def test_env_example_contains_only_variable_names() -> None:
    text = Path(".env.example").read_text(encoding="utf-8")

    assert "FLICKR_API_KEY=" in text
    assert "FLICKR_API_KEY=your_flickr_api_key_here" in text


def test_local_papilio_demoleus_config_files_are_ignored() -> None:
    text = Path(".gitignore").read_text(encoding="utf-8")

    assert "config/papilio_demoleus_flickr_estimator.sh2" in text
    assert "config/papilio_demoleus_multilingual_keywords.json" in text
