from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BucketPolicy:
    gold_species_threshold: float = 0.70
    silver_species_threshold: float = 0.35
    hard_negative_threshold: float = 0.70
    ambiguous_margin_threshold: float = 0.05


DEFAULT_BUCKET_POLICY = BucketPolicy()
