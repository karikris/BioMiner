from __future__ import annotations

import argparse
from pathlib import Path

from flickr_bio_occurrence.benchmark.offline_run import run_existing_payload_benchmark
from flickr_bio_occurrence.vision.pipeline import build_bioclip_row_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-root", default="/home/toffe/BioMiner/data/flickr_butterfly_australia")
    parser.add_argument("--output-dir", default="/home/toffe/BioMiner/data/flickr_butterfly_australia/bioclip25_all")
    parser.add_argument("--model-registry", default="config/model_registry.toml")
    parser.add_argument("--target-records", type=int, default=10_000_000)
    args = parser.parse_args()

    fetch_root = Path(args.fetch_root)
    payload_paths = sorted((fetch_root / "raw").rglob("*.json"))
    if not payload_paths:
        raise RuntimeError(f"No raw Flickr payloads found under {fetch_root / 'raw'}")
    classifier = build_bioclip_row_classifier(
        model_registry_path=args.model_registry,
        cache_root=fetch_root / "cache" / "images",
    )
    report_path = run_existing_payload_benchmark(
        payload_paths=payload_paths,
        output_dir=args.output_dir,
        species_name="Australian butterflies",
        region_id="AU_ALL",
        target_records=args.target_records,
        vision_classifier=classifier,
    )
    print(report_path)


if __name__ == "__main__":
    main()
