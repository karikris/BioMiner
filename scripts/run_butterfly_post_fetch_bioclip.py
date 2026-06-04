from __future__ import annotations

import argparse
from pathlib import Path

from flickr_bio_occurrence.benchmark.offline_run import _build_bronze, _read_payloads, run_existing_payload_benchmark
from flickr_bio_occurrence.vision.prefetch import build_manifest_cache_image, prefetch_image_urls
from flickr_bio_occurrence.vision.pipeline import build_bioclip_row_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-root", default="/home/toffe/BioMiner/data/flickr_butterfly_australia")
    parser.add_argument("--output-dir", default="/home/toffe/BioMiner/data/flickr_butterfly_australia/bioclip25_all")
    parser.add_argument("--model-registry", default="config/model_registry.toml")
    parser.add_argument("--target-records", type=int, default=10_000_000)
    parser.add_argument("--image-cache-workers", type=int, default=16)
    parser.add_argument("--skip-image-prefetch", action="store_true")
    parser.add_argument("--prefetch-fail-on-error", action="store_true")
    args = parser.parse_args()

    fetch_root = Path(args.fetch_root)
    payload_paths = sorted((fetch_root / "raw").rglob("*.json"))
    if not payload_paths:
        raise RuntimeError(f"No raw Flickr payloads found under {fetch_root / 'raw'}")
    image_cache_root = fetch_root / "cache" / "images"
    image_cache_manifest = image_cache_root / "image_url_cache.parquet"
    if not args.skip_image_prefetch:
        payload_items = _read_payloads(payload_paths)
        bronze = _build_bronze(payload_items, "Australian butterflies", "AU_ALL", args.target_records)
        prefetch_result = prefetch_image_urls(
            bronze.to_dicts(),
            cache_root=image_cache_root,
            manifest_path=image_cache_manifest,
            max_workers=args.image_cache_workers,
            fail_on_error=args.prefetch_fail_on_error,
        )
        print(
            {
                "image_prefetch_manifest": str(prefetch_result.manifest_path),
                "requested_urls": prefetch_result.requested_urls,
                "already_cached": prefetch_result.already_cached,
                "newly_cached": prefetch_result.newly_cached,
                "failed": prefetch_result.failed,
            }
        )
    classifier = build_bioclip_row_classifier(
        model_registry_path=args.model_registry,
        cache_root=image_cache_root,
        cache_image=build_manifest_cache_image(manifest_path=image_cache_manifest),
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
