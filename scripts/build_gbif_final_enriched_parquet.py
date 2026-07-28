from __future__ import annotations

import argparse
import json
from pathlib import Path

from biominer.gbif_final.pipeline import build_final_parquet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-parquet", type=Path, required=True)
    parser.add_argument("--pre-temporal-parquet", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--source-assertions", type=Path)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--producer-git-sha", required=True)
    args = parser.parse_args()
    manifest = build_final_parquet(
        temporal_parquet=args.temporal_parquet,
        pre_temporal_parquet=args.pre_temporal_parquet,
        registry_dir=args.registry_dir,
        source_assertions_path=args.source_assertions,
        quality_dir=args.quality_dir,
        output_dir=args.output_dir,
        producer_git_sha=args.producer_git_sha,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
