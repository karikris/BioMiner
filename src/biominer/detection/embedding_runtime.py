"""Opt-in runtime checks and bounded one-image EmbeddingGemma 2 smoke evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from biominer.detection.detector_base import DecodedImage
from biominer.detection.embeddinggemma2 import EmbeddingGemma2Config, EmbeddingGemma2Router, runtime_identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("runtime-check", "prefetch", "smoke"))
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--image")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    router = None
    try:
        config = EmbeddingGemma2Config.from_dict(json.loads(args.config_json))
        runtime = runtime_identity()
        if any(value is None for value in runtime["packages"].values()):
            raise RuntimeError("EmbeddingGemma 2 dependencies are unavailable; install BioMiner's embeddinggemma2 extra")
        from packaging.version import Version

        if Version(runtime["packages"]["sentence-transformers"]) < Version("6.1.0"):
            raise RuntimeError("EmbeddingGemma 2 requires sentence-transformers >=6.1.0")
        if Version(runtime["packages"]["transformers"]) < Version("5.18.0"):
            raise RuntimeError("EmbeddingGemma 2 requires transformers >=5.18.0")
        router = EmbeddingGemma2Router(config=config)
        receipt = {**router.metadata(), "model_weights_loaded": False,
                   "scientific_scope": "Runtime availability; biological accuracy is not assessed."}
        if args.operation != "runtime-check":
            router.load()
            receipt["model_weights_loaded"] = True
        if args.operation == "smoke":
            if not args.image or not args.output_dir:
                raise ValueError("smoke requires --image and --output-dir")
            from PIL import Image

            with Image.open(args.image) as image:
                rgb = image.convert("RGB")
                decoded = DecodedImage(rgb.width, rgb.height, "RGB", rgb.tobytes())
            evidence = router.route_batch([decoded])[0]
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            if any(output.iterdir()):
                raise ValueError("smoke output directory must be empty")
            artifact = output / "image_embedding_evidence.json"
            artifact.write_text(json.dumps(evidence, sort_keys=True, allow_nan=False) + "\n")
            receipt.update({"scientific_scope": "One-image inference smoke evidence; biological performance is not assessed.",
                            "input": str(Path(args.image).resolve()), "image_content_hash": evidence["image_content_hash"],
                            "evidence_path": str(artifact.resolve()), "occurrence_release": "not_assessed"})
            (output / "smoke_manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    finally:
        if router is not None:
            router.close()


if __name__ == "__main__":
    raise SystemExit(main())
