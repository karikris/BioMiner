"""Persistent JSONL worker for the optional EmbeddingGemma 2 runtime."""

from __future__ import annotations

import base64
from collections.abc import Callable
from contextlib import redirect_stdout
import json
import sys
from typing import TextIO

from biominer.detection.detector_base import DecodedImage
from biominer.detection.embeddinggemma2 import (
    EMBEDDINGGEMMA2_PROTOCOL_VERSION,
    EmbeddingGemma2Config,
    EmbeddingGemma2Router,
    _strict_json_loads,
)


def _image(value: object) -> DecodedImage:
    if not isinstance(value, dict):
        raise ValueError("worker image must contain complete RGB image bytes")
    for field in ("width", "height"):
        dimension = value.get(field)
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"worker image {field} must be a positive integer")
    if value.get("mode") != "RGB" or not isinstance(value.get("data_b64"), str):
        raise ValueError("worker image must contain base64 RGB bytes")
    data = base64.b64decode(value["data_b64"], validate=True)
    return DecodedImage(width=value["width"], height=value["height"], mode="RGB", data=data)


def serve(source: TextIO, output: TextIO, *, router_factory: Callable[..., EmbeddingGemma2Router] = EmbeddingGemma2Router) -> None:
    """Serve one immutable configuration; stdout contains protocol JSON only."""
    router = None
    try:
        for line in source:
            request_id = None
            try:
                request = _strict_json_loads(line)
                if not isinstance(request, dict) or request.get("protocol") != EMBEDDINGGEMMA2_PROTOCOL_VERSION:
                    raise ValueError("unsupported EmbeddingGemma 2 worker protocol")
                request_id = request.get("request_id")
                operation = request.get("operation")
                if operation == "shutdown":
                    break
                if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id <= 0:
                    raise ValueError("request_id must be a positive integer")
                response: dict[str, object] = {
                    "protocol": EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": request_id,
                }
                # Third-party loading/progress output must never enter the JSONL channel.
                with redirect_stdout(sys.stderr):
                    if operation == "hello":
                        if router is not None:
                            raise ValueError("worker configuration is immutable after hello")
                        value = request.get("config")
                        if not isinstance(value, dict):
                            raise ValueError("hello requires immutable model configuration")
                        config = EmbeddingGemma2Config.from_dict(value)
                        router = router_factory(config=config)
                    elif operation == "load":
                        if router is None:
                            raise ValueError("hello handshake is required before loading")
                        router.load()
                    elif operation == "route":
                        if router is None:
                            raise ValueError("hello handshake is required before routing")
                        images = request.get("images")
                        if not isinstance(images, list):
                            raise ValueError("route requires a complete RGB image batch")
                        response["results"] = router.route_batch([_image(image) for image in images])
                    else:
                        raise ValueError(f"unsupported worker operation: {operation!r}")
                    assert router is not None
                    response["metadata"] = router.metadata()
            except Exception as exc:
                response = {
                    "protocol": EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": request_id,
                    "error_type": type(exc).__name__, "error": str(exc),
                }
            output.write(json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n")
            output.flush()
    finally:
        if router is not None:
            with redirect_stdout(sys.stderr):
                router.close()


def main() -> None:
    serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
