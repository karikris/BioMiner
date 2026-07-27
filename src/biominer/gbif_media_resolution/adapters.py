from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Protocol, runtime_checkable
from urllib.parse import quote, urlsplit

from biominer.gbif_media_resolution.models import ResolutionInput


@dataclass(frozen=True, slots=True)
class AdapterDiscovery:
    request_url: str | None
    direct_candidates: tuple[str, ...] = ()
    accept: str = "application/json"


@runtime_checkable
class ProviderURLResolver(Protocol):
    adapter_id: str
    version: str

    def supports(self, item: ResolutionInput) -> bool: ...

    def discovery(self, item: ResolutionInput) -> AdapterDiscovery: ...

    def parse(self, item: ResolutionInput, content: bytes) -> tuple[str, ...]: ...


class INaturalistPhotoAdapter:
    adapter_id = "inaturalist_photo"
    version = "inaturalist-photo-adapter/v1"
    _path = re.compile(r"^/photos/(?P<photo_id>[0-9]+)(?:/[^/]*)?/?$", re.IGNORECASE)

    def supports(self, item: ResolutionInput) -> bool:
        parsed = urlsplit(item.media_references)
        return (parsed.hostname or "").casefold() in {
            "inaturalist.org",
            "www.inaturalist.org",
        } and self._path.fullmatch(parsed.path) is not None

    def discovery(self, item: ResolutionInput) -> AdapterDiscovery:
        match = self._path.fullmatch(urlsplit(item.media_references).path)
        if match is None:  # pragma: no cover - guarded by supports.
            return AdapterDiscovery(None)
        photo_id = match.group("photo_id")
        return AdapterDiscovery(
            None,
            (
                "https://inaturalist-open-data.s3.amazonaws.com/"
                f"photos/{photo_id}/large.jpg"
            ),
        )

    def parse(self, item: ResolutionInput, content: bytes) -> tuple[str, ...]:
        del item, content
        return ()


class FlickrOEmbedAdapter:
    adapter_id = "flickr_oembed"
    version = "flickr-oembed-adapter/v1"
    _path = re.compile(r"^/photos/[^/]+/(?P<photo_id>[0-9]+)(?:/.*)?$", re.IGNORECASE)

    def supports(self, item: ResolutionInput) -> bool:
        parsed = urlsplit(item.media_references)
        return (parsed.hostname or "").casefold() in {"flickr.com", "www.flickr.com"} and self._path.fullmatch(parsed.path) is not None

    def discovery(self, item: ResolutionInput) -> AdapterDiscovery:
        endpoint = (
            "https://www.flickr.com/services/oembed/?format=json&url="
            + quote(item.media_references, safe="")
        )
        return AdapterDiscovery(endpoint)

    def parse(self, item: ResolutionInput, content: bytes) -> tuple[str, ...]:
        del item
        try:
            payload = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return ()
        if not isinstance(payload, dict):
            return ()
        candidates: list[str] = []
        for key in ("url", "thumbnail_url"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        return tuple(dict.fromkeys(candidates))


DEFAULT_PROVIDER_ADAPTERS: tuple[ProviderURLResolver, ...] = (
    INaturalistPhotoAdapter(),
    FlickrOEmbedAdapter(),
)
DEFAULT_PROVIDER_ADAPTER_VERSIONS = tuple(
    (adapter.adapter_id, adapter.version) for adapter in DEFAULT_PROVIDER_ADAPTERS
)


__all__ = [
    "AdapterDiscovery",
    "DEFAULT_PROVIDER_ADAPTERS",
    "DEFAULT_PROVIDER_ADAPTER_VERSIONS",
    "FlickrOEmbedAdapter",
    "INaturalistPhotoAdapter",
    "ProviderURLResolver",
]
