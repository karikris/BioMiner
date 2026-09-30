"""Public dependency behavior required by the reviewed security upgrades."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import ssl
from zipfile import ZipFile

import anyio
from anyio.abc import ByteStream
from anyio.streams.tls import TLSStream
from lxml import etree
import pytest


class _ClientHelloCaptured(Exception):
    """Stop the TLS handshake after its first outbound message."""


class _ClientHelloCaptureStream(ByteStream):
    def __init__(self) -> None:
        self.outbound = bytearray()

    async def send(self, item: bytes) -> None:
        self.outbound.extend(item)
        raise _ClientHelloCaptured

    async def receive(self, max_bytes: int = 65536) -> bytes:
        raise AssertionError("The handshake must stop before receiving peer data")

    async def send_eof(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


def test_anyio_tls_sends_idna2008_hostname() -> None:
    stream = _ClientHelloCaptureStream()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    async def capture_client_hello() -> None:
        with pytest.raises(_ClientHelloCaptured):
            await TLSStream.wrap(
                stream,
                hostname="faß.example",
                ssl_context=context,
            )

    anyio.run(capture_client_hello, backend="asyncio")

    # Inspect the actual TLS ClientHello sent through the public stream API.
    assert b"xn--fa-hia.example" in stream.outbound
    assert b"fass.example" not in stream.outbound


def test_lxml_default_parser_rejects_external_parameter_entity(tmp_path: Path) -> None:
    external_dtd = tmp_path / "external.dtd"
    external_dtd.write_text('<!ENTITY payload "external DTD was loaded">', encoding="utf-8")
    document = (
        '<!DOCTYPE root [<!ENTITY % external SYSTEM "'
        f'{external_dtd.as_uri()}">%external;]>'
        "<root>&payload;</root>"
    ).encode("utf-8")

    with pytest.raises(etree.XMLSyntaxError):
        etree.fromstring(document, parser=etree.XMLParser())


def test_skops_rejects_future_archive_protocol() -> None:
    sio = pytest.importorskip("skops.io")
    original = sio.dumps(None)
    modified = BytesIO()

    with ZipFile(BytesIO(original), "r") as source, ZipFile(modified, "w") as target:
        for member in source.infolist():
            contents = source.read(member)
            if member.filename == "schema.json":
                schema = json.loads(contents)
                schema["protocol"] = 2**31 - 1
                contents = json.dumps(schema).encode("utf-8")
            target.writestr(member, contents)

    with pytest.raises(ValueError, match="protocol"):
        sio.loads(modified.getvalue())
