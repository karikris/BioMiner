from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future
from inspect import signature
from typing import TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def bounded_map_ordered(
    executor: Executor,
    function: Callable[[InputT], OutputT],
    items: Iterable[InputT],
    *,
    buffersize: int,
) -> Iterator[OutputT]:
    """Map items with bounded submitted work while preserving input order."""
    if buffersize < 1:
        raise ValueError("buffersize must be >= 1")
    if _executor_map_supports_buffersize(executor):
        yield from executor.map(function, items, buffersize=buffersize)
        return
    yield from _fallback_bounded_map_ordered(executor, function, items, buffersize=buffersize)


def _executor_map_supports_buffersize(executor: Executor) -> bool:
    return "buffersize" in signature(executor.map).parameters


def _fallback_bounded_map_ordered(
    executor: Executor,
    function: Callable[[InputT], OutputT],
    items: Iterable[InputT],
    *,
    buffersize: int,
) -> Iterator[OutputT]:
    iterator = iter(items)
    pending: list[Future[OutputT]] = []

    def fill() -> None:
        while len(pending) < buffersize:
            try:
                item = next(iterator)
            except StopIteration:
                return
            pending.append(executor.submit(function, item))

    fill()
    while pending:
        future = pending.pop(0)
        yield future.result()
        fill()
