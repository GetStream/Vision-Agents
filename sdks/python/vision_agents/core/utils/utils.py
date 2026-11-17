import asyncio
import contextlib
import importlib.metadata
import inspect
import logging
import os
import tempfile
from typing import Awaitable, Callable, Dict, Optional, ParamSpec, TypeVar, Union

import httpx

logger = logging.getLogger(__name__)

_DOWNLOAD_ATTEMPTS = 3


T = TypeVar("T")
P = ParamSpec("P")


# Type alias for markdown file contents: maps filename to file content
MarkdownFileContents = Dict[str, str]

# Cache version at module load time to avoid blocking I/O during async operations
_VISION_AGENTS_VERSION: str | None = None


def _load_version() -> str:
    """Load version once at module import time."""
    try:
        return importlib.metadata.version("vision-agents")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


_VISION_AGENTS_VERSION = _load_version()


def get_vision_agents_version() -> Optional[str]:
    """
    Get the installed vision-agents package version.

    Returns:
        Version string, or "unknown" if not available.
    """
    return _VISION_AGENTS_VERSION or "unknown"


async def ensure_model(path: str, url: str) -> str:
    """
    Download a model file asynchronously if it doesn't exist.

    A dropped connection or a 5xx is tried again. The file appears whole or not at all,
    so processes downloading the same model at once never read a partial one.

    Args:
        path: Local path where the model should be saved
        url: URL to download the model from

    Returns:
        The path to the model file

    Raises:
        RuntimeError: If the model could not be downloaded.
    """

    if not await asyncio.to_thread(os.path.exists, path):
        model_name = os.path.basename(path)
        logger.info(f"Downloading {model_name}...")
        content = await _download(model_name, url)
        await asyncio.to_thread(_write_whole, path, content)
        logger.info(f"{model_name} downloaded.")

    return path


async def _download(model_name: str, url: str) -> bytes:
    attempt = 1
    while True:
        try:
            async with httpx.AsyncClient(
                timeout=300.0, follow_redirects=True
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.content
        except httpx.HTTPError as e:
            transient = (
                not isinstance(e, httpx.HTTPStatusError)
                or e.response.status_code >= 500
            )
            if not transient or attempt == _DOWNLOAD_ATTEMPTS:
                raise RuntimeError(f"Failed to download {model_name}: {e}") from e
            logger.warning(f"Failed to download {model_name}, retrying: {e}")
            await asyncio.sleep(attempt)
            attempt += 1


def _write_whole(path: str, content: bytes) -> None:
    fd, partial = tempfile.mkstemp(
        dir=os.path.dirname(path) or ".",
        prefix=f".{os.path.basename(path)}.",
        suffix=".part",
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


async def await_or_run(
    fn: Callable[P, Union[T, Awaitable[T]]], *args: P.args, **kwargs: P.kwargs
) -> T:
    """
    Run a callable that may be synchronous or asynchronous.
    Automatically awaits if the result is awaitable.
    """
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def cancel_and_wait(*futures: asyncio.Future) -> None:
    """Cancel one or more async tasks/futures and wait for them to complete.

    Args:
        *futures: Futures or Tasks to cancel.
    """
    for fut in futures:
        if not fut.done():
            fut.cancel()

    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.shield(asyncio.gather(*futures, return_exceptions=True))
