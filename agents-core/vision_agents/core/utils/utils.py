import asyncio
import importlib.metadata
import logging
import os
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)


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

    Args:
        path: Local path where the model should be saved
        url: URL to download the model from

    Returns:
        The path to the model file
    """

    if not os.path.exists(path):
        model_name = os.path.basename(path)
        logger.info(f"Downloading {model_name}...")

        try:
            async with httpx.AsyncClient(
                timeout=300.0, follow_redirects=True
            ) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()

                    # Write file in chunks to avoid loading entire file in memory
                    chunks = []
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        chunks.append(chunk)

                    # Write all chunks to file in thread to avoid blocking event loop
                    def write_file():
                        with open(path, "wb") as f:
                            for chunk in chunks:
                                f.write(chunk)

                    await asyncio.to_thread(write_file)

            logger.info(f"{model_name} downloaded.")
        except httpx.HTTPError as e:
            # Clean up partial download on error
            if os.path.exists(path):
                os.remove(path)
            raise RuntimeError(f"Failed to download {model_name}: {e}")

    return path
