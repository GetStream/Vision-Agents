import asyncio
import io
import logging
import re
import os
import importlib.metadata
from dataclasses import dataclass
from typing import Dict, Optional
from PIL import Image
import httpx


# Type alias for markdown file contents: maps filename to file content
MarkdownFileContents = Dict[str, str]


@dataclass
class Instructions:
    """Container for parsed instructions with input text and markdown files."""

    input_text: str
    markdown_contents: MarkdownFileContents  # Maps filename to file content
    base_dir: str = ""  # Base directory for file search, defaults to empty string


def parse_instructions(text: str, base_dir: Optional[str] = None) -> Instructions:
    """
    Parse instructions from a string, extracting @ mentioned markdown files and their contents.

    Args:
        text: Input text that may contain @ mentions of markdown files
        base_dir: Base directory to search for markdown files. If None, uses current working directory.

    Returns:
        Instructions object containing the input text and file contents

    Example:
        >>> text = "Please read @file1.md and @file2.md for context"
        >>> result = parse_instructions(text)
        >>> result.input_text
        "Please read @file1.md and @file2.md for context"
        >>> result.markdown_contents
        {"file1.md": "# File 1 content...", "file2.md": "# File 2 content..."}
    """
    # Find all @ mentions that look like markdown files
    # Pattern matches @ followed by filename with .md extension
    markdown_pattern = r"@([^\s@]+\.md)"
    matches = re.findall(markdown_pattern, text)

    # Create a dictionary mapping filename to file content
    markdown_contents = {}

    # Set base directory for file search
    if base_dir is None:
        base_dir = os.getcwd()

    for match in matches:
        # Try to read the markdown file content
        file_path = os.path.join(base_dir, match)
        try:
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    markdown_contents[match] = f.read()
            else:
                # File not found, store empty string
                markdown_contents[match] = ""
        except (OSError, IOError, UnicodeDecodeError):
            # File read error, store empty string
            markdown_contents[match] = ""

    return Instructions(
        input_text=text, markdown_contents=markdown_contents, base_dir=base_dir
    )


def frame_to_png_bytes(frame) -> bytes:
    """
    Convert a video frame to PNG bytes.

    Args:
        frame: Video frame object that can be converted to an image

    Returns:
        PNG bytes of the frame, or empty bytes if conversion fails
    """
    logger = logging.getLogger(__name__)
    try:
        if hasattr(frame, "to_image"):
            img = frame.to_image()
        else:
            arr = frame.to_ndarray(format="rgb24")
            img = Image.fromarray(arr)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Error converting frame to PNG: {e}")
        return b""


def get_vision_agents_version() -> str:
    """
    Get the installed vision-agents package version.

    Returns:
        Version string, or "unknown" if not available.
    """
    try:
        return importlib.metadata.version("vision-agents")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


async def ensure_model(path: str, url: str) -> str:
    """
    Download a model file asynchronously if it doesn't exist.
    
    Args:
        path: Local path where the model should be saved
        url: URL to download the model from
        
    Returns:
        The path to the model file
    """

    logger = logging.getLogger(__name__)
    if not os.path.exists(path):
        model_name = os.path.basename(path)
        logger.info(f"Downloading {model_name}...")
        
        try:
            async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
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
