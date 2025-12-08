import logging
import os
import re
from pathlib import Path

__all__ = ["Instructions", "InstructionsReadError"]


logger = logging.getLogger(__name__)

_INITIAL_CWD = os.getcwd()

_MD_PATTERN = re.compile(r"@([^\s@]+)")


class InstructionsReadError(Exception): ...


class Instructions:
    """
    Container for parsed instructions with input text and markdown files.

    Attributes:
        input_text: Input text that may contain @ mentioned markdown files.
        full_reference: Full reference that includes input text and contents of @ mentioned markdown files.
    """

    def __init__(self, input_text: str = "", base_dir: str | Path = ""):
        """
        Initialize Instructions object.

        Args:
            input_text: Input text that may contain @ mentioned markdown files.
                Ignores files starting with ".", non-md files, and files outside the base directory.

            base_dir: Base directory to search for markdown files. Default - current working directory.
        """
        self._base_dir = Path(base_dir or _INITIAL_CWD).resolve()
        self.input_text = input_text
        self.full_reference = self._extract_full_reference()

    def _extract_full_reference(self) -> str:
        """
        Parse instructions from an input text string, extracting @ mentioned markdown files and their contents.

        Returns:
            Instructions object containing the input text and file contents
        """

        # Find all @ mentions that look like markdown files
        matches = _MD_PATTERN.findall(self.input_text)

        # Create a dictionary mapping filename to file content
        markdown_contents = {}

        markdown_lines = [self.input_text]
        # Iterate over found @ mentions and try reading instructions from them
        for match in matches:
            # Try to read the markdown file content
            content = self._read_md_file(file_path=match)
            markdown_contents[match] = content

        # Add markdown file contents if any exist
        if markdown_contents:
            markdown_lines.append("\n\n## Referenced Documentation:")
            for filename, content in markdown_contents.items():
                markdown_lines.append(f"\n### {filename}")
                # Only include non-empty content
                markdown_lines.append(content or "*(File is empty)*")
        full_reference = "\n".join(markdown_lines)
        return full_reference

    def _read_md_file(self, file_path: str | Path) -> str:
        """
        Synchronous helper to read a markdown file.

        Args:
            file_path: Absolute or relative path to markdown file.
                Paths outside the base directory are ignored.
        """
        # Resolve the markdown file path
        file_path = Path(file_path)
        full_path = (
            file_path.resolve()
            if file_path.is_absolute()
            else (self._base_dir / file_path).resolve()
        )

        # Check if the path is a file, it exists, and it's a markdown file.
        skip_reason = ""
        if not full_path.exists():
            skip_reason = "file not found"
        elif not full_path.is_file():
            skip_reason = "path is not a file"
        elif full_path.name.startswith("."):
            skip_reason = 'filename cannot start with "."'
        elif full_path.suffix != ".md":
            skip_reason = "file is not .md"
        # The markdown file also must be inside the base_dir
        elif not full_path.is_relative_to(self._base_dir):
            skip_reason = f"path outside the base directory {self._base_dir}"

        if skip_reason:
            raise InstructionsReadError(
                f"Failed to read instructions from {full_path}; reason - {skip_reason}"
            )

        try:
            logger.info(f"Reading instructions from file {full_path}")
            with open(full_path, mode="r") as f:
                return f.read()
        except (OSError, IOError, UnicodeDecodeError) as exc:
            raise InstructionsReadError(
                f"Failed to read instructions from file {full_path}; reason - {exc}"
            ) from exc
