import logging
import re
from pathlib import Path

__all__ = ["Instructions", "InstructionsReadError"]


logger = logging.getLogger(__name__)

_MD_PATTERN = re.compile(r"@([^\s@]+)")


class InstructionsReadError(Exception): ...


class Instructions:
    """
    Container for parsed instructions with input text and markdown files.

    Attributes:
        input_text: Input text that may contain @ mentioned markdown files.
        full_reference: Full reference that includes input text and contents of @ mentioned markdown files.
    """

    def __init__(self, input_text: str = "", base_dir: str | Path | None = None):
        """
        Initialize Instructions object.

        Args:
            input_text: Input text that may contain @ mentioned markdown files.
                Ignores files starting with ".", non-md files, and files outside the base directory.

            base_dir: Base directory to search for markdown files.
                      Defaults to the current working directory (Path.cwd()) at runtime.
        """
        self._base_dir = (
            Path.cwd().resolve() if base_dir is None else Path(base_dir).resolve()
        )
        self.input_text = input_text
        self.full_reference = self._extract_full_reference()

    def _extract_full_reference(self) -> str:
        """
        Parse instructions from an input text string, extracting @ mentioned markdown files and their contents.
        """
        matches = _MD_PATTERN.findall(self.input_text)

        markdown_contents = {}
        markdown_lines = [self.input_text.rstrip()]

        for match in matches:
            try:
                content = self._read_md_file(file_path=match)
                markdown_contents[f"@{match}"] = content
            except InstructionsReadError as e:
                logger.warning(f"Could not resolve @{match}: {e}")
                markdown_contents[f"@{match}"] = (
                    f"*(Warning: File `{match}` not found or inaccessible)*"
                )

        if markdown_contents:
            markdown_lines.append("\n\n## Referenced Documentation:")
            for filename, content in markdown_contents.items():
                markdown_lines.append(f"\n### {filename}")
                markdown_lines.append(content or "*(File is empty)*")

        return "\n".join(markdown_lines)

    def _read_md_file(self, file_path: str | Path) -> str:
        """
        Synchronous helper to read a markdown file.
        """
        file_path = Path(file_path)
        full_path = (
            file_path.resolve()
            if file_path.is_absolute()
            else (self._base_dir / file_path).resolve()
        )

        skip_reason = ""
        if not full_path.exists():
            skip_reason = "file not found"
        elif not full_path.is_file():
            skip_reason = "path is not a file"
        elif full_path.name.startswith("."):
            skip_reason = 'filename cannot start with "."'
        elif full_path.suffix != ".md":
            skip_reason = "file is not .md"
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
