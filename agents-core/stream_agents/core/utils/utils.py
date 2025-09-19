import numpy as np
import re
import os
from dataclasses import dataclass
from typing import Dict, Optional


# Type alias for markdown file contents: maps filename to file content
MarkdownFileContents = Dict[str, str]


@dataclass
class Instructions:
    """Container for parsed instructions with input text and markdown files."""
    input_text: str
    base_dir: str
    markdown_contents: MarkdownFileContents  # Maps filename to file content


def to_mono(samples: np.ndarray, num_channels: int) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels."""
    if num_channels == 1:
        return samples
    if samples.size % num_channels != 0:
        raise ValueError(
            f"Invalid sample array size {samples.size} for {num_channels} channels: "
            "Sample array size not divisible by number of channels"
        )
    samples = samples.reshape(-1, num_channels)
    mono_samples = np.mean(samples, axis=1, dtype=np.int16)
    # Ensure we always return an array, not a scalar
    return np.asarray(mono_samples, dtype=np.int16)




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
    import __main__
    # Find all @ mentions that look like markdown files
    # Pattern matches @ followed by filename with .md extension
    markdown_pattern = r'@([^\s@]+\.md)'
    matches = re.findall(markdown_pattern, text)
    
    # Create a dictionary mapping filename to file content
    markdown_contents = {}
    
    # Set base directory for file search
    if base_dir is None:
        base_dir = os.path.dirname(__main__.__file__)
    
    for match in matches:
        # Try to read the markdown file content
        file_path = os.path.join(base_dir, match)
        try:
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    markdown_contents[match] = f.read()
            else:
                # File not found, store empty string
                markdown_contents[match] = ""
        except (OSError, IOError, UnicodeDecodeError):
            # File read error, store empty string
            markdown_contents[match] = ""
    
    return Instructions(
        input_text=text,
        base_dir=base_dir,
        markdown_contents=markdown_contents
    )


