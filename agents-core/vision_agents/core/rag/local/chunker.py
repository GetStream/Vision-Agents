"""Text chunking strategies for local RAG."""

import abc
import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    """A chunk of text with position information."""

    content: str
    start: int
    end: int
    index: int


class Chunker(abc.ABC):
    """Abstract base class for text chunking strategies."""

    @abc.abstractmethod
    def chunk(self, text: str) -> list[TextChunk]:
        """Split text into chunks.

        Args:
            text: The text to chunk.

        Returns:
            List of text chunks with position information.
        """


class FixedSizeChunker(Chunker):
    """Chunk text into fixed-size pieces with optional overlap.

    Example:
        ```python
        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk("Long document text...")
        ```
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """Initialize the fixed-size chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            overlap: Number of characters to overlap between chunks.
        """
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, text: str) -> list[TextChunk]:
        """Split text into fixed-size chunks."""
        if not text:
            return []

        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append(
                TextChunk(
                    content=chunk_text,
                    start=start,
                    end=end,
                    index=index,
                )
            )

            # Move to next chunk, accounting for overlap
            start = end - self._overlap if end < len(text) else end
            index += 1

        return chunks


class SentenceChunker(Chunker):
    """Chunk text by sentences, respecting a maximum chunk size.

    This chunker tries to keep sentences together while staying
    within the maximum chunk size.

    Example:
        ```python
        chunker = SentenceChunker(max_chunk_size=500)
        chunks = chunker.chunk("First sentence. Second sentence. Third sentence.")
        ```
    """

    # Regex pattern for sentence boundaries
    _SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_chunk_size: int = 500, min_chunk_size: int = 100):
        """Initialize the sentence chunker.

        Args:
            max_chunk_size: Maximum size of each chunk in characters.
            min_chunk_size: Minimum size before starting a new chunk.
        """
        self._max_chunk_size = max_chunk_size
        self._min_chunk_size = min_chunk_size

    def chunk(self, text: str) -> list[TextChunk]:
        """Split text into sentence-based chunks."""
        if not text:
            return []

        # Split into sentences
        sentences = self._SENTENCE_PATTERN.split(text)

        chunks = []
        current_chunk = ""
        current_start = 0
        index = 0
        position = 0

        for sentence in sentences:
            sentence_with_space = sentence + " "

            # If adding this sentence would exceed max size
            if (
                len(current_chunk) + len(sentence_with_space) > self._max_chunk_size
                and len(current_chunk) >= self._min_chunk_size
            ):
                # Save current chunk
                chunks.append(
                    TextChunk(
                        content=current_chunk.strip(),
                        start=current_start,
                        end=position,
                        index=index,
                    )
                )
                index += 1
                current_chunk = sentence_with_space
                current_start = position
            else:
                current_chunk += sentence_with_space

            position += len(sentence_with_space)

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(
                TextChunk(
                    content=current_chunk.strip(),
                    start=current_start,
                    end=len(text),
                    index=index,
                )
            )

        return chunks
