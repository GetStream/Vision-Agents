"""RAG type definitions for documents, chunks, and retrieval results."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Document:
    """A document to be ingested into the RAG system.

    Args:
        content: The text content of the document.
        metadata: Optional metadata (e.g., source file, title, author).
        id: Optional unique identifier. Auto-generated if not provided.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            import uuid

            self.id = str(uuid.uuid4())


@dataclass
class Chunk:
    """A chunk of text extracted from a document.

    Args:
        content: The text content of the chunk.
        document_id: ID of the source document.
        index: Position of this chunk within the document.
        metadata: Optional metadata inherited from document plus chunk-specific data.
        start_char: Starting character position in the original document.
        end_char: Ending character position in the original document.
    """

    content: str
    document_id: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    @property
    def id(self) -> str:
        """Generate a unique ID for this chunk."""
        return f"{self.document_id}_{self.index}"


@dataclass
class RetrievalResult:
    """A result from a RAG retrieval query.

    Args:
        content: The retrieved text content.
        score: Relevance score (higher is more relevant).
        document_id: ID of the source document.
        chunk_index: Index of the chunk within the document.
        metadata: Metadata from the chunk/document.
        citation: Optional citation information for the source.
    """

    content: str
    score: float
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    citation: Optional[str] = None

    def format_citation(self) -> str:
        """Format citation for display in responses."""
        if self.citation:
            return self.citation

        source = self.metadata.get("source") or self.metadata.get("filename")
        if source:
            if self.chunk_index is not None:
                return f"[{source}, chunk {self.chunk_index}]"
            return f"[{source}]"

        if self.document_id:
            return f"[doc:{self.document_id[:8]}]"

        return "[unknown source]"
