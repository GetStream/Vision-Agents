"""RAG (Retrieval-Augmented Generation) module for Vision Agents."""

from .base import RAGProvider
from .events import (
    RAGDocumentAddedEvent,
    RAGFileAddedEvent,
    RAGRetrievalCompleteEvent,
    RAGRetrievalStartEvent,
)
from .types import Chunk, Document, RetrievalResult

__all__ = [
    "RAGProvider",
    "Document",
    "Chunk",
    "RetrievalResult",
    "RAGRetrievalStartEvent",
    "RAGRetrievalCompleteEvent",
    "RAGDocumentAddedEvent",
    "RAGFileAddedEvent",
]

