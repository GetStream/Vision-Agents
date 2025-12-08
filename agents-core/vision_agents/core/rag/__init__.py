"""RAG (Retrieval-Augmented Generation) module for Vision Agents."""

from .base import RAGProvider
from .events import (
    RAGDocumentAddedEvent,
    RAGFileAddedEvent,
    RAGRetrievalCompleteEvent,
    RAGRetrievalStartEvent,
)
from .local import (
    Chunker,
    EmbeddingProvider,
    FixedSizeChunker,
    InMemoryVectorStore,
    LocalRAG,
    OpenAIEmbeddings,
    SentenceChunker,
    VectorStore,
)
from .types import Chunk, Document, RetrievalResult

__all__ = [
    # Base
    "RAGProvider",
    "Document",
    "Chunk",
    "RetrievalResult",
    # Events
    "RAGRetrievalStartEvent",
    "RAGRetrievalCompleteEvent",
    "RAGDocumentAddedEvent",
    "RAGFileAddedEvent",
    # Local RAG
    "LocalRAG",
    "EmbeddingProvider",
    "OpenAIEmbeddings",
    "VectorStore",
    "InMemoryVectorStore",
    "Chunker",
    "FixedSizeChunker",
    "SentenceChunker",
]
