"""Vector store implementations for local RAG."""

import abc
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class VectorStoreEntry:
    """An entry in the vector store."""

    id: str
    vector: list[float]
    content: str
    metadata: dict


class VectorStore(abc.ABC):
    """Abstract base class for vector stores.

    Vector stores handle storage and retrieval of embedding vectors
    with their associated content and metadata.
    """

    @abc.abstractmethod
    async def add(
        self,
        id: str,
        vector: list[float],
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a vector to the store.

        Args:
            id: Unique identifier for the entry.
            vector: The embedding vector.
            content: The text content.
            metadata: Optional metadata.
        """

    @abc.abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[tuple[VectorStoreEntry, float]]:
        """Search for similar vectors.

        Args:
            query_vector: The query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            List of (entry, similarity_score) tuples, sorted by score descending.
        """

    @abc.abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete an entry from the store.

        Args:
            id: ID of the entry to delete.

        Returns:
            True if deleted, False if not found.
        """

    @abc.abstractmethod
    async def clear(self) -> None:
        """Clear all entries from the store."""

    @abc.abstractmethod
    async def count(self) -> int:
        """Return the number of entries in the store."""


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store using cosine similarity.

    This is suitable for small datasets (< 10k documents).
    For larger datasets, consider using FAISS or a dedicated vector database.

    Example:
        ```python
        store = InMemoryVectorStore()
        await store.add("doc1", [0.1, 0.2, 0.3], "Hello world", {"source": "test"})
        results = await store.search([0.1, 0.2, 0.3], top_k=5)
        ```
    """

    def __init__(self):
        """Initialize an empty in-memory vector store."""
        self._entries: dict[str, VectorStoreEntry] = {}

    async def add(
        self,
        id: str,
        vector: list[float],
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a vector to the store."""
        self._entries[id] = VectorStoreEntry(
            id=id,
            vector=vector,
            content=content,
            metadata=metadata or {},
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[tuple[VectorStoreEntry, float]]:
        """Search for similar vectors using cosine similarity."""
        if not self._entries:
            return []

        # Calculate cosine similarity for all entries
        scored_entries = []
        for entry in self._entries.values():
            score = self._cosine_similarity(query_vector, entry.vector)
            scored_entries.append((entry, score))

        # Sort by score descending and return top_k
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        return scored_entries[:top_k]

    async def delete(self, id: str) -> bool:
        """Delete an entry from the store."""
        if id in self._entries:
            del self._entries[id]
            return True
        return False

    async def clear(self) -> None:
        """Clear all entries from the store."""
        self._entries.clear()

    async def count(self) -> int:
        """Return the number of entries in the store."""
        return len(self._entries)

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
