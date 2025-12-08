"""Base RAG provider interface."""

import abc
import time
from typing import Optional

from vision_agents.core.events.manager import EventManager

from . import events
from .types import Document, RetrievalResult


class RAGProvider(abc.ABC):
    """Abstract base class for RAG (Retrieval-Augmented Generation) providers.

    RAG providers handle document ingestion, storage, and retrieval for
    augmenting LLM responses with relevant context.

    Implementations can be:
    - Provider-native (e.g., Gemini File Search, OpenAI Vector Store)
    - Local (e.g., FAISS, ChromaDB with custom embeddings)
    """

    def __init__(self):
        self.events = EventManager()
        self.events.register_events_from_module(events)

    @abc.abstractmethod
    async def add_documents(self, documents: list[Document]) -> None:
        """Ingest documents into the knowledge base.

        Args:
            documents: List of documents to add.
        """

    @abc.abstractmethod
    async def add_file(self, file_path: str, metadata: Optional[dict] = None) -> str:
        """Ingest a file into the knowledge base.

        Args:
            file_path: Path to the file to ingest.
            metadata: Optional metadata to associate with the file.

        Returns:
            ID of the ingested file/document.
        """

    async def add_files(self, file_paths: list[str]) -> list[str]:
        """Ingest multiple files into the knowledge base.

        Args:
            file_paths: List of file paths to ingest.

        Returns:
            List of IDs for the ingested files.
        """
        ids = []
        for path in file_paths:
            file_id = await self.add_file(path)
            ids.append(file_id)
        return ids

    @abc.abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.

        Returns:
            List of retrieval results ordered by relevance.
        """

    async def search_with_events(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Search with event emission for observability.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.

        Returns:
            List of retrieval results ordered by relevance.
        """
        self.events.send(
            events.RAGRetrievalStartEvent(
                query=query,
                top_k=top_k,
            )
        )

        start_time = time.time()
        results = await self.search(query, top_k)
        elapsed_ms = (time.time() - start_time) * 1000

        self.events.send(
            events.RAGRetrievalCompleteEvent(
                query=query,
                results=results,
                retrieval_time_ms=elapsed_ms,
            )
        )

        return results

    def build_context_prompt(
        self,
        results: list[RetrievalResult],
        include_citations: bool = True,
    ) -> str:
        """Format retrieved results for injection into LLM prompt.

        Args:
            results: List of retrieval results.
            include_citations: Whether to include citation markers.

        Returns:
            Formatted context string to prepend to the user's query.
        """
        if not results:
            return ""

        context_parts = [
            "Use the following context to answer the question. "
            "If the context doesn't contain relevant information, say so.\n"
        ]

        for i, result in enumerate(results, 1):
            citation = f" {result.format_citation()}" if include_citations else ""
            context_parts.append(f"[{i}]{citation}: {result.content}\n")

        return "\n".join(context_parts)

    @abc.abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the knowledge base.

        Args:
            document_id: ID of the document to delete.

        Returns:
            True if deleted, False if not found.
        """

    async def clear(self) -> None:
        """Clear all documents from the knowledge base.

        Default implementation does nothing. Override if supported.
        """
        pass

