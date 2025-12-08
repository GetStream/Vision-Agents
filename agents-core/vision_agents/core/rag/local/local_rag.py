"""Local RAG provider with pluggable components."""

import logging
import uuid
from pathlib import Path
from typing import Optional

from vision_agents.core.rag.base import RAGProvider
from vision_agents.core.rag.events import RAGDocumentAddedEvent, RAGFileAddedEvent
from vision_agents.core.rag.types import Document, RetrievalResult

from .chunker import Chunker, FixedSizeChunker
from .embeddings import EmbeddingProvider
from .vector_store import InMemoryVectorStore, VectorStore

logger = logging.getLogger(__name__)


class LocalRAG(RAGProvider):
    """Local RAG provider with pluggable embeddings and vector store.

    This provider allows you to build a fully local RAG pipeline with
    customizable components for embedding generation, vector storage,
    and text chunking.

    Example:
        ```python
        from vision_agents.core.rag.local import (
            LocalRAG,
            OpenAIEmbeddings,
            InMemoryVectorStore,
            SentenceChunker,
        )

        # Create with default components (OpenAI embeddings, in-memory store)
        rag = LocalRAG(
            embeddings=OpenAIEmbeddings(),
            vector_store=InMemoryVectorStore(),
            chunker=SentenceChunker(max_chunk_size=500),
        )

        # Add documents
        await rag.add_documents([
            Document(content="Your document text here...", id="doc1"),
        ])

        # Or add files directly
        await rag.add_file("path/to/document.txt")

        # Search
        results = await rag.search("your query", top_k=5)
        ```
    """

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vector_store: Optional[VectorStore] = None,
        chunker: Optional[Chunker] = None,
    ):
        """Initialize the local RAG provider.

        Args:
            embeddings: The embedding provider to use for vectorization.
            vector_store: The vector store for storage and retrieval.
                Defaults to InMemoryVectorStore.
            chunker: The text chunking strategy.
                Defaults to FixedSizeChunker with 500 char chunks.
        """
        super().__init__()
        self._embeddings = embeddings
        self._vector_store = vector_store or InMemoryVectorStore()
        self._chunker = chunker or FixedSizeChunker(chunk_size=500, overlap=50)
        self._document_chunks: dict[str, list[str]] = {}  # doc_id -> [chunk_ids]

    @property
    def embeddings(self) -> EmbeddingProvider:
        """Get the embedding provider."""
        return self._embeddings

    @property
    def vector_store(self) -> VectorStore:
        """Get the vector store."""
        return self._vector_store

    async def add_documents(self, documents: list[Document]) -> None:
        """Ingest documents into the knowledge base.

        Documents are chunked, embedded, and stored in the vector store.

        Args:
            documents: List of documents to add.
        """
        for doc in documents:
            doc_id = doc.id or str(uuid.uuid4())

            # Chunk the document
            chunks = self._chunker.chunk(doc.content)

            if not chunks:
                logger.warning(f"No chunks generated for document {doc_id}")
                continue

            # Generate embeddings for all chunks
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self._embeddings.embed_batch(chunk_texts)

            # Store chunks in vector store
            chunk_ids = []
            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = f"{doc_id}_chunk_{chunk.index}"
                chunk_ids.append(chunk_id)

                await self._vector_store.add(
                    id=chunk_id,
                    vector=embedding,
                    content=chunk.content,
                    metadata={
                        "document_id": doc_id,
                        "chunk_index": chunk.index,
                        "start_position": chunk.start,
                        "end_position": chunk.end,
                        **doc.metadata,
                    },
                )

            self._document_chunks[doc_id] = chunk_ids

            logger.info(f"Added document {doc_id} with {len(chunks)} chunks")

            self.events.send(
                RAGDocumentAddedEvent(
                    document_id=doc_id,
                    metadata=doc.metadata,
                    chunk_count=len(chunks),
                )
            )

    async def add_file(self, file_path: str, metadata: Optional[dict] = None) -> str:
        """Ingest a file into the knowledge base.

        Args:
            file_path: Path to the file to ingest.
            metadata: Optional metadata to associate with the file.

        Returns:
            ID of the ingested document.
        """
        import aiofiles
        import aiofiles.os

        path = Path(file_path)
        if not await aiofiles.os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content asynchronously
        async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
            content = await f.read()

        # Create document
        doc_id = str(uuid.uuid4())
        doc_metadata = {
            "filename": path.name,
            "file_path": file_path,
            **(metadata or {}),
        }

        doc = Document(
            id=doc_id,
            content=content,
            metadata=doc_metadata,
        )

        await self.add_documents([doc])

        self.events.send(
            RAGFileAddedEvent(
                file_path=file_path,
                file_id=doc_id,
                metadata=doc_metadata,
            )
        )

        return doc_id

    async def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.

        Returns:
            List of retrieval results ordered by relevance.
        """
        # Generate query embedding
        query_embedding = await self._embeddings.embed(query)

        # Search vector store
        results = await self._vector_store.search(query_embedding, top_k=top_k)

        # Convert to RetrievalResult
        retrieval_results = []
        for entry, score in results:
            retrieval_results.append(
                RetrievalResult(
                    content=entry.content,
                    score=score,
                    document_id=entry.metadata.get("document_id", entry.id),
                    metadata=entry.metadata,
                    citation=self._format_citation(entry.metadata),
                )
            )

        return retrieval_results

    def _format_citation(self, metadata: dict) -> str:
        """Format a citation from metadata."""
        if "filename" in metadata:
            return f"[{metadata['filename']}]"
        if "document_id" in metadata:
            return f"[Document: {metadata['document_id']}]"
        return "[Unknown source]"

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks from the knowledge base.

        Args:
            document_id: ID of the document to delete.

        Returns:
            True if deleted, False if not found.
        """
        chunk_ids = self._document_chunks.get(document_id)
        if not chunk_ids:
            return False

        for chunk_id in chunk_ids:
            await self._vector_store.delete(chunk_id)

        del self._document_chunks[document_id]
        logger.info(f"Deleted document {document_id}")
        return True

    async def clear(self) -> None:
        """Clear all documents from the knowledge base."""
        await self._vector_store.clear()
        self._document_chunks.clear()
        logger.info("Cleared all documents from local RAG")
