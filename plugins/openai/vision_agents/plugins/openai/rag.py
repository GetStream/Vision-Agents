"""OpenAI Vector Store RAG provider.

This module provides a RAG implementation using OpenAI's Vector Store API,
which is a managed vector database for file search and retrieval.
"""

import logging
import mimetypes
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from vision_agents.core.rag import (
    Document,
    RAGDocumentAddedEvent,
    RAGFileAddedEvent,
    RAGProvider,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


# Supported file types for OpenAI Vector Store
SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".html",
    ".json",
    ".pdf",
    ".doc",
    ".docx",
    ".pptx",
    ".py",
    ".js",
    ".ts",
    ".c",
    ".cpp",
    ".java",
    ".rb",
    ".go",
    ".rs",
}


class OpenAIVectorStoreRAG(RAGProvider):
    """RAG provider using OpenAI's Vector Store API.

    This provider uses OpenAI's managed vector store which handles:
    - File storage and indexing
    - Automatic chunking
    - Embedding generation
    - Vector search with relevance scores

    Example:
        ```python
        from vision_agents.plugins.openai import OpenAIVectorStoreRAG, LLM

        # Create RAG provider
        rag = OpenAIVectorStoreRAG(store_name="my-knowledge-base")

        # Add files
        await rag.add_file("docs/manual.pdf")
        await rag.add_file("docs/faq.md")

        # Attach to LLM for automatic context injection
        llm = LLM(model="gpt-4o")
        llm.set_rag_provider(rag)

        # Queries will now be augmented with relevant context
        response = await llm.simple_response("How do I reset my password?")
        ```
    """

    def __init__(
        self,
        store_id: Optional[str] = None,
        store_name: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[AsyncOpenAI] = None,
        top_k: int = 5,
    ):
        """Initialize the OpenAI Vector Store RAG provider.

        Args:
            store_id: ID of an existing vector store to use.
                If not provided, a new store will be created on first use.
            store_name: Display name for the vector store (used when creating new).
            api_key: Optional API key. By default loads from OPENAI_API_KEY.
            client: Optional async OpenAI client. By default creates a new client.
            top_k: Default number of results to retrieve.
        """
        super().__init__()
        self._store_id = store_id
        self._store_name = store_name or "vision-agents-rag-store"
        self._top_k = top_k
        self._file_ids: dict[str, str] = {}  # Maps document_id -> file_id

        if client is not None:
            self._client = client
        elif api_key is not None:
            self._client = AsyncOpenAI(api_key=api_key)
        else:
            self._client = AsyncOpenAI()

    async def _ensure_store(self) -> str:
        """Ensure a vector store exists, creating one if needed."""
        if self._store_id is not None:
            return self._store_id

        # Create new store
        store = await self._client.vector_stores.create(name=self._store_name)
        self._store_id = store.id
        logger.info(f"Created new vector store: {self._store_id}")
        return self._store_id

    @property
    def store_id(self) -> Optional[str]:
        """Get the ID of the vector store."""
        return self._store_id

    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type for a file."""
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

    async def add_documents(self, documents: list[Document]) -> None:
        """Ingest documents into the knowledge base.

        For OpenAI Vector Store, documents are converted to temporary files
        and uploaded to the store.

        Args:
            documents: List of documents to add.
        """
        import tempfile
        import os

        for doc in documents:
            # Create a temporary file with the document content
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(doc.content)
                temp_path = f.name

            try:
                doc_id = doc.id or ""
                file_id = await self.add_file(
                    temp_path,
                    metadata={
                        "document_id": doc_id,
                        "source": "document",
                        **doc.metadata,
                    },
                )
                self._file_ids[doc_id] = file_id

                self.events.send(
                    RAGDocumentAddedEvent(
                        document_id=doc_id,
                        metadata=doc.metadata,
                        chunk_count=1,  # OpenAI handles chunking internally
                    )
                )
            finally:
                os.unlink(temp_path)

    async def add_file(self, file_path: str, metadata: Optional[dict] = None) -> str:
        """Ingest a file into the knowledge base.

        Args:
            file_path: Path to the file to ingest.
            metadata: Optional metadata to associate with the file.

        Returns:
            ID of the ingested file.
        """
        store_id = await self._ensure_store()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if file type is supported
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(
                f"File type {path.suffix} may not be fully supported by OpenAI Vector Store"
            )

        # Upload file and add to vector store
        with open(file_path, "rb") as f:
            # Upload and poll waits for the file to be processed
            vector_store_file = await self._client.vector_stores.files.upload_and_poll(
                vector_store_id=store_id,
                file=f,
            )

        file_id = vector_store_file.id
        logger.info(f"Uploaded file to vector store: {file_path} -> {file_id}")

        self.events.send(
            RAGFileAddedEvent(
                file_path=file_path,
                file_id=file_id,
                metadata=metadata or {},
            )
        )

        return file_id

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
        store_id = await self._ensure_store()

        # Search the vector store
        search_results = await self._client.vector_stores.search(
            vector_store_id=store_id,
            query=query,
            max_num_results=top_k,
        )

        results = []
        async for result in search_results:
            # Extract content from the result
            content_parts = []
            for content_item in result.content:
                if content_item.type == "text":
                    content_parts.append(content_item.text)

            content = "\n".join(content_parts)

            results.append(
                RetrievalResult(
                    content=content,
                    score=result.score,
                    document_id=result.file_id,
                    metadata={
                        "filename": result.filename,
                        "file_id": result.file_id,
                        **(result.attributes or {}),
                    },
                    citation=f"[{result.filename}]",
                )
            )

        return results

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the knowledge base.

        Args:
            document_id: ID of the document to delete.

        Returns:
            True if deleted, False if not found.
        """
        file_id = self._file_ids.get(document_id)
        if not file_id or not self._store_id:
            return False

        try:
            await self._client.vector_stores.files.delete(
                vector_store_id=self._store_id,
                file_id=file_id,
            )
            del self._file_ids[document_id]
            logger.info(f"Deleted file from vector store: {file_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete file: {e}")
            return False

    async def clear(self) -> None:
        """Clear all documents by deleting the vector store."""
        if self._store_id:
            try:
                await self._client.vector_stores.delete(
                    vector_store_id=self._store_id,
                )
                logger.info(f"Deleted vector store: {self._store_id}")
            except Exception as e:
                logger.warning(f"Failed to delete vector store: {e}")

        self._store_id = None
        self._file_ids.clear()
