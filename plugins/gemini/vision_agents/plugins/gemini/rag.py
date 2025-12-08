"""Gemini File Search RAG provider.

This module provides a RAG implementation using Google's native File Search tool,
which is a fully managed RAG system built directly into the Gemini API.
"""

import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Optional

from google.genai import types
from google.genai.client import AsyncClient, Client

from vision_agents.core.rag import (
    Document,
    RAGDocumentAddedEvent,
    RAGFileAddedEvent,
    RAGProvider,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


# Supported MIME types for Gemini File Search
SUPPORTED_MIME_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".css": "text/css",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".json": "application/json",
    ".xml": "application/xml",
    ".csv": "text/csv",
    ".py": "text/x-python",
    ".java": "text/x-java",
    ".c": "text/x-c",
    ".cpp": "text/x-c++",
    ".h": "text/x-c",
    ".hpp": "text/x-c++",
    ".go": "text/x-go",
    ".rs": "text/x-rust",
    ".rb": "text/x-ruby",
    ".php": "text/x-php",
    ".swift": "text/x-swift",
    ".kt": "text/x-kotlin",
    ".scala": "text/x-scala",
    ".sh": "text/x-shellscript",
    ".bash": "text/x-shellscript",
    ".zsh": "text/x-shellscript",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".toml": "text/x-toml",
    ".ini": "text/x-ini",
    ".cfg": "text/x-ini",
    ".conf": "text/plain",
    ".log": "text/plain",
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class GeminiFileSearchRAG(RAGProvider):
    """RAG provider using Gemini's native File Search tool.

    This provider uses Google's fully managed RAG system which handles:
    - File storage and indexing
    - Automatic chunking
    - Embedding generation
    - Vector search
    - Built-in citations

    Example:
        ```python
        from vision_agents.plugins.gemini import GeminiFileSearchRAG, LLM

        # Create RAG provider
        rag = GeminiFileSearchRAG(store_name="my-knowledge-base")

        # Add files
        await rag.add_file("docs/manual.pdf")
        await rag.add_file("docs/faq.md")

        # Attach to LLM for automatic context injection
        llm = LLM()
        llm.set_rag_provider(rag)

        # Queries will now be augmented with relevant context
        response = await llm.simple_response("How do I reset my password?")
        ```
    """

    def __init__(
        self,
        store_name: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[AsyncClient] = None,
        top_k: int = 5,
    ):
        """Initialize the Gemini File Search RAG provider.

        Args:
            store_name: Name of an existing file search store to use.
                If not provided, a new store will be created on first use.
            api_key: Optional API key. By default loads from GOOGLE_API_KEY.
            client: Optional async Gemini client. By default creates a new client.
            top_k: Default number of results to retrieve.
        """
        super().__init__()
        self._store_name = store_name
        self._store: Optional[types.FileSearchStore] = None
        self._top_k = top_k
        self._file_ids: dict[str, str] = {}  # Maps document_id -> file resource name

        # Use sync client for store management, async for queries
        if client is not None:
            self._async_client = client
            self._sync_client = Client(api_key=api_key)
        else:
            self._sync_client = Client(api_key=api_key)
            self._async_client = self._sync_client.aio

    async def _ensure_store(self) -> types.FileSearchStore:
        """Ensure a file search store exists, creating one if needed."""
        if self._store is not None:
            return self._store

        if self._store_name:
            # Try to get existing store
            try:
                self._store = self._sync_client.file_search_stores.get(
                    name=self._store_name
                )
                logger.info(f"Using existing file search store: {self._store_name}")
                return self._store
            except Exception as e:
                logger.warning(
                    f"Could not find store {self._store_name}, creating new: {e}"
                )

        # Create new store
        self._store = self._sync_client.file_search_stores.create(
            config=types.CreateFileSearchStoreConfig(
                display_name=self._store_name or "vision-agents-rag-store"
            )
        )
        self._store_name = self._store.name
        logger.info(f"Created new file search store: {self._store_name}")
        return self._store

    @property
    def store_name(self) -> Optional[str]:
        """Get the name of the file search store."""
        return self._store_name

    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type for a file."""
        ext = Path(file_path).suffix.lower()
        if ext in SUPPORTED_MIME_TYPES:
            return SUPPORTED_MIME_TYPES[ext]

        # Fall back to mimetypes module
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "text/plain"

    async def add_documents(self, documents: list[Document]) -> None:
        """Ingest documents into the knowledge base.

        For Gemini File Search, documents are converted to temporary files
        and uploaded to the store.

        Args:
            documents: List of documents to add.
        """
        import tempfile

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
                        chunk_count=1,  # Gemini handles chunking internally
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
        await self._ensure_store()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        mime_type = self._get_mime_type(file_path)
        display_name = (
            metadata.get("display_name", path.name) if metadata else path.name
        )

        # Build custom metadata if provided
        custom_metadata = None
        if metadata:
            custom_metadata = [
                types.CustomMetadata(key=k, string_value=str(v))
                for k, v in metadata.items()
                if k != "display_name"
            ]

        # Upload file to the store
        # _ensure_store guarantees _store_name is set
        assert self._store_name is not None
        operation = self._sync_client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=self._store_name,
            file=file_path,
            config=types.UploadToFileSearchStoreConfig(
                mime_type=mime_type,
                display_name=display_name,
                custom_metadata=custom_metadata if custom_metadata else None,
            ),
        )

        # Get the response from the operation
        result = operation.response
        file_id = result.document_name if result and result.document_name else path.name

        logger.info(f"Uploaded file to store: {file_path} -> {file_id}")

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

        This method uses Gemini's file_search tool internally to find
        relevant content from the uploaded files.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.

        Returns:
            List of retrieval results ordered by relevance.
        """
        await self._ensure_store()

        # Use the file_search tool via generate_content
        # The model will search the store and return relevant chunks
        # _ensure_store guarantees _store_name is set
        assert self._store_name is not None
        config = types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[self._store_name],
                        top_k=top_k,
                    )
                )
            ],
            system_instruction=(
                "You are a retrieval assistant. Search the knowledge base and return "
                "the most relevant passages for the user's query. Include citations."
            ),
        )

        response = await self._async_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Find relevant information for: {query}",
            config=config,
        )

        results = []
        if response.candidates:
            for candidate in response.candidates:
                if candidate.grounding_metadata:
                    # Extract grounding chunks with citations
                    chunks = candidate.grounding_metadata.grounding_chunks or []
                    for i, chunk in enumerate(chunks):
                        if chunk.web:
                            # Web grounding (not from our store)
                            continue

                        content = ""
                        citation = None
                        metadata: dict[str, Any] = {}

                        # Extract content from retrieved passages
                        if chunk.retrieved_context:
                            content = chunk.retrieved_context.text or ""
                            citation = chunk.retrieved_context.uri

                        results.append(
                            RetrievalResult(
                                content=content,
                                score=1.0 - (i * 0.1),  # Approximate score by position
                                citation=citation,
                                metadata=metadata,
                            )
                        )

                # Also check for inline citations in the response
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text and not results:
                            # If no grounding chunks, use the response text
                            results.append(
                                RetrievalResult(
                                    content=part.text,
                                    score=1.0,
                                    metadata={"source": "file_search"},
                                )
                            )

        return results[:top_k]

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the knowledge base.

        Args:
            document_id: ID of the document to delete.

        Returns:
            True if deleted, False if not found.
        """
        file_id = self._file_ids.get(document_id)
        if not file_id:
            return False

        # Note: Gemini File Search API may not support individual file deletion
        # from a store. This is a limitation of the current API.
        logger.warning(
            f"Document deletion not fully supported for Gemini File Search: {document_id}"
        )
        del self._file_ids[document_id]
        return True

    async def clear(self) -> None:
        """Clear all documents by deleting the store."""
        if self._store_name:
            try:
                self._sync_client.file_search_stores.delete(name=self._store_name)
                logger.info(f"Deleted file search store: {self._store_name}")
            except Exception as e:
                logger.warning(f"Failed to delete store: {e}")

        self._store = None
        self._store_name = None
        self._file_ids.clear()

    def get_file_search_tool(self, top_k: Optional[int] = None) -> types.Tool:
        """Get a Gemini Tool configured for file search.

        This can be passed directly to generate_content for native integration.

        Args:
            top_k: Number of results to retrieve. Uses default if not specified.

        Returns:
            A Gemini Tool configured for file search.
        """
        if not self._store_name:
            raise ValueError("No file search store configured. Add files first.")

        return types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=[self._store_name],
                top_k=top_k or self._top_k,
            )
        )
