"""Gemini File Search RAG implementation.

This module provides a RAG implementation using Gemini's File Search tool, which
uploads, indexes, and searches documents using Google's infrastructure.

See: https://ai.google.dev/gemini-api/docs/file-search
"""

import asyncio
import hashlib
import logging
import tempfile
from pathlib import Path

from google.genai import Client, types
from google.genai.types import (
    CreateFileSearchStoreConfig,
    GenerateContentConfig,
)

from vision_agents.core.rag import RAG, Document

logger = logging.getLogger(__name__)


def _compute_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


class GeminiFilesearchRAG(RAG):
    """
    RAG implementation using Gemini's File Search.

    File Search imports, chunks, and indexes your data to enable fast retrieval
    of relevant information. Search is performed by Gemini's infrastructure.

    The store automatically reuses existing stores with the same name and skips
    uploading documents that already exist (based on content hash stored in metadata).

    Usage:
        rag = GeminiFilesearchRAG(name="my-knowledge-base")
        await rag.create()  # Reuses existing store if found
        await rag.add_directory("./knowledge")

        # Search
        results = await rag.search("How does the API work?")

        # Or use with GeminiLLM directly
        llm = gemini.LLM(file_search_store=rag)
    """

    def __init__(
        self,
        name: str,
        client: Client | None = None,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
    ):
        """
        Initialize a GeminiFilesearchRAG.

        Args:
            name: Display name for the file search store.
            client: Optional Gemini client. Creates one if not provided.
            api_key: Optional API key. By default loads from GOOGLE_API_KEY.
            model: Model to use for search queries (default: gemini-2.0-flash).
        """
        self.name = name
        self._store_name: str | None = None
        self._uploaded_files: list[str] = []
        self._uploaded_hashes: set[str] = set()
        # Map of content_hash -> display_name for existing documents (loaded from API)
        self._existing_hashes: set[str] = set()
        self._model = model

        if client is not None:
            self._client = client
        else:
            self._client = Client(api_key=api_key)

    @property
    def store_name(self) -> str | None:
        """Get the full store resource name (e.g., 'fileSearchStores/abc123')."""
        return self._store_name

    @property
    def is_created(self) -> bool:
        """Check if the store has been created."""
        return self._store_name is not None

    @property
    def uploaded_hashes(self) -> set[str]:
        """Set of content hashes for uploaded documents."""
        return self._uploaded_hashes

    async def create(self) -> str:
        """
        Create or reuse an existing file search store.

        If a store with the same display_name already exists, it will be reused
        and existing documents will be loaded for deduplication.

        Returns:
            The store resource name.
        """
        if self._store_name:
            logger.info(
                f"GeminiFilesearchRAG '{self.name}' already created: {self._store_name}"
            )
            return self._store_name

        loop = asyncio.get_event_loop()

        # Check if a store with this name already exists
        existing_store = await loop.run_in_executor(None, self._find_existing_store)

        if existing_store:
            self._store_name = existing_store
            await self._load_existing_documents()
            logger.info(
                f"Reusing existing store '{self.name}': {self._store_name} "
                f"({len(self._existing_hashes)} documents with hashes)"
            )
            return self._store_name

        # Create new store if none exists
        store = await loop.run_in_executor(
            None,
            lambda: self._client.file_search_stores.create(
                config=CreateFileSearchStoreConfig(display_name=self.name)
            ),
        )
        self._store_name = store.name
        logger.info(f"Created new store '{self.name}': {self._store_name}")
        assert self._store_name is not None
        return self._store_name

    def _find_existing_store(self) -> str | None:
        """Find an existing store with the same display_name."""
        for store in self._client.file_search_stores.list():
            if store.display_name == self.name:
                return store.name
        return None

    async def _load_existing_documents(self) -> None:
        """Load existing document hashes from the store for deduplication."""
        if not self._store_name:
            return

        loop = asyncio.get_event_loop()
        store_name = self._store_name  # Capture for closure

        def list_docs():
            return list(
                self._client.file_search_stores.documents.list(parent=store_name)
            )

        docs = await loop.run_in_executor(None, list_docs)

        for doc in docs:
            self._uploaded_files.append(doc.display_name)
            # Extract content_hash from custom_metadata if present
            if doc.custom_metadata:
                for meta in doc.custom_metadata:
                    if meta.key == "content_hash" and meta.string_value:
                        self._existing_hashes.add(meta.string_value)
                        break

        logger.debug(
            f"Loaded {len(docs)} documents, {len(self._existing_hashes)} with hashes"
        )

    async def _upload_file(
        self,
        file_path: str | Path,
        display_name: str | None = None,
        content_hash: str | None = None,
    ) -> bool:
        """
        Upload a single file to the file search store.

        Skips upload if the content hash matches a previously uploaded file
        (checked against both in-memory session hashes and API-stored hashes).

        The content hash is stored in the document's custom_metadata for
        persistent deduplication across restarts.

        Args:
            file_path: Path to the file to upload.
            display_name: Optional display name (defaults to filename).
            content_hash: Optional hash of file content for deduplication.

        Returns:
            True if file was uploaded, False if skipped (duplicate).
        """
        if not self._store_name:
            raise ValueError("Store not created. Call create() first.")

        store_name = self._store_name

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        display_name = display_name or file_path.name

        # Compute hash if not provided
        if content_hash is None:
            content_hash = _compute_hash(file_path.read_text())

        # Check if hash already exists (from API or this session)
        if content_hash in self._existing_hashes:
            logger.info(f"Skipping (hash exists in store): {display_name}")
            return False

        if content_hash in self._uploaded_hashes:
            logger.info(f"Skipping (duplicate in session): {display_name}")
            return False

        loop = asyncio.get_event_loop()

        # Upload with content_hash in custom_metadata for persistent deduplication
        # Capture variables for lambda closure
        file_str = str(file_path)
        operation = await loop.run_in_executor(
            None,
            lambda: self._client.file_search_stores.upload_to_file_search_store(
                file=file_str,
                file_search_store_name=store_name,
                config=types.UploadToFileSearchStoreConfig(
                    display_name=display_name,
                    custom_metadata=[
                        types.CustomMetadata(
                            key="content_hash", string_value=content_hash
                        ),
                    ],
                ),
            ),
        )

        # Wait for the upload operation to complete
        while not operation.done:
            await asyncio.sleep(1)
            operation = await loop.run_in_executor(
                None, self._client.operations.get, operation
            )

        self._uploaded_files.append(display_name)
        self._uploaded_hashes.add(content_hash)
        self._existing_hashes.add(content_hash)
        logger.info(f"Uploaded and indexed: {display_name}")
        return True

    async def add_documents(self, documents: list[Document]) -> int:
        """
        Add documents to the RAG index.

        Documents are written to temporary files and uploaded to Gemini's
        File Search store. Duplicate documents (same content hash) are skipped.

        Args:
            documents: List of documents to index.

        Returns:
            Number of documents indexed (excluding duplicates).
        """
        if not self._store_name:
            raise ValueError("Store not created. Call create() first.")

        if not documents:
            return 0

        uploaded_count = 0

        # Write documents to temp files and upload
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            for doc in documents:
                # Compute hash for deduplication
                content_hash = _compute_hash(doc.text)

                # Use source as filename, default to .txt extension
                filename = doc.source
                if not Path(filename).suffix:
                    filename = f"{filename}.txt"
                filepath = tmppath / filename
                filepath.write_text(doc.text)

                if await self._upload_file(
                    filepath, display_name=doc.source, content_hash=content_hash
                ):
                    uploaded_count += 1

        logger.info(
            f"Indexed {uploaded_count} documents ({len(documents) - uploaded_count} duplicates skipped)"
        )
        return uploaded_count

    async def add_directory(
        self,
        path: str | Path,
        extensions: list[str] | None = None,
        batch_size: int = 5,
    ) -> int:
        """
        Add all files from a directory to the RAG index.

        Args:
            path: Path to directory containing files.
            extensions: File extensions to include. Defaults to common doc types.
            batch_size: Number of files to upload concurrently (default 5).

        Returns:
            Number of files indexed.
        """
        if not self._store_name:
            raise ValueError("Store not created. Call create() first.")

        directory = Path(path)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Default extensions for common document types
        if extensions is None:
            extensions = [".md", ".txt", ".pdf", ".json", ".html", ".csv"]

        # Normalize extensions
        extensions = [
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in extensions
        ]

        files = [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]

        if not files:
            logger.warning(
                f"No files found in {directory} with extensions {extensions}"
            )
            return 0

        logger.info(
            f"Uploading {len(files)} files from {directory} (batch_size={batch_size})"
        )

        # Upload files in batches concurrently
        uploaded_count = 0
        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            results = await asyncio.gather(*[self._upload_file(f) for f in batch])
            uploaded_count += sum(results)

        logger.info(
            f"Indexed {uploaded_count} files ({len(files) - uploaded_count} duplicates skipped)"
        )
        return uploaded_count

    async def search(self, query: str, top_k: int = 3) -> str:
        """
        Search the knowledge base using Gemini's File Search.

        Args:
            query: Search query.
            top_k: Number of results to return (hint, Gemini controls actual count).

        Returns:
            Formatted string with search results from Gemini.
        """
        if not self._store_name:
            raise ValueError("Store not created. Call create() first.")

        loop = asyncio.get_event_loop()

        # Use Gemini to search with the file search tool
        response = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=f"Search the knowledge base and return relevant information for: {query}\n\nReturn up to {top_k} relevant excerpts with their sources.",
                config=GenerateContentConfig(
                    tools=[self.get_tool()],
                ),
            ),
        )

        if response.text:
            return response.text
        return "No relevant information found in the knowledge base."

    async def clear(self) -> None:
        """Delete the file search store and all its contents."""
        if not self._store_name:
            return

        loop = asyncio.get_event_loop()
        store_name = self._store_name

        # Delete the store with force=True to also delete all documents
        await loop.run_in_executor(
            None,
            lambda: self._client.file_search_stores.delete(
                name=store_name, config={"force": True}
            ),
        )
        logger.info(f"Deleted GeminiFilesearchRAG: {store_name}")
        self._store_name = None
        self._uploaded_files = []
        self._uploaded_hashes = set()
        self._existing_hashes = set()

    async def close(self) -> None:
        """Close resources. Note: does not delete the store."""
        pass

    def get_tool(self) -> types.Tool:
        """
        Get the File Search tool configuration for use with Gemini LLM.

        Returns:
            Tool object configured with this file search store.
        """
        if not self._store_name:
            raise ValueError("Store not created. Call create() first.")

        return types.Tool(
            file_search=types.FileSearch(file_search_store_names=[self._store_name])
        )

    def get_tool_config(self) -> dict:
        """
        Get the file search tool as a dict for use in GenerateContentConfig.

        Returns:
            Dict representation of the file search tool.
        """
        if not self._store_name:
            raise ValueError("Store not created. Call create() first.")

        return {"file_search": {"file_search_store_names": [self._store_name]}}


# Keep old name as alias for backwards compatibility
FileSearchStore = GeminiFilesearchRAG


async def create_file_search_store(
    name: str,
    knowledge_dir: str | Path,
    client: Client | None = None,
    api_key: str | None = None,
    extensions: list[str] | None = None,
    batch_size: int = 5,
) -> GeminiFilesearchRAG:
    """
    Convenience function to create a GeminiFilesearchRAG and upload files.

    Args:
        name: Display name for the store.
        knowledge_dir: Directory containing knowledge files to upload.
        client: Optional Gemini client.
        api_key: Optional API key.
        extensions: Optional file extensions to include.
        batch_size: Number of files to upload concurrently (default 5).

    Returns:
        Configured GeminiFilesearchRAG with files uploaded.

    Example:
        rag = await create_file_search_store(
            name="product-knowledge",
            knowledge_dir="./knowledge"
        )
        results = await rag.search("How does the API work?")

        # Or use with GeminiLLM directly
        llm = gemini.LLM(file_search_store=rag)
    """
    rag = GeminiFilesearchRAG(name=name, client=client, api_key=api_key)
    await rag.create()
    await rag.add_directory(knowledge_dir, extensions=extensions, batch_size=batch_size)
    return rag
