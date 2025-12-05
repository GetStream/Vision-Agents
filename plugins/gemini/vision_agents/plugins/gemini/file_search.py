"""Gemini File Search (RAG) utilities.

This module provides functionality for Gemini's File Search tool, which enables
Retrieval Augmented Generation (RAG) by uploading, indexing, and searching documents.

See: https://ai.google.dev/gemini-api/docs/file-search
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from google.genai import Client
from google.genai.types import CreateFileSearchStoreConfig, Tool, FileSearch

logger = logging.getLogger(__name__)


class FileSearchStore:
    """
    Manages a Gemini File Search Store for RAG functionality.
    
    File Search imports, chunks, and indexes your data to enable fast retrieval
    of relevant information based on prompts.
    
    Usage:
        store = FileSearchStore(name="my-knowledge-base")
        await store.create()
        await store.upload_directory("./knowledge")
        
        # Use with GeminiLLM
        llm = gemini.LLM(file_search_store=store)
    """
    
    def __init__(
        self,
        name: str,
        client: Optional[Client] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize a FileSearchStore.
        
        Args:
            name: Display name for the file search store.
            client: Optional Gemini client. Creates one if not provided.
            api_key: Optional API key. By default loads from GOOGLE_API_KEY.
        """
        self.name = name
        self._store_name: Optional[str] = None
        self._uploaded_files: list[str] = []
        
        if client is not None:
            self._client = client
        else:
            self._client = Client(api_key=api_key)
    
    @property
    def store_name(self) -> Optional[str]:
        """Get the full store resource name (e.g., 'fileSearchStores/abc123')."""
        return self._store_name
    
    @property
    def is_created(self) -> bool:
        """Check if the store has been created."""
        return self._store_name is not None
    
    async def create(self) -> str:
        """
        Create the file search store.
        
        Returns:
            The store resource name.
        """
        if self._store_name:
            logger.info(f"FileSearchStore '{self.name}' already created: {self._store_name}")
            return self._store_name
        
        loop = asyncio.get_event_loop()
        store = await loop.run_in_executor(
            None,
            lambda: self._client.file_search_stores.create(
                config=CreateFileSearchStoreConfig(display_name=self.name)
            )
        )
        self._store_name = store.name
        logger.info(f"Created FileSearchStore '{self.name}': {self._store_name}")
        assert self._store_name is not None
        return self._store_name
    
    async def upload_file(self, file_path: str | Path, display_name: Optional[str] = None) -> None:
        """
        Upload a single file to the file search store.
        
        Args:
            file_path: Path to the file to upload.
            display_name: Optional display name (defaults to filename).
        """
        if not self._store_name:
            raise ValueError("Store not created. Call create() first.")
        
        store_name = self._store_name
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        display_name = display_name or file_path.name
        
        loop = asyncio.get_event_loop()
        
        # Upload and wait for indexing
        operation = await loop.run_in_executor(
            None,
            lambda: self._client.file_search_stores.upload_to_file_search_store(
                file=str(file_path),
                file_search_store_name=store_name,
                config={"display_name": display_name}
            )
        )
        
        # Wait for the upload operation to complete
        while not operation.done:
            await asyncio.sleep(1)
            operation = await loop.run_in_executor(
                None,
                lambda: self._client.operations.get(operation)
            )
        
        self._uploaded_files.append(display_name)
        logger.info(f"Uploaded and indexed: {display_name}")
    
    async def upload_directory(
        self,
        directory: str | Path,
        extensions: Optional[list[str]] = None,
        batch_size: int = 5,
    ) -> int:
        """
        Upload all files from a directory to the file search store.
        
        Args:
            directory: Path to directory containing files.
            extensions: Optional list of file extensions to include (e.g., ['.md', '.txt']).
                       If None, uploads all supported file types.
            batch_size: Number of files to upload concurrently (default 5).
        
        Returns:
            Number of files uploaded.
        """
        if not self._store_name:
            raise ValueError("Store not created. Call create() first.")
        
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        
        # Default extensions for common document types
        if extensions is None:
            extensions = ['.md', '.txt', '.pdf', '.json', '.html', '.csv']
        
        # Normalize extensions
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]
        
        files = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in extensions]
        
        if not files:
            logger.warning(f"No files found in {directory} with extensions {extensions}")
            return 0
        
        logger.info(f"Uploading {len(files)} files from {directory} (batch_size={batch_size})")
        
        # Upload files in batches concurrently
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            await asyncio.gather(*[self.upload_file(f) for f in batch])
        
        return len(files)
    
    def get_tool(self) -> Tool:
        """
        Get the File Search tool configuration for use with Gemini.
        
        Returns:
            Tool object configured with this file search store.
        """
        if not self._store_name:
            raise ValueError("Store not created. Call create() first.")
        
        return Tool(
            file_search=FileSearch(
                file_search_store_names=[self._store_name]
            )
        )
    
    def get_tool_config(self) -> dict:
        """
        Get the file search tool as a dict for use in GenerateContentConfig.
        
        Returns:
            Dict representation of the file search tool.
        """
        if not self._store_name:
            raise ValueError("Store not created. Call create() first.")
        
        return {
            "file_search": {
                "file_search_store_names": [self._store_name]
            }
        }


async def create_file_search_store(
    name: str,
    knowledge_dir: str | Path,
    client: Optional[Client] = None,
    api_key: Optional[str] = None,
    extensions: Optional[list[str]] = None,
    batch_size: int = 5,
) -> FileSearchStore:
    """
    Convenience function to create a file search store and upload files.
    
    Args:
        name: Display name for the store.
        knowledge_dir: Directory containing knowledge files to upload.
        client: Optional Gemini client.
        api_key: Optional API key.
        extensions: Optional file extensions to include.
        batch_size: Number of files to upload concurrently (default 5).
    
    Returns:
        Configured FileSearchStore with files uploaded.
    
    Example:
        store = await create_file_search_store(
            name="product-knowledge",
            knowledge_dir="./knowledge"
        )
        llm = gemini.LLM(file_search_store=store)
    """
    store = FileSearchStore(name=name, client=client, api_key=api_key)
    await store.create()
    await store.upload_directory(knowledge_dir, extensions=extensions, batch_size=batch_size)
    return store

