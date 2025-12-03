"""
TurboPuffer + LangChain RAG implementation.

This module provides a vector-based RAG (Retrieval Augmented Generation) implementation
using TurboPuffer for vector storage and LangChain for embeddings.

Usage:
    from rag_turbopuffer import TurboPufferRAG

    # Initialize with knowledge directory
    rag = TurboPufferRAG(namespace="my-knowledge")
    await rag.index_directory("./knowledge")
    
    # Search for relevant documents
    results = await rag.search("How does the chat API work?")
    
    # Register as LLM function
    @llm.register_function(description="Search knowledge base for product information")
    async def search_knowledge(query: str) -> str:
        return await rag.search(query)

Environment variables:
    TURBO_PUFFER_KEY: TurboPuffer API key
    OPENAI_API_KEY: OpenAI API key (for embeddings)
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from turbopuffer import AsyncTurbopuffer
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TurboPufferRAG:
    """
    RAG implementation using TurboPuffer vector database and LangChain embeddings.
    
    TurboPuffer is a fast, serverless vector database optimized for RAG workloads.
    Combined with OpenAI embeddings via LangChain, it provides high-quality
    semantic search over your knowledge base.
    """

    def __init__(
        self,
        namespace: str,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        region: str = "gcp-us-central1",
    ):
        """
        Initialize the TurboPuffer RAG.
        
        Args:
            namespace: TurboPuffer namespace for storing vectors.
            embedding_model: OpenAI embedding model to use.
            chunk_size: Size of text chunks for splitting documents.
            chunk_overlap: Overlap between chunks for context continuity.
            region: TurboPuffer region (default "gcp-us-central1").
        """
        self._namespace_name = namespace
        
        # Initialize async TurboPuffer client (v0.5+ API)
        self._client = AsyncTurbopuffer(
            api_key=os.environ.get("TURBO_PUFFER_KEY"),
            region=region,
        )
        
        # Initialize embeddings
        self._embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize text splitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        self._indexed_files: list[str] = []

    async def index_file(self, file_path: str | Path, source_name: Optional[str] = None) -> int:
        """
        Index a single file into the vector database.
        
        Args:
            file_path: Path to the file to index.
            source_name: Optional name for the source (defaults to filename).
            
        Returns:
            Number of chunks indexed.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        source_name = source_name or file_path.name
        content = file_path.read_text()
        
        # Split into chunks
        chunks = self._splitter.split_text(content)
        if not chunks:
            logger.warning(f"No chunks generated from {file_path}")
            return 0
        
        # Generate embeddings (run in executor since it's sync)
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self._embeddings.embed_documents, chunks
        )
        
        # Prepare rows for TurboPuffer (v0.5+ flattened attribute format)
        rows = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            rows.append({
                "id": f"{source_name}_{i}",
                "vector": embedding,
                "text": chunk,
                "source": source_name,
                "chunk_index": i,
            })
        
        # Upsert to TurboPuffer using namespace.write()
        ns = self._client.namespace(self._namespace_name)
        await ns.write(
            upsert_rows=rows,
            distance_metric="cosine_distance",
        )
        
        self._indexed_files.append(source_name)
        logger.info(f"Indexed {len(chunks)} chunks from {source_name}")
        
        return len(chunks)

    async def index_directory(
        self,
        directory: str | Path,
        extensions: Optional[list[str]] = None,
    ) -> int:
        """
        Index all files from a directory.
        
        Args:
            directory: Path to directory containing files.
            extensions: File extensions to include (e.g., ['.md', '.txt']).
            
        Returns:
            Total number of chunks indexed.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        
        if extensions is None:
            extensions = [".md", ".txt"]
        
        # Normalize extensions
        extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions]
        
        files = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in extensions]
        
        if not files:
            logger.warning(f"No files found in {directory} with extensions {extensions}")
            return 0
        
        logger.info(f"Indexing {len(files)} files from {directory}")
        
        total_chunks = 0
        for file_path in files:
            chunks = await self.index_file(file_path)
            total_chunks += chunks
        
        return total_chunks

    async def search(self, query: str, top_k: int = 3) -> str:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            
        Returns:
            Formatted string with search results.
        """
        # Generate query embedding (run in executor since it's sync)
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None, self._embeddings.embed_query, query
        )
        
        # Search TurboPuffer using async client
        ns = self._client.namespace(self._namespace_name)
        results = await ns.query(
            rank_by=("vector", "ANN", query_embedding),
            top_k=top_k,
            include_attributes=["text", "source"],
        )
        
        if not results.rows:
            return "No relevant information found in the knowledge base."
        
        # Format results (v0.5+ - rows have attributes as properties)
        formatted_results = []
        for i, row in enumerate(results.rows, 1):
            source = row.source if hasattr(row, "source") else "unknown"
            text = row.text if hasattr(row, "text") else ""
            formatted_results.append(f"[{i}] From {source}:\n{text}")
        
        return "\n\n".join(formatted_results)

    async def clear(self) -> None:
        """Clear all vectors from the namespace."""
        ns = self._client.namespace(self._namespace_name)
        await ns.delete_all()
        self._indexed_files = []
        logger.info(f"Cleared namespace: {self._namespace_name}")

    async def close(self) -> None:
        """Close the TurboPuffer client."""
        await self._client.close()


async def create_rag(
    namespace: str,
    knowledge_dir: str | Path,
    extensions: Optional[list[str]] = None,
    region: str = "gcp-us-central1",
) -> TurboPufferRAG:
    """
    Convenience function to create and initialize a TurboPuffer RAG.
    
    Args:
        namespace: TurboPuffer namespace name.
        knowledge_dir: Directory containing knowledge files.
        extensions: File extensions to include.
        region: TurboPuffer region.
        
    Returns:
        Initialized TurboPufferRAG with files indexed.
        
    Example:
        rag = await create_rag(
            namespace="product-knowledge",
            knowledge_dir="./knowledge"
        )
        
        @llm.register_function(description="Search knowledge base")
        async def search_knowledge(query: str) -> str:
            return await rag.search(query)
    """
    rag = TurboPufferRAG(namespace=namespace, region=region)
    await rag.index_directory(knowledge_dir, extensions=extensions)
    return rag
