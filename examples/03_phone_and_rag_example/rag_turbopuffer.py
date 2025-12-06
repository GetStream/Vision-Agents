"""
TurboPuffer Hybrid Search RAG implementation.

This module provides a hybrid search RAG (Retrieval Augmented Generation) implementation
using TurboPuffer for vector + BM25 full-text search, with Gemini for embeddings.

Hybrid search combines:
- Vector search: Semantic similarity using embeddings
- BM25 full-text search: Keyword matching for exact terms (SKUs, names, etc.)

Results are combined using Reciprocal Rank Fusion (RRF) for better retrieval quality.
See: https://turbopuffer.com/docs/hybrid

Usage:
    from rag_turbopuffer import TurboPufferRAG

    # Initialize with knowledge directory
    rag = TurboPufferRAG(namespace="my-knowledge")
    await rag.index_directory("./knowledge")
    
    # Hybrid search (vector + BM25)
    results = await rag.search("How does the chat API work?")
    
    # Vector-only search
    results = await rag.search("How does the chat API work?", mode="vector")
    
    # BM25-only search
    results = await rag.search("chat API pricing", mode="bm25")

Environment variables:
    TURBO_PUFFER_KEY: TurboPuffer API key
    GOOGLE_API_KEY: Google API key (for Gemini embeddings)

Note:
    For embedding model selection best practices and benchmarks, see:
    https://huggingface.co/spaces/mteb/leaderboard
"""

import asyncio
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

from turbopuffer import AsyncTurbopuffer
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Schema for hybrid search - enables BM25 full-text search on the text field
HYBRID_SCHEMA = {
    "text": {
        "type": "string",
        "full_text_search": True,
    },
    "source": {"type": "string"},
    "chunk_index": {"type": "uint"},
}


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).
    
    RRF is a simple but effective rank fusion algorithm that combines
    results from multiple search strategies.
    
    Args:
        ranked_lists: List of ranked results, each as [(id, score), ...].
        k: RRF constant (default 60, as per original paper).
        
    Returns:
        Fused ranking as [(id, rrf_score), ...] sorted by score descending.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
    
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


class TurboPufferRAG:
    """
    Hybrid search RAG using TurboPuffer (vector + BM25) and Gemini embeddings.
    
    Combines semantic vector search with BM25 keyword search for better
    retrieval quality. Uses Reciprocal Rank Fusion to merge results.
    
    For hybrid search best practices, see:
    https://turbopuffer.com/docs/hybrid
    
    For embedding model benchmarks, see the MTEB leaderboard:
    https://huggingface.co/spaces/mteb/leaderboard
    """

    def __init__(
        self,
        namespace: str,
        embedding_model: str = "models/gemini-embedding-001",
        chunk_size: int = 10000,
        chunk_overlap: int = 200,
        region: str = "gcp-us-central1",
    ):
        """
        Initialize the TurboPuffer Hybrid RAG.
        
        Args:
            namespace: TurboPuffer namespace for storing vectors.
            embedding_model: Gemini embedding model (default: gemini-embedding-001).
            chunk_size: Size of text chunks for splitting documents.
            chunk_overlap: Overlap between chunks for context continuity.
            region: TurboPuffer region (default "gcp-us-central1").
        """
        self._namespace_name = namespace
        
        # Initialize async TurboPuffer client
        self._client = AsyncTurbopuffer(
            api_key=os.environ.get("TURBO_PUFFER_KEY"),
            region=region,
        )
        
        # Initialize Gemini embeddings
        self._embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        
        # Initialize text splitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        self._indexed_files: list[str] = []
        # Cache for retrieved documents (id -> attributes)
        self._doc_cache: dict[str, dict] = {}

    @property
    def indexed_files(self) -> list[str]:
        """List of indexed file names."""
        return self._indexed_files

    async def index_file(self, file_path: str | Path, source_name: Optional[str] = None) -> int:
        """
        Index a single file into the vector database with hybrid search support.
        
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
        
        # Prepare rows for TurboPuffer
        rows = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            rows.append({
                "id": f"{source_name}_{i}",
                "vector": embedding,
                "text": chunk,
                "source": source_name,
                "chunk_index": i,
            })
        
        # Upsert with schema enabling full-text search on text field
        ns = self._client.namespace(self._namespace_name)
        await ns.write(
            upsert_rows=rows,
            distance_metric="cosine_distance",
            schema=HYBRID_SCHEMA,
        )
        
        self._indexed_files.append(source_name)
        logger.info(f"Indexed {len(chunks)} chunks from {source_name}")
        
        return len(chunks)

    async def warm_cache(self) -> None:
        """
        Hint TurboPuffer to prepare for low-latency requests.
        
        Call this after indexing to ensure fast query responses.
        See: https://turbopuffer.com/docs/warm-cache
        """
        ns = self._client.namespace(self._namespace_name)
        await ns.hint_cache_warm()
        logger.info(f"Cache warmed for namespace: {self._namespace_name}")

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
        
        # Warm cache for low-latency queries
        await self.warm_cache()
        
        return total_chunks

    async def _vector_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Run vector similarity search."""
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None, self._embeddings.embed_query, query
        )
        
        ns = self._client.namespace(self._namespace_name)
        results = await ns.query(
            rank_by=("vector", "ANN", query_embedding),
            top_k=top_k,
            include_attributes=["text", "source"],
        )
        
        ranked = []
        for row in results.rows:
            doc_id = str(row.id)
            # Cache the document for later retrieval
            self._doc_cache[doc_id] = {
                "text": row.text if hasattr(row, "text") else "",
                "source": row.source if hasattr(row, "source") else "unknown",
            }
            # Lower distance = better, so we use negative for ranking
            dist = getattr(row, "$dist", 0)
            ranked.append((doc_id, -dist))
        
        return ranked

    async def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Run BM25 full-text search."""
        ns = self._client.namespace(self._namespace_name)
        results = await ns.query(
            rank_by=("text", "BM25", query),
            top_k=top_k,
            include_attributes=["text", "source"],
        )
        
        ranked = []
        for row in results.rows:
            doc_id = str(row.id)
            # Cache the document for later retrieval
            self._doc_cache[doc_id] = {
                "text": row.text if hasattr(row, "text") else "",
                "source": row.source if hasattr(row, "source") else "unknown",
            }
            # BM25 score (higher = better)
            score = getattr(row, "$dist", 0)
            ranked.append((doc_id, score))
        
        return ranked

    async def search(
        self,
        query: str,
        top_k: int = 3,
        mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
    ) -> str:
        """
        Search the knowledge base using hybrid, vector, or BM25 search.
        
        Hybrid search combines vector (semantic) and BM25 (keyword) search
        using Reciprocal Rank Fusion for better retrieval quality.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            mode: Search mode - "hybrid" (default), "vector", or "bm25".
            
        Returns:
            Formatted string with search results.
        """
        # Clear doc cache for fresh search
        self._doc_cache.clear()
        
        # Fetch more candidates for fusion, then trim to top_k
        fetch_k = top_k * 3
        
        if mode == "vector":
            ranked = await self._vector_search(query, fetch_k)
            final_ids = [doc_id for doc_id, _ in ranked[:top_k]]
        elif mode == "bm25":
            ranked = await self._bm25_search(query, fetch_k)
            final_ids = [doc_id for doc_id, _ in ranked[:top_k]]
        else:
            # Hybrid: run both searches in parallel and fuse
            vector_results, bm25_results = await asyncio.gather(
                self._vector_search(query, fetch_k),
                self._bm25_search(query, fetch_k),
            )
            
            # Combine using Reciprocal Rank Fusion
            fused = reciprocal_rank_fusion([vector_results, bm25_results])
            final_ids = [doc_id for doc_id, _ in fused[:top_k]]
        
        if not final_ids:
            return "No relevant information found in the knowledge base."
        
        # Format results from cache
        formatted_results = []
        for i, doc_id in enumerate(final_ids, 1):
            doc = self._doc_cache.get(doc_id, {})
            source = doc.get("source", "unknown")
            text = doc.get("text", "")
            formatted_results.append(f"[{i}] From {source}:\n{text}")
        
        return "\n\n".join(formatted_results)

    async def clear(self) -> None:
        """Clear all vectors from the namespace."""
        ns = self._client.namespace(self._namespace_name)
        await ns.delete_all()
        self._indexed_files = []
        self._doc_cache.clear()
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
    Convenience function to create and initialize a TurboPuffer Hybrid RAG.
    
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
            return await rag.search(query)  # Uses hybrid search by default
    """
    rag = TurboPufferRAG(namespace=namespace, region=region)
    await rag.index_directory(knowledge_dir, extensions=extensions)
    return rag
