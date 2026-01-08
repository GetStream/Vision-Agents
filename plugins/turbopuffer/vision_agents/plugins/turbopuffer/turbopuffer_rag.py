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
    from vision_agents.plugins import turbopuffer

    # Initialize with knowledge directory
    rag = turbopuffer.TurboPufferRAG(namespace="my-knowledge")
    await rag.add_directory("./knowledge")

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
from typing import Literal

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from turbopuffer import AsyncTurbopuffer, NotFoundError

from vision_agents.core.rag import RAG, Document

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


class TurboPufferRAG(RAG):
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

    async def add_documents(self, documents: list[Document]) -> int:
        """
        Add documents to the RAG index.

        Args:
            documents: List of documents to index.

        Returns:
            Number of chunks indexed.
        """
        if not documents:
            return 0

        all_chunks: list[str] = []
        chunk_sources: list[tuple[str, int]] = []  # (source, chunk_index)

        for doc in documents:
            chunks = self._splitter.split_text(doc.text)
            if not chunks:
                logger.warning(f"No chunks generated from document: {doc.source}")
                continue
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_sources.append((doc.source, i))
            self._indexed_files.append(doc.source)

        if not all_chunks:
            return 0

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self._embeddings.embed_documents, all_chunks
        )

        rows = []
        for chunk, embedding, (source, idx) in zip(
            all_chunks, embeddings, chunk_sources
        ):
            rows.append(
                {
                    "id": f"{source}_{idx}",
                    "vector": embedding,
                    "text": chunk,
                    "source": source,
                    "chunk_index": idx,
                }
            )

        ns = self._client.namespace(self._namespace_name)
        await ns.write(
            upsert_rows=rows,
            distance_metric="cosine_distance",
            schema=HYBRID_SCHEMA,  # type: ignore[arg-type]
        )

        logger.info(f"Indexed {len(all_chunks)} chunks from {len(documents)} documents")
        return len(all_chunks)

    async def add_directory(
        self,
        path: str | Path,
        extensions: list[str] | None = None,
    ) -> int:
        """
        Add all files from a directory to the RAG index.

        Args:
            path: Path to directory containing files.
            extensions: File extensions to include (e.g., ['.md', '.txt']).

        Returns:
            Total number of chunks indexed.
        """
        total_chunks = await super().add_directory(path, extensions)

        # Warm cache for low-latency queries
        if total_chunks > 0:
            await self.warm_cache()

        return total_chunks

    async def warm_cache(self) -> None:
        """
        Hint TurboPuffer to prepare for low-latency requests.

        Call this after indexing to ensure fast query responses.
        See: https://turbopuffer.com/docs/warm-cache
        """
        ns = self._client.namespace(self._namespace_name)
        await ns.hint_cache_warm()
        logger.info(f"Cache warmed for namespace: {self._namespace_name}")

    async def _vector_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Run vector similarity search."""
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None, self._embeddings.embed_query, query
        )

        ns = self._client.namespace(self._namespace_name)
        try:
            results = await ns.query(
                rank_by=("vector", "ANN", query_embedding),
                top_k=top_k,
                include_attributes=["text", "source"],
            )
        except NotFoundError:
            return []

        ranked = []
        for row in results.rows or []:
            doc_id = str(row.id)
            # Cache the document for later retrieval
            self._doc_cache[doc_id] = {
                "text": row["text"] or "",
                "source": row["source"] or "unknown",
            }
            # Lower distance = better, so we use negative for ranking
            dist = row["$dist"] or 0
            ranked.append((doc_id, -dist))

        return ranked

    async def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Run BM25 full-text search."""
        ns = self._client.namespace(self._namespace_name)
        try:
            results = await ns.query(
                rank_by=("text", "BM25", query),
                top_k=top_k,
                include_attributes=["text", "source"],
            )
        except NotFoundError:
            return []

        ranked = []
        for row in results.rows or []:
            doc_id = str(row.id)
            # Cache the document for later retrieval
            self._doc_cache[doc_id] = {
                "text": row["text"] or "",
                "source": row["source"] or "unknown",
            }
            # BM25 score (higher = better)
            score = row["$dist"] or 0
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
        try:
            await ns.delete_all()
        except NotFoundError:
            pass  # Namespace doesn't exist, nothing to clear
        self._indexed_files = []
        self._doc_cache.clear()
        logger.info(f"Cleared namespace: {self._namespace_name}")

    async def close(self) -> None:
        """Close the TurboPuffer client."""
        await self._client.close()


async def create_rag(
    namespace: str,
    knowledge_dir: str | Path,
    extensions: list[str] | None = None,
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
    await rag.add_directory(knowledge_dir, extensions=extensions)
    return rag
