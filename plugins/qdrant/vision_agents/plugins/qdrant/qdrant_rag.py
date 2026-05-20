"""
Qdrant Hybrid Search RAG implementation.

Uses Qdrant's built-in fastembed integration for dense and BM25 sparse embeddings.
Hybrid search uses Qdrant's native Reciprocal Rank Fusion (RRF).
See: https://qdrant.tech/documentation/concepts/hybrid-queries/

Usage:
    from vision_agents.plugins import qdrant

    # Initialize with a Qdrant collection
    rag = qdrant.QdrantRAG(collection="my-knowledge")
    await rag.add_directory("./knowledge")

    # Hybrid search (vector + BM25)
    results = await rag.search("How does the chat API work?")

    # Vector-only search
    results = await rag.search("How does the chat API work?", mode="vector")

    # BM25-only search
    results = await rag.search("chat API pricing", mode="bm25")

Environment variables:
    QDRANT_API_KEY: Qdrant API key (Optional)
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import AsyncQdrantClient
from qdrant_client import models

from vision_agents.core.rag import RAG, Document

logger = logging.getLogger(__name__)

_DENSE = "dense"
_SPARSE = "sparse"
_DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


class QdrantRAG(RAG):
    def __init__(
        self,
        collection: str,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        dense_model: str = _DEFAULT_DENSE_MODEL,
        sparse_model: str = _DEFAULT_SPARSE_MODEL,
        chunk_size: int = 10000,
        chunk_overlap: int = 200,
        cloud_inference: bool = False,
    ):
        self._collection = collection
        self._client = AsyncQdrantClient(
            url=url,
            api_key=api_key or os.environ.get("QDRANT_API_KEY"),
            cloud_inference=cloud_inference,
        )
        self._dense_model = dense_model
        self._sparse_model = sparse_model
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self._indexed_files: list[str] = []

    @property
    def indexed_files(self) -> list[str]:
        return self._indexed_files

    async def _ensure_collection(self) -> None:
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    _DENSE: models.VectorParams(
                        size=self._client.get_embedding_size(self._dense_model),
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    _SPARSE: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False),
                    ),
                },
            )

    async def add_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0

        all_chunks: list[str] = []
        chunk_sources: list[tuple[str, int]] = []

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

        await self._ensure_collection()
        await self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source}_{idx}")),
                    vector={
                        _DENSE: models.Document(text=chunk, model=self._dense_model),
                        _SPARSE: models.Document(text=chunk, model=self._sparse_model),
                    },
                    payload={"text": chunk, "source": source, "chunk_index": idx},
                )
                for chunk, (source, idx) in zip(all_chunks, chunk_sources)
            ],
        )

        logger.info(f"Indexed {len(all_chunks)} chunks from {len(documents)} documents")
        return len(all_chunks)

    async def _search_single(
        self, query: str, using: str, limit: int
    ) -> list[models.ScoredPoint]:
        model = self._dense_model if using == _DENSE else self._sparse_model
        return (
            await self._client.query_points(
                collection_name=self._collection,
                query=models.Document(text=query, model=model),
                using=using,
                limit=limit,
                with_payload=["text", "source"],
            )
        ).points

    async def search(
        self,
        query: str,
        top_k: int = 3,
        mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
    ) -> str:
        if not await self._client.collection_exists(self._collection):
            return "No relevant information found in the knowledge base."

        if mode == "vector":
            points = await self._search_single(query, _DENSE, top_k)
        elif mode == "bm25":
            points = await self._search_single(query, _SPARSE, top_k)
        else:
            results = await self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    models.Prefetch(
                        query=models.Document(text=query, model=self._dense_model),
                        using=_DENSE,
                        limit=top_k,
                    ),
                    models.Prefetch(
                        query=models.Document(text=query, model=self._sparse_model),
                        using=_SPARSE,
                        limit=top_k,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=["text", "source"],
            )
            points = results.points

        if not points:
            return "No relevant information found in the knowledge base."

        formatted_results = []
        for i, p in enumerate(points, 1):
            payload = p.payload or {}
            formatted_results.append(
                f"[{i}] From {payload.get('source', 'unknown')}:\n{payload.get('text', '')}"
            )
        return "\n\n".join(formatted_results)

    async def clear(self) -> None:
        if await self._client.collection_exists(self._collection):
            await self._client.delete_collection(self._collection)
        self._indexed_files = []
        logger.info(f"Cleared collection: {self._collection}")

    async def close(self) -> None:
        await self._client.close()


async def create_rag(
    collection: str,
    knowledge_dir: str | Path,
    extensions: list[str] | None = None,
    url: str = "http://localhost:6333",
    api_key: str | None = None,
    dense_model: str = _DEFAULT_DENSE_MODEL,
    sparse_model: str = _DEFAULT_SPARSE_MODEL,
    chunk_size: int = 10000,
    chunk_overlap: int = 200,
    cloud_inference: bool = False,
) -> QdrantRAG:
    rag = QdrantRAG(
        collection=collection,
        url=url,
        api_key=api_key,
        dense_model=dense_model,
        sparse_model=sparse_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        cloud_inference=cloud_inference,
    )
    await rag.add_directory(knowledge_dir, extensions=extensions)
    return rag
