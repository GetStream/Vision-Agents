"""Tests for the local RAG module."""

import pytest

from vision_agents.core.rag import (
    Document,
    FixedSizeChunker,
    InMemoryVectorStore,
    LocalRAG,
    SentenceChunker,
)
from vision_agents.core.rag.local.embeddings import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 3):
        self._dimension = dimension
        self._call_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Generate a simple hash-based embedding."""
        self._call_count += 1
        # Create a simple deterministic embedding based on text hash
        h = hash(text) % 1000
        return [h / 1000, (h * 2) % 1000 / 1000, (h * 3) % 1000 / 1000]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_basic_chunking(self):
        """Test basic text chunking."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=0)
        text = "Hello world, how are you?"

        chunks = chunker.chunk(text)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello worl"
        assert chunks[1].content == "d, how are"
        assert chunks[2].content == " you?"

    def test_chunking_with_overlap(self):
        """Test chunking with overlap."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=3)
        text = "Hello world, how are you?"

        chunks = chunker.chunk(text)

        # With overlap, chunks should share some content
        assert chunks[0].content[-3:] == chunks[1].content[:3]

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=0)

        chunks = chunker.chunk("")

        assert len(chunks) == 0

    def test_text_smaller_than_chunk_size(self):
        """Test text smaller than chunk size."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        text = "Short text"

        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == "Short text"

    def test_chunk_positions(self):
        """Test chunk start/end positions."""
        chunker = FixedSizeChunker(chunk_size=5, overlap=0)
        text = "Hello world"

        chunks = chunker.chunk(text)

        assert chunks[0].start == 0
        assert chunks[0].end == 5
        assert chunks[1].start == 5
        assert chunks[1].end == 10

    def test_invalid_overlap(self):
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, overlap=10)

        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, overlap=15)


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_basic_sentence_chunking(self):
        """Test basic sentence chunking."""
        chunker = SentenceChunker(max_chunk_size=50, min_chunk_size=10)
        text = "First sentence. Second sentence. Third sentence."

        chunks = chunker.chunk(text)

        # Should group sentences together up to max_chunk_size
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.content) <= 50

    def test_long_sentence(self):
        """Test handling of sentence longer than max_chunk_size."""
        chunker = SentenceChunker(max_chunk_size=20, min_chunk_size=5)
        text = "This is a very long sentence that exceeds the maximum chunk size."

        chunks = chunker.chunk(text)

        # Should still create chunks even if sentence is too long
        assert len(chunks) >= 1

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = SentenceChunker(max_chunk_size=100)

        chunks = chunker.chunk("")

        assert len(chunks) == 0

    def test_single_sentence(self):
        """Test single sentence."""
        chunker = SentenceChunker(max_chunk_size=100)
        text = "Just one sentence."

        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == "Just one sentence."


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore."""

    async def test_add_and_search(self):
        """Test adding vectors and searching."""
        store = InMemoryVectorStore()

        await store.add("id1", [1.0, 0.0, 0.0], "Content 1", {"key": "value1"})
        await store.add("id2", [0.0, 1.0, 0.0], "Content 2", {"key": "value2"})
        await store.add("id3", [0.9, 0.1, 0.0], "Content 3", {"key": "value3"})

        # Search with a vector similar to id1 and id3
        results = await store.search([1.0, 0.0, 0.0], top_k=2)

        assert len(results) == 2
        # id1 should be the best match (exact match)
        assert results[0][0].id == "id1"
        assert results[0][1] == pytest.approx(1.0)
        # id3 should be second (similar)
        assert results[1][0].id == "id3"

    async def test_search_empty_store(self):
        """Test searching an empty store."""
        store = InMemoryVectorStore()

        results = await store.search([1.0, 0.0, 0.0], top_k=5)

        assert len(results) == 0

    async def test_delete(self):
        """Test deleting entries."""
        store = InMemoryVectorStore()

        await store.add("id1", [1.0, 0.0, 0.0], "Content 1")

        assert await store.count() == 1

        deleted = await store.delete("id1")
        assert deleted is True
        assert await store.count() == 0

        # Try to delete again
        deleted_again = await store.delete("id1")
        assert deleted_again is False

    async def test_clear(self):
        """Test clearing the store."""
        store = InMemoryVectorStore()

        await store.add("id1", [1.0, 0.0, 0.0], "Content 1")
        await store.add("id2", [0.0, 1.0, 0.0], "Content 2")

        assert await store.count() == 2

        await store.clear()

        assert await store.count() == 0

    async def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Identical vectors should have similarity 1.0
        sim = InMemoryVectorStore._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert sim == pytest.approx(1.0)

        # Orthogonal vectors should have similarity 0.0
        sim = InMemoryVectorStore._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert sim == pytest.approx(0.0)

        # Opposite vectors should have similarity -1.0
        sim = InMemoryVectorStore._cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert sim == pytest.approx(-1.0)


class TestLocalRAG:
    """Tests for LocalRAG."""

    async def test_add_documents(self):
        """Test adding documents."""
        embeddings = MockEmbeddingProvider()
        store = InMemoryVectorStore()
        rag = LocalRAG(embeddings=embeddings, vector_store=store)

        docs = [
            Document(content="First document content", id="doc1"),
            Document(content="Second document content", id="doc2"),
        ]

        await rag.add_documents(docs)

        # Verify chunks were added to the store
        count = await store.count()
        assert count > 0

    async def test_search(self):
        """Test searching for documents."""
        embeddings = MockEmbeddingProvider()
        store = InMemoryVectorStore()
        rag = LocalRAG(
            embeddings=embeddings,
            vector_store=store,
            chunker=FixedSizeChunker(chunk_size=100, overlap=0),
        )

        docs = [
            Document(content="Python is a programming language", id="doc1"),
            Document(content="JavaScript runs in browsers", id="doc2"),
        ]

        await rag.add_documents(docs)

        # Search should return results
        results = await rag.search("programming", top_k=2)

        assert len(results) > 0
        assert all(r.score >= 0 for r in results)

    async def test_add_file(self):
        """Test adding a file."""
        import aiofiles
        import aiofiles.os

        embeddings = MockEmbeddingProvider()
        rag = LocalRAG(embeddings=embeddings)

        # Create a temporary file using aiofiles
        temp_path = "/tmp/test_rag_file.txt"
        async with aiofiles.open(temp_path, mode="w") as f:
            await f.write("This is test file content for RAG.")

        try:
            doc_id = await rag.add_file(temp_path)

            assert doc_id is not None

            # Search should find the content
            results = await rag.search("test file content", top_k=1)
            assert len(results) > 0
        finally:
            await aiofiles.os.remove(temp_path)

    async def test_delete_document(self):
        """Test deleting a document."""
        embeddings = MockEmbeddingProvider()
        store = InMemoryVectorStore()
        rag = LocalRAG(embeddings=embeddings, vector_store=store)

        doc = Document(content="Test content", id="test-doc")
        await rag.add_documents([doc])

        initial_count = await store.count()
        assert initial_count > 0

        deleted = await rag.delete_document("test-doc")
        assert deleted is True

        final_count = await store.count()
        assert final_count == 0

    async def test_clear(self):
        """Test clearing all documents."""
        embeddings = MockEmbeddingProvider()
        store = InMemoryVectorStore()
        rag = LocalRAG(embeddings=embeddings, vector_store=store)

        docs = [
            Document(content="First", id="doc1"),
            Document(content="Second", id="doc2"),
        ]
        await rag.add_documents(docs)

        await rag.clear()

        count = await store.count()
        assert count == 0

    async def test_file_not_found(self):
        """Test adding a non-existent file."""
        embeddings = MockEmbeddingProvider()
        rag = LocalRAG(embeddings=embeddings)

        # Use a path that definitely doesn't exist
        with pytest.raises(FileNotFoundError):
            await rag.add_file("/tmp/nonexistent_rag_test_file_12345.txt")

    async def test_custom_chunker(self):
        """Test using a custom chunker."""
        embeddings = MockEmbeddingProvider()
        chunker = SentenceChunker(max_chunk_size=50)
        rag = LocalRAG(embeddings=embeddings, chunker=chunker)

        doc = Document(
            content="First sentence. Second sentence. Third sentence.",
            id="doc1",
        )
        await rag.add_documents([doc])

        # Verify the chunker was used
        results = await rag.search("sentence", top_k=5)
        assert len(results) > 0
