"""Tests for OpenAI Vector Store RAG provider."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from vision_agents.plugins.openai import OpenAIVectorStoreRAG


class TestOpenAIVectorStoreRAG:
    """Tests for OpenAIVectorStoreRAG."""

    async def test_init_with_store_id(self):
        """Test initialization with existing store ID."""
        mock_client = AsyncMock()
        rag = OpenAIVectorStoreRAG(
            store_id="vs_existing123",
            client=mock_client,
        )

        assert rag.store_id == "vs_existing123"

    async def test_init_with_custom_client(self):
        """Test that custom client is used when provided."""
        mock_client = AsyncMock()

        rag = OpenAIVectorStoreRAG(client=mock_client)

        assert rag._client == mock_client

    async def test_ensure_store_creates_new(self):
        """Test that _ensure_store creates a new store when needed."""
        mock_client = AsyncMock()
        mock_store = MagicMock()
        mock_store.id = "vs_new123"
        mock_client.vector_stores.create = AsyncMock(return_value=mock_store)

        rag = OpenAIVectorStoreRAG(client=mock_client)
        store_id = await rag._ensure_store()

        assert store_id == "vs_new123"
        mock_client.vector_stores.create.assert_called_once()

    async def test_ensure_store_reuses_existing(self):
        """Test that _ensure_store reuses existing store ID."""
        mock_client = AsyncMock()

        rag = OpenAIVectorStoreRAG(
            store_id="vs_existing",
            client=mock_client,
        )
        store_id = await rag._ensure_store()

        assert store_id == "vs_existing"
        mock_client.vector_stores.create.assert_not_called()

    async def test_search_returns_results(self):
        """Test search returns properly formatted results."""
        mock_client = AsyncMock()

        # Mock search results
        mock_result = MagicMock()
        mock_result.file_id = "file_123"
        mock_result.filename = "test.txt"
        mock_result.score = 0.95
        mock_result.attributes = {"key": "value"}

        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = "Retrieved content"
        mock_result.content = [mock_content]

        # Create async iterator for search results
        async def async_iter():
            yield mock_result

        mock_paginator = MagicMock()
        mock_paginator.__aiter__ = lambda self: async_iter()
        mock_client.vector_stores.search = AsyncMock(return_value=mock_paginator)

        rag = OpenAIVectorStoreRAG(
            store_id="vs_test",
            client=mock_client,
        )
        results = await rag.search("test query", top_k=5)

        assert len(results) == 1
        assert results[0].content == "Retrieved content"
        assert results[0].score == 0.95
        assert results[0].document_id == "file_123"
        assert results[0].citation == "[test.txt]"

    async def test_delete_document_success(self):
        """Test successful document deletion."""
        mock_client = AsyncMock()
        mock_client.vector_stores.files.delete = AsyncMock()

        rag = OpenAIVectorStoreRAG(
            store_id="vs_test",
            client=mock_client,
        )
        # Manually add a tracked file
        rag._file_ids["doc1"] = "file_123"

        deleted = await rag.delete_document("doc1")

        assert deleted is True
        assert "doc1" not in rag._file_ids
        mock_client.vector_stores.files.delete.assert_called_once()

    async def test_delete_document_not_found(self):
        """Test deletion of non-existent document."""
        mock_client = AsyncMock()

        rag = OpenAIVectorStoreRAG(
            store_id="vs_test",
            client=mock_client,
        )

        deleted = await rag.delete_document("nonexistent")

        assert deleted is False
        mock_client.vector_stores.files.delete.assert_not_called()

    async def test_clear_deletes_store(self):
        """Test clear deletes the vector store."""
        mock_client = AsyncMock()
        mock_client.vector_stores.delete = AsyncMock()

        rag = OpenAIVectorStoreRAG(
            store_id="vs_test",
            client=mock_client,
        )

        await rag.clear()

        assert rag.store_id is None
        mock_client.vector_stores.delete.assert_called_once()


@pytest.mark.integration
class TestOpenAIVectorStoreRAGIntegration:
    """Integration tests for OpenAI Vector Store RAG.

    These tests require a valid OPENAI_API_KEY environment variable.
    """

    async def test_create_store_and_search(self):
        """Test creating a store, adding content, and searching."""
        import tempfile
        from pathlib import Path

        rag = OpenAIVectorStoreRAG(store_name="test-integration-store")

        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(
                    "Python is a high-level programming language known for its "
                    "readability and versatility. It was created by Guido van Rossum."
                )
                temp_path = f.name

            try:
                # Add the file
                file_id = await rag.add_file(temp_path)
                assert file_id is not None

                # Search for content
                results = await rag.search("Who created Python?", top_k=3)

                # Should find relevant content
                assert len(results) > 0
                assert any("Python" in r.content for r in results)

            finally:
                Path(temp_path).unlink()

        finally:
            # Clean up
            await rag.clear()
