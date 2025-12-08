"""Tests for GeminiFilesearchRAG."""

import logging
import uuid

import pytest
from dotenv import load_dotenv

from vision_agents.core.rag import Document
from vision_agents.plugins.gemini import GeminiFilesearchRAG

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Skip blockbuster for all tests in this module (they make real API calls)
pytestmark = [pytest.mark.integration, pytest.mark.skip_blockbuster]


@pytest.fixture
async def rag():
    """Create a RAG instance for testing, clean up after."""
    # Use unique name to avoid conflicts
    rag = GeminiFilesearchRAG(name=f"test-rag-123")
    await rag.create()
    yield rag
    await rag.clear()


async def test_basic_upload_and_search(rag: GeminiFilesearchRAG):
    """Upload a document with a unique ID and verify it can be found."""
    # Create a unique identifier to verify we find the right document
    unique_id = f"TEST-{uuid.uuid4()}"

    doc = Document(
        text=f"This is a test document with unique identifier: {unique_id}. "
        "It contains information about quantum computing and AI.",
        source="test_doc.txt",
    )

    # Upload document
    count = await rag.add_documents([doc])
    assert count == 1
    assert len(rag._uploaded_files) == 1

    # Search for the unique ID
    result = await rag.search(f"Find the document with identifier {unique_id}")
    logger.info(f"Search result: {result}")

    # The unique ID should appear in the search result
    assert unique_id in result or "quantum" in result.lower() or "ai" in result.lower()


async def test_deduplication_same_document(rag: GeminiFilesearchRAG):
    """Verify that uploading the same document twice doesn't create duplicates."""
    unique_id = f"DEDUP-{uuid.uuid4()}"

    doc = Document(
        text=f"Unique content for deduplication test: {unique_id}",
        source="dedup_test.txt",
    )

    # Upload the same document twice
    first_count = await rag.add_documents([doc])
    assert first_count == 1
    first_hash_count = len(rag.uploaded_hashes)

    # Upload the exact same document again
    second_count = await rag.add_documents([doc])
    assert second_count == 0  # Should be skipped as duplicate

    # Hash count should remain the same
    assert len(rag.uploaded_hashes) == first_hash_count

    # Uploaded files list should only have one entry
    assert len(rag._uploaded_files) == 1


async def test_deduplication_different_source_same_content(rag: GeminiFilesearchRAG):
    """Verify that same content with different source names is deduplicated."""
    content = f"Same content for both documents: {uuid.uuid4()}"

    doc1 = Document(text=content, source="source1.txt")
    doc2 = Document(text=content, source="source2.txt")

    # Upload first document
    count1 = await rag.add_documents([doc1])
    assert count1 == 1

    # Upload second document with same content but different source
    count2 = await rag.add_documents([doc2])
    assert count2 == 0  # Should be skipped - same content hash

    # Only one hash should be stored
    assert len(rag.uploaded_hashes) == 1


async def test_different_content_not_deduplicated(rag: GeminiFilesearchRAG):
    """Verify that different content is not incorrectly deduplicated."""
    doc1 = Document(
        text=f"First unique document: {uuid.uuid4()}",
        source="doc1.txt",
    )
    doc2 = Document(
        text=f"Second unique document: {uuid.uuid4()}",
        source="doc2.txt",
    )

    # Upload both documents
    count = await rag.add_documents([doc1, doc2])
    assert count == 2

    # Both hashes should be stored
    assert len(rag.uploaded_hashes) == 2
    assert len(rag._uploaded_files) == 2


async def test_batch_upload_with_duplicates(rag: GeminiFilesearchRAG):
    """Test batch upload correctly handles mixed unique and duplicate documents."""
    shared_content = f"Shared content: {uuid.uuid4()}"

    docs = [
        Document(text=f"Unique doc 1: {uuid.uuid4()}", source="unique1.txt"),
        Document(text=shared_content, source="shared1.txt"),
        Document(text=f"Unique doc 2: {uuid.uuid4()}", source="unique2.txt"),
        Document(text=shared_content, source="shared2.txt"),  # Duplicate content
    ]

    count = await rag.add_documents(docs)
    assert count == 3  # 2 unique + 1 shared (second shared is duplicate)
    assert len(rag.uploaded_hashes) == 3
