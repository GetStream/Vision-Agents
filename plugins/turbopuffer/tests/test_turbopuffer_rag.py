"""Tests for TurboPufferRAG."""

import uuid

import pytest
from dotenv import load_dotenv

from vision_agents.core.rag import Document
from vision_agents.plugins.turbopuffer import TurboPufferRAG

load_dotenv()

# Skip blockbuster for all tests in this module (they make real API calls)
pytestmark = [pytest.mark.integration, pytest.mark.skip_blockbuster]


@pytest.fixture
async def rag():
    """Create a RAG instance for testing, clean up after."""
    namespace = f"test-rag-{uuid.uuid4().hex[:8]}"
    rag = TurboPufferRAG(namespace=namespace)
    yield rag
    await rag.clear()
    await rag.close()


@pytest.fixture
def unique_doc():
    """Create a document with unique content."""
    unique_id = uuid.uuid4()
    return Document(
        text=f"Test document {unique_id}. Contains quantum computing and AI info.",
        source="test_doc.txt",
    ), str(unique_id)


async def test_basic_upload_and_search(rag: TurboPufferRAG, unique_doc):
    """Upload a document and verify it can be found."""
    doc, unique_id = unique_doc
    count = await rag.add_documents([doc])

    assert count >= 1
    assert len(rag.indexed_files) == 1

    result = await rag.search(f"document {unique_id}")
    assert unique_id in result


async def test_vector_search_mode(rag: TurboPufferRAG):
    """Test vector-only search finds semantically similar content."""
    doc = Document(text="Neural networks for pattern recognition.", source="ml.txt")
    await rag.add_documents([doc])

    result = await rag.search("deep learning patterns", mode="vector")
    assert "neural" in result.lower() or "pattern" in result.lower()


async def test_bm25_search_mode(rag: TurboPufferRAG):
    """Test BM25 keyword search finds exact matches."""
    unique_sku = f"SKU-{uuid.uuid4().hex[:8].upper()}"
    doc = Document(
        text=f"Product code: {unique_sku}. High-quality widget.", source="product.txt"
    )
    await rag.add_documents([doc])

    result = await rag.search(unique_sku, mode="bm25")
    assert unique_sku in result


async def test_hybrid_search_mode(rag: TurboPufferRAG):
    """Test hybrid search combines vector and BM25."""
    doc = Document(
        text="The API endpoint supports real-time data streaming.", source="api.txt"
    )
    await rag.add_documents([doc])

    result = await rag.search("real-time streaming API")
    assert "streaming" in result.lower() or "api" in result.lower()


async def test_batch_upload_multiple_documents(rag: TurboPufferRAG):
    """Test uploading multiple documents in a batch."""
    docs = [
        Document(text=f"Document about {topic}: {uuid.uuid4()}", source=f"{topic}.txt")
        for topic in ["cats", "dogs", "birds"]
    ]

    count = await rag.add_documents(docs)
    assert count >= 3
    assert len(rag.indexed_files) == 3


async def test_search_empty_namespace(rag: TurboPufferRAG):
    """Test search returns appropriate message when namespace is empty."""
    result = await rag.search("anything")
    assert "No relevant information found" in result


async def test_clear_removes_all_documents(rag: TurboPufferRAG, unique_doc):
    """Test that clear() removes all indexed documents."""
    doc, _ = unique_doc
    await rag.add_documents([doc])
    assert len(rag.indexed_files) == 1

    await rag.clear()
    assert len(rag.indexed_files) == 0

    result = await rag.search("anything")
    assert "No relevant information found" in result
