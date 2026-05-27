import uuid

import pytest
from dotenv import load_dotenv

from vision_agents.core.rag import Document
from vision_agents.plugins.qdrant import QdrantRAG

load_dotenv()

pytestmark = [pytest.mark.integration, pytest.mark.skip_blockbuster]


@pytest.fixture
async def rag():
    collection = f"test-rag-{uuid.uuid4().hex[:8]}"
    rag = QdrantRAG(collection=collection)
    yield rag
    await rag.clear()
    await rag.close()


@pytest.fixture
def unique_doc():
    unique_id = uuid.uuid4()
    return Document(
        text=f"Test document {unique_id}. Contains quantum computing and AI info.",
        source="test_doc.txt",
    ), str(unique_id)


async def test_basic_upload_and_search(rag: QdrantRAG, unique_doc):
    doc, unique_id = unique_doc
    count = await rag.add_documents([doc])

    assert count >= 1
    assert len(rag.indexed_files) == 1

    result = await rag.search(f"document {unique_id}")
    assert unique_id in result


async def test_vector_search_mode(rag: QdrantRAG):
    doc = Document(text="Neural networks for pattern recognition.", source="ml.txt")
    await rag.add_documents([doc])

    result = await rag.search("deep learning patterns", mode="vector")
    assert "neural" in result.lower() or "pattern" in result.lower()


async def test_bm25_search_mode(rag: QdrantRAG):
    unique_sku = f"SKU-{uuid.uuid4().hex[:8].upper()}"
    doc = Document(
        text=f"Product code: {unique_sku}. High-quality widget.", source="product.txt"
    )
    await rag.add_documents([doc])

    result = await rag.search(unique_sku, mode="bm25")
    assert unique_sku in result


async def test_hybrid_search_mode(rag: QdrantRAG):
    doc = Document(
        text="The API endpoint supports real-time data streaming.", source="api.txt"
    )
    await rag.add_documents([doc])

    result = await rag.search("real-time streaming API")
    assert "streaming" in result.lower() or "api" in result.lower()


async def test_batch_upload_multiple_documents(rag: QdrantRAG):
    docs = [
        Document(text=f"Document about {topic}: {uuid.uuid4()}", source=f"{topic}.txt")
        for topic in ["cats", "dogs", "birds"]
    ]

    count = await rag.add_documents(docs)
    assert count >= 3
    assert len(rag.indexed_files) == 3


async def test_search_empty_collection(rag: QdrantRAG):
    result = await rag.search("anything")
    assert "No relevant information found" in result


async def test_clear_removes_all_documents(rag: QdrantRAG, unique_doc):
    doc, _ = unique_doc
    await rag.add_documents([doc])
    assert len(rag.indexed_files) == 1

    await rag.clear()
    assert len(rag.indexed_files) == 0

    result = await rag.search("anything")
    assert "No relevant information found" in result
