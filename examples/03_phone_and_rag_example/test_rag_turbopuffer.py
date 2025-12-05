"""Tests for TurboPuffer Hybrid RAG."""

import logging
from pathlib import Path

import pytest
from dotenv import load_dotenv

from rag_turbopuffer import TurboPufferRAG

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    """Create temp directory with a test document."""
    doc = tmp_path / "test.md"
    doc.write_text("# Stream Chat API\n\nThe Chat API supports real-time messaging and moderation.")
    return tmp_path


@pytest.mark.integration
async def test_index_and_search(knowledge_dir: Path):
    """Index a document and find it via search."""
    rag = TurboPufferRAG(namespace="test-rag")

    await rag.index_directory(knowledge_dir)

    result = await rag.search("chat messaging")
    assert "Chat" in result
    logger.info("result %s", result)

    await rag.close()
