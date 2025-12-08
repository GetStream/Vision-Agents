"""Tests for the core RAG module."""

from vision_agents.core.rag import (
    Chunk,
    Document,
    RAGProvider,
    RetrievalResult,
)
from vision_agents.core.rag.events import (
    RAGDocumentAddedEvent,
    RAGFileAddedEvent,
    RAGRetrievalCompleteEvent,
    RAGRetrievalStartEvent,
)


class TestDocument:
    """Tests for the Document model."""

    def test_document_creation(self):
        """Test basic document creation."""
        doc = Document(content="Hello world")

        assert doc.content == "Hello world"
        assert doc.metadata == {}
        assert doc.id is not None

    def test_document_with_metadata(self):
        """Test document with metadata."""
        doc = Document(
            content="Test content",
            metadata={"source": "test.txt", "author": "test"},
        )

        assert doc.metadata["source"] == "test.txt"
        assert doc.metadata["author"] == "test"

    def test_document_with_custom_id(self):
        """Test document with custom ID."""
        doc = Document(content="Test", id="my-custom-id")

        assert doc.id == "my-custom-id"

    def test_document_auto_id_is_unique(self):
        """Test that auto-generated IDs are unique."""
        doc1 = Document(content="First")
        doc2 = Document(content="Second")

        assert doc1.id != doc2.id


class TestChunk:
    """Tests for the Chunk model."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            content="Chunk content",
            document_id="doc-123",
            index=0,
        )

        assert chunk.content == "Chunk content"
        assert chunk.document_id == "doc-123"
        assert chunk.index == 0

    def test_chunk_id_generation(self):
        """Test chunk ID is generated from document_id and index."""
        chunk = Chunk(
            content="Test",
            document_id="doc-abc",
            index=5,
        )

        assert chunk.id == "doc-abc_5"

    def test_chunk_with_positions(self):
        """Test chunk with character positions."""
        chunk = Chunk(
            content="Test content",
            document_id="doc-123",
            index=0,
            start_char=0,
            end_char=12,
        )

        assert chunk.start_char == 0
        assert chunk.end_char == 12


class TestRetrievalResult:
    """Tests for the RetrievalResult model."""

    def test_retrieval_result_creation(self):
        """Test basic retrieval result creation."""
        result = RetrievalResult(
            content="Retrieved content",
            score=0.95,
        )

        assert result.content == "Retrieved content"
        assert result.score == 0.95

    def test_format_citation_with_explicit_citation(self):
        """Test citation formatting with explicit citation."""
        result = RetrievalResult(
            content="Test",
            score=0.9,
            citation="[manual.pdf, page 5]",
        )

        assert result.format_citation() == "[manual.pdf, page 5]"

    def test_format_citation_with_source_metadata(self):
        """Test citation formatting with source in metadata."""
        result = RetrievalResult(
            content="Test",
            score=0.9,
            metadata={"source": "guide.md"},
        )

        assert result.format_citation() == "[guide.md]"

    def test_format_citation_with_source_and_chunk(self):
        """Test citation formatting with source and chunk index."""
        result = RetrievalResult(
            content="Test",
            score=0.9,
            metadata={"source": "guide.md"},
            chunk_index=3,
        )

        assert result.format_citation() == "[guide.md, chunk 3]"

    def test_format_citation_with_filename_metadata(self):
        """Test citation formatting with filename in metadata."""
        result = RetrievalResult(
            content="Test",
            score=0.9,
            metadata={"filename": "data.json"},
        )

        assert result.format_citation() == "[data.json]"

    def test_format_citation_with_document_id(self):
        """Test citation formatting with document ID only."""
        result = RetrievalResult(
            content="Test",
            score=0.9,
            document_id="abcdefgh-1234-5678",
        )

        assert result.format_citation() == "[doc:abcdefgh]"

    def test_format_citation_unknown_source(self):
        """Test citation formatting with no source info."""
        result = RetrievalResult(
            content="Test",
            score=0.9,
        )

        assert result.format_citation() == "[unknown source]"


class TestRAGEvents:
    """Tests for RAG events."""

    def test_retrieval_start_event(self):
        """Test RAGRetrievalStartEvent."""
        event = RAGRetrievalStartEvent(
            query="test query",
            top_k=5,
        )

        assert event.query == "test query"
        assert event.top_k == 5
        assert event.plugin_name == "rag"

    def test_retrieval_complete_event(self):
        """Test RAGRetrievalCompleteEvent."""
        results = [
            RetrievalResult(content="Result 1", score=0.9),
            RetrievalResult(content="Result 2", score=0.8),
        ]
        event = RAGRetrievalCompleteEvent(
            query="test query",
            results=results,
            retrieval_time_ms=150.5,
        )

        assert event.query == "test query"
        assert event.result_count == 2
        assert event.retrieval_time_ms == 150.5

    def test_document_added_event(self):
        """Test RAGDocumentAddedEvent."""
        event = RAGDocumentAddedEvent(
            document_id="doc-123",
            metadata={"source": "test.txt"},
            chunk_count=5,
        )

        assert event.document_id == "doc-123"
        assert event.chunk_count == 5

    def test_file_added_event(self):
        """Test RAGFileAddedEvent."""
        event = RAGFileAddedEvent(
            file_path="/path/to/file.pdf",
            file_id="file-abc",
            metadata={"type": "manual"},
        )

        assert event.file_path == "/path/to/file.pdf"
        assert event.file_id == "file-abc"


class MockRAGProvider(RAGProvider):
    """Mock RAG provider for testing base class functionality."""

    def __init__(self):
        super().__init__()
        self._documents: dict[str, Document] = {}
        self._chunks: list[Chunk] = []

    async def add_documents(self, documents: list[Document]) -> None:
        for doc in documents:
            self._documents[doc.id] = doc

    async def add_file(self, file_path: str, metadata: dict | None = None) -> str:
        doc_id = f"file-{file_path}"
        return doc_id

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        # Simple mock: return all documents as results
        results = []
        for doc in list(self._documents.values())[:top_k]:
            results.append(
                RetrievalResult(
                    content=doc.content,
                    score=1.0,
                    document_id=doc.id,
                    metadata=doc.metadata,
                )
            )
        return results

    async def delete_document(self, document_id: str) -> bool:
        if document_id in self._documents:
            del self._documents[document_id]
            return True
        return False


class TestRAGProviderBase:
    """Tests for the RAGProvider base class."""

    async def test_build_context_prompt_empty(self):
        """Test context prompt with no results."""
        provider = MockRAGProvider()
        prompt = provider.build_context_prompt([])

        assert prompt == ""

    async def test_build_context_prompt_with_results(self):
        """Test context prompt with results."""
        provider = MockRAGProvider()
        results = [
            RetrievalResult(
                content="First passage about Python.",
                score=0.95,
                citation="[python.md]",
            ),
            RetrievalResult(
                content="Second passage about JavaScript.",
                score=0.85,
                metadata={"source": "js.md"},
            ),
        ]

        prompt = provider.build_context_prompt(results)

        assert "First passage about Python" in prompt
        assert "Second passage about JavaScript" in prompt
        assert "[python.md]" in prompt
        assert "[js.md]" in prompt
        assert "Use the following context" in prompt

    async def test_build_context_prompt_no_citations(self):
        """Test context prompt without citations."""
        provider = MockRAGProvider()
        results = [
            RetrievalResult(
                content="Test content.",
                score=0.9,
            ),
        ]

        prompt = provider.build_context_prompt(results, include_citations=False)

        assert "Test content" in prompt
        assert "[unknown source]" not in prompt

    async def test_add_files(self):
        """Test adding multiple files."""
        provider = MockRAGProvider()

        ids = await provider.add_files(["/path/a.txt", "/path/b.txt"])

        assert len(ids) == 2
        assert "file-/path/a.txt" in ids
        assert "file-/path/b.txt" in ids

    async def test_search_with_events(self):
        """Test search with event emission."""
        provider = MockRAGProvider()

        # Add a document first
        doc = Document(content="Test document content")
        await provider.add_documents([doc])

        # Track events
        start_events = []
        complete_events = []

        @provider.events.subscribe
        async def on_start(event: RAGRetrievalStartEvent):
            start_events.append(event)

        @provider.events.subscribe
        async def on_complete(event: RAGRetrievalCompleteEvent):
            complete_events.append(event)

        # Perform search
        await provider.search_with_events("test query", top_k=3)

        # Wait for events
        await provider.events.wait()

        # Verify events were emitted
        assert len(start_events) == 1
        assert start_events[0].query == "test query"
        assert start_events[0].top_k == 3

        assert len(complete_events) == 1
        assert complete_events[0].query == "test query"
        assert complete_events[0].retrieval_time_ms > 0

    async def test_delete_document(self):
        """Test document deletion."""
        provider = MockRAGProvider()

        # Add a document
        doc = Document(content="Test", id="test-doc")
        await provider.add_documents([doc])

        # Verify it exists
        assert "test-doc" in provider._documents

        # Delete it
        deleted = await provider.delete_document("test-doc")
        assert deleted is True
        assert "test-doc" not in provider._documents

        # Try to delete again
        deleted_again = await provider.delete_document("test-doc")
        assert deleted_again is False
