"""Tests for Gemini File Search RAG provider."""

import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

from vision_agents.core.rag import Document, RetrievalResult
from vision_agents.core.rag.events import (
    RAGDocumentAddedEvent,
    RAGFileAddedEvent,
    RAGRetrievalCompleteEvent,
)
from vision_agents.plugins.gemini import GeminiFileSearchRAG, LLM

load_dotenv()


class TestGeminiFileSearchRAG:
    """Tests for GeminiFileSearchRAG provider."""

    def test_supported_mime_types(self):
        """Test that common file types have MIME type mappings."""
        from vision_agents.plugins.gemini.rag import SUPPORTED_MIME_TYPES

        assert ".txt" in SUPPORTED_MIME_TYPES
        assert ".md" in SUPPORTED_MIME_TYPES
        assert ".pdf" in SUPPORTED_MIME_TYPES
        assert ".py" in SUPPORTED_MIME_TYPES
        assert ".json" in SUPPORTED_MIME_TYPES

    async def test_get_mime_type(self):
        """Test MIME type detection."""
        rag = GeminiFileSearchRAG()

        assert rag._get_mime_type("test.txt") == "text/plain"
        assert rag._get_mime_type("test.md") == "text/markdown"
        assert rag._get_mime_type("test.py") == "text/x-python"
        assert rag._get_mime_type("test.json") == "application/json"
        assert rag._get_mime_type("test.pdf") == "application/pdf"

    def test_document_model(self):
        """Test Document model creation and ID generation."""
        doc = Document(content="Test content", metadata={"source": "test"})

        assert doc.content == "Test content"
        assert doc.metadata == {"source": "test"}
        assert doc.id is not None
        assert len(doc.id) > 0

    def test_document_with_custom_id(self):
        """Test Document model with custom ID."""
        doc = Document(content="Test", id="custom-id")

        assert doc.id == "custom-id"

    def test_retrieval_result_citation(self):
        """Test RetrievalResult citation formatting."""
        # With explicit citation
        result = RetrievalResult(
            content="Test content",
            score=0.9,
            citation="[manual.pdf, page 5]",
        )
        assert result.format_citation() == "[manual.pdf, page 5]"

        # With source in metadata
        result2 = RetrievalResult(
            content="Test content",
            score=0.8,
            metadata={"source": "faq.md"},
            chunk_index=2,
        )
        assert result2.format_citation() == "[faq.md, chunk 2]"

        # With document_id only
        result3 = RetrievalResult(
            content="Test content",
            score=0.7,
            document_id="abc12345-6789",
        )
        assert result3.format_citation() == "[doc:abc12345]"

    async def test_rag_provider_initialization(self):
        """Test RAG provider can be initialized."""
        rag = GeminiFileSearchRAG()

        assert rag.store_name is None
        assert rag._top_k == 5

    async def test_rag_provider_with_store_name(self):
        """Test RAG provider with custom store name."""
        rag = GeminiFileSearchRAG(store_name="test-store", top_k=10)

        assert rag._store_name == "test-store"
        assert rag._top_k == 10

    @pytest.mark.integration
    async def test_add_file_not_found(self):
        """Test that adding a non-existent file raises an error."""
        rag = GeminiFileSearchRAG()

        with pytest.raises(FileNotFoundError):
            await rag.add_file("/nonexistent/file.txt")

    @pytest.mark.integration
    async def test_create_store_and_upload(self):
        """Test creating a store and uploading a file."""
        rag = GeminiFileSearchRAG()

        # Track events
        file_added_events = []

        @rag.events.subscribe
        async def on_file_added(event: RAGFileAddedEvent):
            file_added_events.append(event)

        # Create a test file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(
                "Vision Agents is a framework for building AI agents that can "
                "process video and audio in real-time. It supports multiple LLM "
                "providers including OpenAI, Anthropic, and Google Gemini."
            )
            temp_path = f.name

        try:
            # Upload file
            file_id = await rag.add_file(temp_path)

            # Verify store was created
            assert rag.store_name is not None

            # Verify file was uploaded
            assert file_id is not None

            # Wait for events
            await rag.events.wait()

            # Verify event was emitted
            assert len(file_added_events) == 1
            assert file_added_events[0].file_path == temp_path
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
            await rag.clear()

    @pytest.mark.integration
    async def test_search(self):
        """Test searching for relevant content."""
        rag = GeminiFileSearchRAG()

        # Track events
        retrieval_events = []

        @rag.events.subscribe
        async def on_retrieval(event: RAGRetrievalCompleteEvent):
            retrieval_events.append(event)

        # Create test files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(
                "Password Reset Instructions:\n"
                "1. Go to the login page\n"
                "2. Click 'Forgot Password'\n"
                "3. Enter your email address\n"
                "4. Check your email for a reset link\n"
                "5. Click the link and create a new password"
            )
            password_file = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(
                "Account Settings:\n"
                "You can update your profile picture, display name, and "
                "notification preferences in the account settings page."
            )
            settings_file = f.name

        try:
            # Upload files
            await rag.add_file(password_file, metadata={"topic": "password"})
            await rag.add_file(settings_file, metadata={"topic": "settings"})

            # Search for password-related content
            results = await rag.search_with_events(
                "How do I reset my password?",
                top_k=3,
            )

            # Wait for events
            await rag.events.wait()

            # Verify results
            assert len(results) > 0

            # Verify events
            assert len(retrieval_events) == 1
            assert retrieval_events[0].query == "How do I reset my password?"
        finally:
            # Clean up
            Path(password_file).unlink(missing_ok=True)
            Path(settings_file).unlink(missing_ok=True)
            await rag.clear()

    @pytest.mark.integration
    async def test_add_documents(self):
        """Test adding Document objects."""
        rag = GeminiFileSearchRAG()

        # Track events
        doc_events = []

        @rag.events.subscribe
        async def on_doc_added(event: RAGDocumentAddedEvent):
            doc_events.append(event)

        try:
            # Add documents
            docs = [
                Document(
                    content="Python is a programming language known for its simplicity.",
                    metadata={"topic": "python"},
                ),
                Document(
                    content="JavaScript is commonly used for web development.",
                    metadata={"topic": "javascript"},
                ),
            ]
            await rag.add_documents(docs)

            # Wait for events
            await rag.events.wait()

            # Verify events
            assert len(doc_events) == 2
        finally:
            await rag.clear()

    @pytest.mark.integration
    async def test_llm_with_rag(self):
        """Test LLM with RAG provider for automatic context injection."""
        rag = GeminiFileSearchRAG()
        llm = LLM(model="gemini-2.0-flash")

        # Create a test file with specific information
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(
                "Company Policy Document\n\n"
                "Vacation Policy:\n"
                "All employees are entitled to 25 days of paid vacation per year. "
                "Vacation requests must be submitted at least 2 weeks in advance. "
                "Unused vacation days can be carried over to the next year, "
                "up to a maximum of 5 days.\n\n"
                "Remote Work Policy:\n"
                "Employees may work remotely up to 3 days per week. "
                "Remote work requests must be approved by your manager."
            )
            policy_file = f.name

        try:
            # Upload file to RAG
            await rag.add_file(policy_file, metadata={"type": "policy"})

            # Attach RAG to LLM
            llm.set_rag_provider(rag, top_k=3)

            # Ask a question that requires RAG context
            response = await llm.simple_response(
                "How many vacation days do employees get per year?"
            )

            # The response should mention 25 days
            assert "25" in response.text
        finally:
            Path(policy_file).unlink(missing_ok=True)
            await rag.clear()

    @pytest.mark.integration
    async def test_get_file_search_tool(self):
        """Test getting the file search tool for native integration."""
        rag = GeminiFileSearchRAG()

        # Should raise error before store is created
        with pytest.raises(ValueError):
            rag.get_file_search_tool()

        # Create a test file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Test content for file search tool.")
            temp_path = f.name

        try:
            # Upload file to create store
            await rag.add_file(temp_path)

            # Now should work
            tool = rag.get_file_search_tool()

            assert tool.file_search is not None
            assert rag.store_name in tool.file_search.file_search_store_names
        finally:
            Path(temp_path).unlink(missing_ok=True)
            await rag.clear()


class TestRAGProviderBase:
    """Tests for the base RAGProvider class."""

    async def test_build_context_prompt_empty(self):
        """Test context prompt with no results."""
        rag = GeminiFileSearchRAG()
        prompt = rag.build_context_prompt([])

        assert prompt == ""

    async def test_build_context_prompt_with_results(self):
        """Test context prompt with results."""
        rag = GeminiFileSearchRAG()
        results = [
            RetrievalResult(
                content="First relevant passage.",
                score=0.9,
                citation="[doc1.txt]",
            ),
            RetrievalResult(
                content="Second relevant passage.",
                score=0.8,
                metadata={"source": "doc2.txt"},
            ),
        ]
        prompt = rag.build_context_prompt(results)

        assert "First relevant passage" in prompt
        assert "Second relevant passage" in prompt
        assert "[doc1.txt]" in prompt
        assert "[doc2.txt]" in prompt

    async def test_build_context_prompt_no_citations(self):
        """Test context prompt without citations."""
        rag = GeminiFileSearchRAG()
        results = [
            RetrievalResult(
                content="Test passage.",
                score=0.9,
            ),
        ]
        prompt = rag.build_context_prompt(results, include_citations=False)

        assert "Test passage" in prompt
        assert "[unknown source]" not in prompt

