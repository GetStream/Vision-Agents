"""Tests for Gemini built-in tools."""

from unittest.mock import MagicMock

from google.genai import types

from vision_agents.plugins.gemini import tools


class TestGoogleSearch:
    """Tests for GoogleSearch tool."""

    def test_to_tool(self):
        """Test GoogleSearch converts to Tool correctly."""
        tool = tools.GoogleSearch()
        result = tool.to_tool()

        assert isinstance(result, types.Tool)
        assert result.google_search is not None


class TestCodeExecution:
    """Tests for CodeExecution tool."""

    def test_to_tool(self):
        """Test CodeExecution converts to Tool correctly."""
        tool = tools.CodeExecution()
        result = tool.to_tool()

        assert isinstance(result, types.Tool)
        assert result.code_execution is not None


class TestURLContext:
    """Tests for URLContext tool."""

    def test_to_tool(self):
        """Test URLContext converts to Tool correctly."""
        tool = tools.URLContext()
        result = tool.to_tool()

        assert isinstance(result, types.Tool)
        assert result.url_context is not None


class TestGoogleMaps:
    """Tests for GoogleMaps tool."""

    def test_to_tool(self):
        """Test GoogleMaps converts to Tool correctly."""
        tool = tools.GoogleMaps()
        result = tool.to_tool()

        assert isinstance(result, types.Tool)
        assert result.google_maps is not None


class TestComputerUse:
    """Tests for ComputerUse tool."""

    def test_to_tool(self):
        """Test ComputerUse converts to Tool correctly."""
        tool = tools.ComputerUse()
        result = tool.to_tool()

        assert isinstance(result, types.Tool)
        assert result.computer_use is not None

    def test_custom_environment(self):
        """Test ComputerUse with custom environment."""
        tool = tools.ComputerUse(environment=types.Environment.ENVIRONMENT_BROWSER)
        result = tool.to_tool()

        assert result.computer_use.environment == types.Environment.ENVIRONMENT_BROWSER


class TestFileSearch:
    """Tests for FileSearch tool."""

    def test_to_tool_with_created_store(self):
        """Test FileSearch converts to Tool when store is created."""
        # Mock the store
        mock_store = MagicMock()
        mock_store.is_created = True
        mock_store.get_tool.return_value = types.Tool(
            file_search=types.FileSearch(file_search_store_names=["test-store"])
        )

        tool = tools.FileSearch(mock_store)
        result = tool.to_tool()

        assert isinstance(result, types.Tool)
        assert result.file_search is not None
        mock_store.get_tool.assert_called_once()

    def test_to_tool_raises_when_store_not_created(self):
        """Test FileSearch raises error when store is not created."""
        mock_store = MagicMock()
        mock_store.is_created = False

        tool = tools.FileSearch(mock_store)

        try:
            tool.to_tool()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not created" in str(e)
