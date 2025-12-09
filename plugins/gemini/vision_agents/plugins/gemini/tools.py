"""Gemini built-in tools.

This module provides wrapper classes for Gemini's built-in tools, making them
easy to use with GeminiLLM.

See: https://ai.google.dev/gemini-api/docs/tools

Available tools:
- FileSearch: RAG over your documents
- GoogleSearch: Ground responses with web data
- CodeExecution: Run Python code
- URLContext: Read specific web pages
- GoogleMaps: Location-aware queries (Preview)
- ComputerUse: Interact with browser UIs (Preview)

Usage:
    from vision_agents.plugins.gemini import LLM, tools

    llm = LLM(
        model="gemini-2.5-flash",
        tools=[
            tools.GoogleSearch(),
            tools.CodeExecution(),
        ]
    )
"""

import abc
from typing import TYPE_CHECKING

from google.genai import types

if TYPE_CHECKING:
    from .file_search import GeminiFilesearchRAG


class GeminiTool(abc.ABC):
    """Abstract base class for Gemini built-in tools."""

    @abc.abstractmethod
    def to_tool(self) -> types.Tool:
        """Convert to a Gemini Tool object."""


class FileSearch(GeminiTool):
    """File Search tool for RAG over your documents.

    File Search imports, chunks, and indexes your data to enable fast
    retrieval of relevant information. Search is performed by Gemini's
    infrastructure.

    See: https://ai.google.dev/gemini-api/docs/file-search

    Usage:
        # Create and populate a store
        store = gemini.FileSearchStore(name="my-kb")
        await store.create()
        await store.add_directory("./knowledge")

        # Use with LLM
        llm = gemini.LLM(tools=[gemini.tools.FileSearch(store)])
    """

    def __init__(self, store: "GeminiFilesearchRAG"):
        """Initialize FileSearch tool.

        Args:
            store: A GeminiFilesearchRAG store that has been created and populated.
        """
        self.store = store

    def to_tool(self) -> types.Tool:
        """Convert to a Gemini Tool object."""
        if not self.store.is_created:
            raise ValueError("FileSearch store not created. Call store.create() first.")
        return self.store.get_tool()


class GoogleSearch(GeminiTool):
    """Google Search tool for grounding responses with web data.

    Grounds responses in current events and facts from the web to reduce
    hallucinations.

    See: https://ai.google.dev/gemini-api/docs/google-search

    Usage:
        llm = gemini.LLM(tools=[gemini.tools.GoogleSearch()])
        response = await llm.send_message("What happened in the news today?")
    """

    def to_tool(self) -> types.Tool:
        """Convert to a Gemini Tool object."""
        return types.Tool(google_search=types.GoogleSearch())


class CodeExecution(GeminiTool):
    """Code Execution tool for running Python code.

    Allows the model to write and run Python code to solve math problems
    or process data accurately.

    See: https://ai.google.dev/gemini-api/docs/code-execution

    Usage:
        llm = gemini.LLM(tools=[gemini.tools.CodeExecution()])
        response = await llm.send_message("Calculate the factorial of 20")
    """

    def to_tool(self) -> types.Tool:
        """Convert to a Gemini Tool object."""
        return types.Tool(code_execution=types.ToolCodeExecution())


class URLContext(GeminiTool):
    """URL Context tool for reading web pages.

    Directs the model to read and analyze content from specific web pages
    or documents.

    See: https://ai.google.dev/gemini-api/docs/url-context

    Usage:
        llm = gemini.LLM(tools=[gemini.tools.URLContext()])
        response = await llm.send_message(
            "Summarize the content at https://example.com/article"
        )
    """

    def to_tool(self) -> types.Tool:
        """Convert to a Gemini Tool object."""
        return types.Tool(url_context=types.UrlContext())


class GoogleMaps(GeminiTool):
    """Google Maps tool for location-aware queries (Preview).

    Build location-aware assistants that can find places, get directions,
    and provide rich local context.

    See: https://ai.google.dev/gemini-api/docs/maps-grounding

    Usage:
        llm = gemini.LLM(tools=[gemini.tools.GoogleMaps()])
        response = await llm.send_message("Find coffee shops near Times Square")
    """

    def to_tool(self) -> types.Tool:
        """Convert to a Gemini Tool object."""
        return types.Tool(google_maps=types.GoogleMaps())


class ComputerUse(GeminiTool):
    """Computer Use tool for browser automation (Preview).

    Enables Gemini to view a screen and generate actions to interact with
    web browser UIs. Actions are executed client-side.

    See: https://ai.google.dev/gemini-api/docs/computer-use

    Note: This is a preview feature and requires client-side execution
    of the generated actions.

    Usage:
        llm = gemini.LLM(tools=[gemini.tools.ComputerUse()])
        # Model will generate actions like click, type, scroll
    """

    def __init__(
        self, environment: types.Environment = types.Environment.ENVIRONMENT_BROWSER
    ):
        """Initialize ComputerUse tool.

        Args:
            environment: The environment type. Use types.Environment.ENVIRONMENT_BROWSER
                for browser automation.
        """
        self.environment = environment

    def to_tool(self) -> types.Tool:
        """Convert to a Gemini Tool object."""
        return types.Tool(computer_use=types.ComputerUse(environment=self.environment))
