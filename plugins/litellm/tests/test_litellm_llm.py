"""Tests for LiteLLM plugin."""

import ast
from pathlib import Path

import pytest

PLUGIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "vision_agents"
    / "plugins"
    / "litellm"
    / "litellm_llm.py"
)


class TestLiteLLMPluginStructure:
    def _parse(self):
        return ast.parse(PLUGIN_PATH.read_text())

    def test_file_exists(self):
        assert PLUGIN_PATH.exists()

    def test_has_litellm_chat_completions_class(self):
        tree = self._parse()
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert "LiteLLMChatCompletions" in classes

    def test_inherits_llm(self):
        tree = self._parse()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "LiteLLMChatCompletions":
                base_names = [b.id for b in node.bases if isinstance(b, ast.Name)]
                assert "LLM" in base_names
                return
        pytest.fail("LiteLLMChatCompletions not found")

    def test_has_simple_response(self):
        tree = self._parse()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "LiteLLMChatCompletions":
                methods = [
                    n.name for n in node.body if isinstance(n, ast.AsyncFunctionDef)
                ]
                assert "simple_response" in methods
                assert "create_response" in methods
                return

    def test_has_streaming_handler(self):
        src = PLUGIN_PATH.read_text()
        assert "_handle_streaming" in src
        assert "_handle_non_streaming" in src

    def test_uses_drop_params_true(self):
        src = PLUGIN_PATH.read_text()
        assert '"drop_params": True' in src

    def test_uses_litellm_acompletion(self):
        src = PLUGIN_PATH.read_text()
        assert "litellm.acompletion" in src

    def test_emits_events(self):
        src = PLUGIN_PATH.read_text()
        assert "LLMRequestStartedEvent" in src
        assert "LLMResponseChunkEvent" in src
        assert "LLMResponseCompletedEvent" in src

    def test_plugin_name(self):
        src = PLUGIN_PATH.read_text()
        assert 'PLUGIN_NAME = "litellm"' in src

    def test_converts_tools_to_provider_format(self):
        src = PLUGIN_PATH.read_text()
        assert "_convert_tools_to_provider_format" in src

    def test_extracts_tool_calls(self):
        src = PLUGIN_PATH.read_text()
        assert "_extract_tool_calls_from_response" in src


class TestPluginPackage:
    def test_pyproject_exists(self):
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        assert pyproject.exists()

    def test_litellm_in_dependencies(self):
        pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
        assert "litellm" in pyproject

    def test_init_exports_class(self):
        init = (
            Path(__file__).resolve().parents[1]
            / "vision_agents"
            / "plugins"
            / "litellm"
            / "__init__.py"
        ).read_text()
        assert "LiteLLMChatCompletions" in init
