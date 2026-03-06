"""Tests for the computer-use plugin."""

import pytest
from vision_agents.core.llm import FunctionRegistry

from vision_agents.plugins.computer_use import ComputerUseToolkit
from vision_agents.plugins.computer_use._actions import (
    click,
    double_click,
    key_press,
    mouse_move,
    scroll,
    type_text,
)

EXPECTED_TOOLS = {
    "click",
    "double_click",
    "type_text",
    "key_press",
    "scroll",
    "mouse_move",
    "open_path",
}


class _FakeLLM:
    """Minimal stand-in that exposes a real FunctionRegistry."""

    def __init__(self):
        self.function_registry = FunctionRegistry()


class TestComputerUseToolkit:
    def test_register_adds_all_tools(self):
        llm = _FakeLLM()
        ComputerUseToolkit().register(llm)

        registered = set(llm.function_registry._functions.keys())
        assert registered == EXPECTED_TOOLS

    def test_register_tool_schemas_have_descriptions(self):
        llm = _FakeLLM()
        ComputerUseToolkit().register(llm)

        schemas = llm.function_registry.get_tool_schemas()
        for schema in schemas:
            assert schema["description"], f"{schema['name']} has no description"

    def test_register_is_idempotent(self):
        llm = _FakeLLM()
        toolkit = ComputerUseToolkit()
        toolkit.register(llm)
        toolkit.register(llm)

        schemas = llm.function_registry.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert len(names) == len(EXPECTED_TOOLS)

    def test_click_schema_has_parameters(self):
        llm = _FakeLLM()
        ComputerUseToolkit().register(llm)

        schemas = {s["name"]: s for s in llm.function_registry.get_tool_schemas()}
        click_schema = schemas["click"]
        props = click_schema["parameters_schema"]["properties"]
        assert "x" in props
        assert "y" in props
        assert "button" in props

    def test_scroll_schema_has_direction(self):
        llm = _FakeLLM()
        ComputerUseToolkit().register(llm)

        schemas = {s["name"]: s for s in llm.function_registry.get_tool_schemas()}
        scroll_schema = schemas["scroll"]
        props = scroll_schema["parameters_schema"]["properties"]
        assert "direction" in props


class TestActions:
    """Test that action functions are valid async callables with correct signatures."""

    @pytest.mark.integration
    async def test_click_executes(self):
        result = await click(0, 0, button="left")
        assert "Clicked" in result

    @pytest.mark.integration
    async def test_double_click_executes(self):
        result = await double_click(0, 0)
        assert "Double-clicked" in result

    @pytest.mark.integration
    async def test_type_text_executes(self):
        result = await type_text("")
        assert "Typed" in result

    @pytest.mark.integration
    async def test_key_press_executes(self):
        result = await key_press("shift")
        assert "Pressed" in result

    @pytest.mark.integration
    async def test_scroll_executes(self):
        result = await scroll(0, 0, clicks=1, direction="down")
        assert "Scrolled" in result

    @pytest.mark.integration
    async def test_mouse_move_executes(self):
        result = await mouse_move(0, 0)
        assert "Moved" in result
