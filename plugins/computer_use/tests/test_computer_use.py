"""Tests for the computer-use plugin."""

import pytest
from vision_agents.core.llm import FunctionRegistry

from vision_agents.plugins.computer_use import Grid, register
from vision_agents.plugins.computer_use._actions import (
    key_press,
    make_grid_actions,
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


class TestGrid:
    def test_defaults(self):
        grid = Grid()
        assert grid.cols == 15
        assert grid.rows == 15
        assert grid.col_labels[0] == "A"
        assert grid.col_labels[-1] == "O"

    def test_custom_size(self):
        grid = Grid(cols=5, rows=5)
        assert grid.cols == 5
        assert grid.rows == 5
        assert grid.col_labels == ["A", "B", "C", "D", "E"]
        assert grid.label == "A-E / 1-5"

    def test_invalid_cols(self):
        with pytest.raises(ValueError, match="cols must be 1-26"):
            Grid(cols=0)
        with pytest.raises(ValueError, match="cols must be 1-26"):
            Grid(cols=27)

    def test_invalid_rows(self):
        with pytest.raises(ValueError, match="rows must be 1-99"):
            Grid(rows=0)
        with pytest.raises(ValueError, match="rows must be 1-99"):
            Grid(rows=100)

    def test_cell_to_virtual_center(self):
        grid = Grid(cols=10, rows=10)
        vx, vy = grid.cell_to_virtual("A1")
        assert vx == 50
        assert vy == 50

    def test_cell_to_virtual_position(self):
        grid = Grid(cols=10, rows=10)
        vx, vy = grid.cell_to_virtual("A1", position="top-left")
        assert vx == 20
        assert vy == 20

    def test_cell_to_virtual_last_cell(self):
        grid = Grid(cols=10, rows=10)
        vx, vy = grid.cell_to_virtual("J10")
        assert vx == 950
        assert vy == 950

    def test_cell_to_virtual_invalid_cell(self):
        grid = Grid(cols=5, rows=5)
        with pytest.raises(ValueError, match="Invalid cell reference"):
            grid.cell_to_virtual("Z1")

    def test_cell_to_virtual_out_of_range_row(self):
        grid = Grid(cols=5, rows=5)
        with pytest.raises(ValueError, match="Row must be 1-5"):
            grid.cell_to_virtual("A6")

    def test_cell_to_virtual_invalid_position(self):
        grid = Grid(cols=5, rows=5)
        with pytest.raises(ValueError, match="Invalid position"):
            grid.cell_to_virtual("A1", position="middle")

    def test_label(self):
        assert Grid(cols=15, rows=15).label == "A-O / 1-15"
        assert Grid(cols=26, rows=1).label == "A-Z / 1-1"


class TestRegister:
    def test_register_adds_all_tools(self):
        llm = _FakeLLM()
        register(llm)

        registered = set(llm.function_registry._functions.keys())
        assert registered == EXPECTED_TOOLS

    def test_register_tool_schemas_have_descriptions(self):
        llm = _FakeLLM()
        register(llm)

        schemas = llm.function_registry.get_tool_schemas()
        for schema in schemas:
            assert schema["description"], f"{schema['name']} has no description"

    def test_register_is_idempotent(self):
        llm = _FakeLLM()
        register(llm)
        register(llm)

        schemas = llm.function_registry.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert len(names) == len(EXPECTED_TOOLS)

    def test_click_schema_has_parameters(self):
        llm = _FakeLLM()
        register(llm)

        schemas = {s["name"]: s for s in llm.function_registry.get_tool_schemas()}
        click_schema = schemas["click"]
        props = click_schema["parameters_schema"]["properties"]
        assert "cell" in props
        assert "position" in props
        assert "button" in props

    def test_scroll_schema_has_direction(self):
        llm = _FakeLLM()
        register(llm)

        schemas = {s["name"]: s for s in llm.function_registry.get_tool_schemas()}
        scroll_schema = schemas["scroll"]
        props = scroll_schema["parameters_schema"]["properties"]
        assert "direction" in props

    def test_custom_grid_size_in_descriptions(self):
        llm = _FakeLLM()
        register(llm, cols=8, rows=8)

        schemas = {s["name"]: s for s in llm.function_registry.get_tool_schemas()}
        assert "A-H" in schemas["click"]["description"]
        assert "1-8" in schemas["click"]["description"]

    def test_register_with_shared_grid(self):
        llm = _FakeLLM()
        grid = Grid(cols=10, rows=10)
        register(llm, grid=grid)

        schemas = {s["name"]: s for s in llm.function_registry.get_tool_schemas()}
        assert "A-J" in schemas["click"]["description"]
        assert "1-10" in schemas["click"]["description"]


class TestActions:
    """Test that action functions are valid async callables with correct signatures."""

    @pytest.mark.integration
    async def test_click_executes(self):
        actions = make_grid_actions(Grid())
        result = await actions["click"](cell="A1", position="center")
        assert "Clicked" in result

    @pytest.mark.integration
    async def test_double_click_executes(self):
        actions = make_grid_actions(Grid())
        result = await actions["double_click"](cell="A1", position="center")
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
        actions = make_grid_actions(Grid())
        result = await actions["scroll"](
            cell="A1", position="center", clicks=1, direction="down"
        )
        assert "Scrolled" in result

    @pytest.mark.integration
    async def test_mouse_move_executes(self):
        actions = make_grid_actions(Grid())
        result = await actions["mouse_move"](cell="A1", position="center")
        assert "Moved" in result
