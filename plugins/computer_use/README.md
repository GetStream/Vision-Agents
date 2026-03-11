# Computer Use Plugin

Model-agnostic desktop control tools for Vision Agents. Lets any LLM with vision (via screen share) interact with the user's desktop — clicking, typing, scrolling, and opening files.

## Install

```bash
pip install vision-agents-plugins-computer-use
```

## Usage

Register the tools on any LLM, then use with an agent that receives screen-share frames:

```python
from vision_agents.plugins import gemini, computer_use

llm = gemini.Realtime(fps=2)
computer_use.register(llm)

agent = Agent(
    llm=llm,
    processors=[computer_use.GridOverlayProcessor(fps=2)],
)
```

The `GridOverlayProcessor` draws a labeled grid on screen frames so the model can reference cells by name. Grid size is customizable:

```python
computer_use.register(llm, cols=10, rows=10)
computer_use.GridOverlayProcessor(cols=10, rows=10, fps=2)
```

With screen sharing active, the model sees the grid and can call:

| Tool | Description |
|------|-------------|
| `click(cell, position, button)` | Click at a grid cell |
| `double_click(cell, position)` | Double-click at a grid cell |
| `type_text(text)` | Type text into the focused element |
| `key_press(keys)` | Press a key combo, e.g. `"cmd+c"` |
| `scroll(cell, position, clicks, direction)` | Scroll at a grid cell |
| `mouse_move(cell, position)` | Move cursor to a grid cell |
| `open_path(path)` | Open a file/folder with the OS default handler |

## How it works

The SDK's screen-share pipeline (`TrackType.SCREEN_SHARE`) feeds frames to the VLM/Realtime model continuously. The `GridOverlayProcessor` annotates these frames with a labeled grid (e.g. A-O / 1-15). The model reads the grid labels, picks the right cell, and calls action tools backed by [PyAutoGUI](https://pyautogui.readthedocs.io/).

## Platform support

Actions use PyAutoGUI (macOS, Linux, Windows). `open_path` uses `open` (macOS), `xdg-open` (Linux), or `explorer` (Windows).
