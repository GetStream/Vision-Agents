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

llm = gemini.Realtime("gemini-2.0-flash-live-001")
computer_use.ComputerUseToolkit().register(llm)

agent = Agent(llm=llm)
```

With screen sharing active, the model sees the desktop and can call:

| Tool | Description |
|------|-------------|
| `click(x, y, button)` | Click at coordinates |
| `double_click(x, y)` | Double-click at coordinates |
| `type_text(text)` | Type text into the focused element |
| `key_press(keys)` | Press a key combo, e.g. `"cmd+c"` |
| `scroll(x, y, clicks, direction)` | Scroll at coordinates |
| `mouse_move(x, y)` | Move cursor to coordinates |
| `open_path(path)` | Open a file/folder with the OS default handler |

## How it works

The SDK's screen-share pipeline (`TrackType.SCREEN_SHARE`) feeds frames to the VLM/Realtime model continuously. The model sees the screen, decides what to do, and calls action tools backed by [PyAutoGUI](https://pyautogui.readthedocs.io/).

## Platform support

Actions use PyAutoGUI (macOS, Linux, Windows). `open_path` uses `open` (macOS), `xdg-open` (Linux), or `explorer` (Windows).
