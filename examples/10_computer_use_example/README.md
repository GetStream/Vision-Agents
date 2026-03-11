# Computer Use Example

An AI desktop assistant that can see your screen and control your computer. Share your screen in a video call and ask the agent to perform actions like opening folders, clicking buttons, typing text, or using keyboard shortcuts.

## How it works

1. You join a video call and share your screen
2. The agent receives your screen-share frames via Gemini Realtime
3. You ask the agent to do something (e.g. "open my Downloads folder")
4. The agent sees your screen, identifies what to interact with, and calls action tools
5. PyAutoGUI executes the actions on the host machine

## Prerequisites

- Python 3.10+
- A display environment (the agent controls the machine it runs on)
- API keys for:
  - [Google AI (Gemini)](https://ai.google.dev/) — for the Realtime LLM
  - [Stream](https://getstream.io/) — for video infrastructure

## Setup

1. Navigate to this example:
   ```bash
   cd examples/10_computer_use_example
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Set up your `.env`:
   ```bash
   GOOGLE_API_KEY=your_google_key
   STREAM_API_KEY=your_stream_key
   STREAM_API_SECRET=your_stream_secret
   ```

## Run

```bash
uv run computer_use_example.py run
```

The agent will create a call and open a demo UI. Share your screen in the call, then ask the agent to perform actions.

## Available actions

| Tool | What it does |
|------|-------------|
| `click(cell, position, button)` | Click at a grid cell |
| `double_click(cell, position)` | Double-click at a grid cell |
| `type_text(text)` | Type into the focused element |
| `key_press(keys)` | Press a key combo, e.g. `"cmd+c"` |
| `scroll(cell, position, clicks, direction)` | Scroll at a grid cell |
| `mouse_move(cell, position)` | Move the cursor to a grid cell |
| `open_path(path)` | Open a file or folder with the OS default handler |

## Important notes

- The agent controls the machine it runs on, not the caller's machine. For remote control, run the agent on the target machine.
- PyAutoGUI requires accessibility permissions on macOS (System Settings > Privacy & Security > Accessibility).
- Consider running in a sandboxed environment (VM or container) for safety.
