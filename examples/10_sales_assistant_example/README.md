# Sales Assistant — AI Meeting Coach
<img src="https://github.com/GetStream/vision-agents-sales-assistant-demo/blob/main/gh_assets/screenshot.png" alt="Sales Assistant Example" height="300">

A real-time AI coaching example app that provides coaching suggestions during meetings and interviews. It captures your microphone audio, sends it to an AI agent for transcription and analysis, and displays coaching suggestions on a translucent macOS overlay.

## Architecture

The project has two components:

| Component | Location | Description |
|-----------|----------|-------------|
| **Python Agent** | This directory | Vision Agents backend that joins a Stream Video call, transcribes audio with Deepgram, analyzes transcripts with Gemini, and sends coaching text back |
| **Flutter App** | [vision-agents-sales-assistant-demo](https://github.com/GetStream/vision-agents-sales-assistant-demo) | Translucent macOS overlay that captures mic+system audio via a Stream Video call and displays the agent's suggestions |

**Flow:**

1. User opens the Flutter overlay and clicks **Start**
2. The Flutter app creates a Stream Video call with screen sharing (including system audio capture)
3. The Flutter app tells the Python agent server to join the call
4. The agent transcribes audio (Deepgram STT) and generates coaching suggestions from transcripts (Gemini LLM)
5. Coaching suggestions appear as text on the translucent overlay via Stream Chat

## Prerequisites

- macOS 13.0 or later
- Python 3.12+, [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys for:
  - [Stream](https://getstream.io/try-for-free/) (Video API key + secret)
  - [Google AI Studio](https://aistudio.google.com) (Gemini API key)
  - [Deepgram](https://console.deepgram.com) (STT API key)

## Setup

### 1. Python Agent

```bash
# Copy and fill in your API keys
cp .env.example .env
# Edit .env with your keys

# Install dependencies (using uv)
uv sync

# Start the agent HTTP server
uv run main.py serve --host 0.0.0.0 --port 8000
```

The server will listen on `http://localhost:8000`. The Flutter app calls `POST /sessions` to start coaching sessions.

### 2. Flutter App

The macOS Flutter app lives in a separate repository:
**https://github.com/GetStream/vision-agents-sales-assistant-demo**

See the README there for setup and run instructions. The app expects the agent server to be running at `http://localhost:8000`.

## Usage

1. Start the Python agent server (Terminal 1):
   ```bash
   uv run main.py serve
   ```

2. Run the Flutter overlay (Terminal 2) — see the [Flutter app repo](https://github.com/GetStream/vision-agents-sales-assistant-demo) for instructions.

3. The translucent overlay window appears in the top-right corner of your screen.

4. Click **Start** to begin a coaching session. The app will:
   - Share your screen (with system audio)
   - Connect the AI agent
   - Display coaching suggestions as they arrive

5. Click **Stop** to end the session.

## Project Structure

```
sales_assistant/
├── main.py              # Agent definition + HTTP server
├── instructions.md      # System prompt for the coaching agent
├── pyproject.toml       # Python dependencies
├── .env.example         # API key template
└── README.md
```

## How It Works

### AI Pipeline

The agent uses a non-realtime STT + LLM pipeline:
- **Deepgram STT** transcribes meeting audio into text with built-in turn detection
- **Gemini LLM** analyzes transcripts and generates short coaching suggestions
- Responses are synced to a **Stream Chat** channel (`messaging:{callId}`) that the Flutter app listens to
- No TTS is needed since suggestions are displayed as text

> **Tip:** To add screen analysis, swap `gemini.LLM` for `gemini.Realtime(fps=3)`.
> Note that Realtime mode also outputs audio, so the agent would speak its suggestions aloud in addition to writing them to chat.
