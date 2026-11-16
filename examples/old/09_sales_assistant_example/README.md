# Sales Assistant — AI Meeting Copilot

<p align="center">
<img src="https://github.com/GetStream/vision-agents-sales-assistant-demo/blob/main/gh_assets/screenshot.png" alt="Sales Assistant Example" height="300">
</p>

A real-time AI copilot that silently listens to your microphone and system audio during meetings, interviews, and sales calls. It transcribes the conversation with speaker diarization, analyzes the dialogue, and surfaces coaching suggestions on a translucent macOS overlay — invisible to other participants.

The agent can be extended with RAG and custom knowledge bases to tailor suggestions to your product, company playbook, or deal context.

## Architecture

The project has two components:

| Component | Location | Description |
|-----------|----------|-------------|
| **Python Agent** | This directory | Vision Agents backend that joins a Stream Video call, transcribes audio with AssemblyAI (with diarization), analyzes transcripts with Gemini, and sends coaching text back |
| **macOS App** | [vision-agents-sales-assistant-demo](https://github.com/GetStream/vision-agents-sales-assistant-demo) | Translucent macOS overlay that captures mic + system audio via a Stream Video call and displays the agent's suggestions |

**Flow:**

1. User opens the macOS overlay and clicks **Start**
2. The app creates a Stream Video call with screen sharing (including system audio capture)
3. The app tells the Python agent server to join the call
4. The agent transcribes audio (AssemblyAI STT with diarization) and generates coaching suggestions (Gemini LLM)
5. Coaching suggestions appear as text on the translucent overlay via Stream Chat

## Prerequisites

- macOS 13.0 or later
- Python 3.12+, [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys for:
  - [Stream](https://getstream.io/try-for-free/?utm_source=github.com&utm_medium=referral&utm_campaign=vision_agents) (Video API key + secret)
  - [Google AI Studio](https://aistudio.google.com) (Gemini API key)
  - [AssemblyAI](https://www.assemblyai.com/) (STT API key)

## Setup

### 1. Python Agent

```bash
# Copy and fill in your API keys
cp .env.example .env
# Edit .env with your keys

# Install dependencies (using uv)
uv sync

# Start the agent HTTP server
uv run main.py serve
```

The server will listen on `http://localhost:8000`. The macOS app calls `POST /sessions` to start coaching sessions.

### 2. Flutter App

The companion macOS app lives in a separate repository:
**https://github.com/GetStream/vision-agents-sales-assistant-demo**

See the README there for build and run instructions. The app expects the agent server to be running at `http://localhost:8000`.

## Usage

1. Start the Python agent server (Terminal 1):
   ```bash
   uv run main.py serve
   ```
2. Run the macOS overlay — see the [companion app repo](https://github.com/GetStream/vision-agents-sales-assistant-demo) for instructions.

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
- **AssemblyAI STT** transcribes meeting audio into text with speaker diarisation
- **Gemini LLM** analyzes transcripts and generates short coaching suggestions
- Responses are synced to a **Stream Chat** channel (`messaging:{callId}`) that the Flutter app listens to
- No TTS is needed since suggestions are displayed as text

> **Tip:** To add screen analysis, swap `gemini.LLM` for `gemini.Realtime(fps=3)`.
> Note that Realtime mode also outputs audio, so the agent would speak its suggestions aloud in addition to writing them to chat.
