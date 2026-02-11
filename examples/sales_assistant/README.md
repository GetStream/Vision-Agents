# Sales Assistant — AI Meeting Coach

A real-time AI coaching example app that provides coaching suggestions during meetings and interviews. It captures your microphone audio, sends it to an AI agent for transcription and analysis, and displays coaching suggestions on a translucent macOS overlay.

## Architecture

The project has two components:

| Component | Path | Description |
|-----------|------|-------------|
| **Python Agent** | `agent/` | Vision Agents backend that joins a Stream Video call, transcribes audio with Deepgram, analyzes transcripts with Gemini, and sends coaching text back |
| **Flutter App** | `app/` | Translucent macOS overlay that captures mic+system audio via a Stream Video call and displays the agent's suggestions |

**Flow:**

1. User opens the Flutter overlay and clicks **Start**
2. The Flutter app creates a Stream Video call with screen sharing (including system audio capture)
3. The Flutter app tells the Python agent server to join the call
4. The agent transcribes audio (Deepgram STT) and generates coaching suggestions from transcripts (Gemini LLM)
5. Coaching suggestions appear as text on the translucent overlay via Stream Chat

## Prerequisites

- macOS 13.0 or later
- Flutter 3.32+ / Dart 3.8+
- Python 3.12+, [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys for:
  - [Stream](https://dashboard.getstream.io) (Video API key + secret)
  - [Google AI Studio](https://aistudio.google.com) (Gemini API key)
  - [Deepgram](https://console.deepgram.com) (STT API key)

## Setup

### 1. Python Agent

```bash
cd agent

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

```bash
cd app

# Set your Stream API key and user token in lib/overlay_app.dart
# Look for the TODO comment near the top of OverlayApp

# Get dependencies
flutter pub get

# Run on macOS
flutter run -d macos
```

**Important:** You need to set your Stream API key and user token in `lib/overlay_app.dart` before running. You can generate a user token from the [Stream Dashboard](https://dashboard.getstream.io) or via the server-side SDK.

## Usage

1. Start the Python agent server (Terminal 1):
   ```bash
   cd agent && uv run main.py serve
   ```

2. Run the Flutter overlay (Terminal 2):
   ```bash
   cd app && flutter run -d macos
   ```

3. The translucent overlay window appears in the top-right corner of your screen.

4. Click **Start** to begin a coaching session. The app will:
   - Share your screen (with system audio)
   - Connect the AI agent
   - Display coaching suggestions as they arrive

5. Click **Stop** to end the session.

## Project Structure

```
sales_assistant/
├── agent/
│   ├── main.py              # Agent definition + HTTP server
│   ├── instructions.md      # System prompt for the coaching agent
│   ├── pyproject.toml       # Python dependencies
│   └── .env.example         # API key template
├── app/
│   ├── lib/
│   │   ├── main.dart           # Entry point + translucent window setup
│   │   ├── overlay_app.dart    # Stream Video initialization
│   │   ├── overlay_screen.dart # Main UI (Start/Stop + suggestion cards)
│   │   └── agent_service.dart  # HTTP client for agent server
│   ├── macos/
│   │   └── Runner/
│   │       ├── MainFlutterWindow.swift  # Translucent NSWindow config
│   │       ├── DebugProfile.entitlements
│   │       └── Release.entitlements
│   └── pubspec.yaml
└── README.md
```

## How It Works

### Translucent Window

The macOS overlay uses `NSVisualEffectView` with `.hudWindow` material for the frosted-glass blur effect. The window is configured as:
- Always-on-top (floating window level)
- Transparent background with blur
- Rounded corners
- No titlebar buttons
- Compact size (420x640), positioned in the top-right corner
- **Hidden from screen capture** — the window sets `NSWindow.sharingType = .none`, so it is invisible to Zoom screen share, OBS, QuickTime, and any other screen-recording tool. Coaching suggestions stay private.

### Screen Audio Capture

The Flutter app uses the `feature/macos-screen-audio-capture` branch of `stream_webrtc_flutter` which adds ScreenCaptureKit-based system audio capture on macOS. This means the AI agent can hear both your microphone and any audio from other apps (like a Zoom call).

### AI Pipeline

The agent uses a non-realtime STT + LLM pipeline:
- **Deepgram STT** transcribes meeting audio into text with built-in turn detection
- **Gemini LLM** analyzes transcripts and generates short coaching suggestions
- Responses are synced to a **Stream Chat** channel (`messaging:{callId}`) that the Flutter app listens to
- No TTS is needed since suggestions are displayed as text

> **Tip:** To add screen analysis, swap `gemini.LLM` for `gemini.Realtime(fps=3)`.
> Note that Realtime mode also outputs audio, so the agent would speak its suggestions aloud in addition to writing them to chat.
