## Stream + Gemini Live Multimodal Example

This example shows how to connect a Google Gemini Live speech-to-speech agent to a Stream video call for real-time voice conversations. It has turn detection and interruption enabled so you can interrupt the bot as it speaks. It also sends video frames from your webcam to Gemini Live so it can describe things it can see.

### What It Does

- AI agent: joins a Stream video call as a bot user
- Browser UI: opens a link so you can join the same call
- Live audio: streams your microphone audio to Gemini
- Live video: sends video frames from your webcam to Gemini at 1FPS
- Assistant speech: plays Gemini’s synthesized audio into the call
- Model: uses a Gemini Live model with voice output

### Prerequisites

1. **Stream account**: Get API credentials from [Stream Dashboard](https://dashboard.getstream.io)
2. **Google account**: Get a `GOOGLE_API_KEY` with Gemini Live access
3. **Python 3.10+**: Required to run the example

### Installation

You can use your preferred package manager, but we recommend `uv`.

1. **Navigate to this directory:**
   ```bash
   cd examples/gemini_live
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Set up environment variables:**
   - Copy `env.example` to `.env`
   - Set at least the following:
     - `STREAM_API_KEY`: your Stream key
     - `STREAM_API_SECRET`: your Stream secret
     - `GOOGLE_API_KEY`: your Gemini API key
     - `EXAMPLE_BASE_URL`: base URL for the example UI

### Usage

Run the example:
```bash
uv run main.py
```

The script creates a user and a bot user, starts a call, and opens a browser link. Join the call in the browser, then speak; the agent listens and responds with synthesized speech. Ask it about something in your environment!
