# 🏏 Third Umpire AI — Women's Cricket Decision Agent

An AI-powered Third Umpire built with [Vision Agents](https://visionagents.ai) that watches live cricket video and makes real-time decisions on disputed events — run outs, stumpings, boundary catches, and more.

Built for the **Vision AI Hackathon** (Feb 23 – Mar 1, 2026).

## What It Does

- 🎥 Watches live or recorded women's cricket video in real-time
- 🤖 Uses YOLO object detection to track players, ball, and stumps
- 🧠 Uses Gemini Live to reason about the scene and make decisions
- 📢 Announces verdicts in an official third umpire style (voice + text)
- ⚡ Runs with sub-30ms latency via Stream's edge network

## Decisions Supported

| Scenario | Decision |
|---|---|
| Run Out | OUT / NOT OUT |
| Stumping | OUT / NOT OUT |
| Boundary Catch | OUT / SIX |
| Clean Catch | OUT / NOT OUT |
| LBW (basic) | Directional analysis |

## Tech Stack

- **Vision Agents SDK** — core framework
- **YOLO (Ultralytics)** — real-time object detection
- **Gemini Live** — real-time multimodal LLM
- **Stream Edge Network** — ultra-low latency video (<30ms)

## Setup

### Prerequisites
- Python 3.12+
- Stream account → [getstream.io/try-for-free](https://getstream.io/try-for-free)
- Google Gemini API key → [aistudio.google.com](https://aistudio.google.com)

### Installation

```bash
# Clone the repo
git clone https://github.com/jaya6400/Vision-Agents.git
cd Vision-Agents

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# or: venv\Scripts\activate   # Windows CMD

# Install SDK
pip install -e agents-core

# Install dependencies
pip install vision-agents[gemini,ultralytics,getstream] python-dotenv opencv-python
```

### Configuration

Create a `.env` file in the root of the project:

```env
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
GOOGLE_API_KEY=your_gemini_api_key
```

### Run

```bash
cd examples/09_cricket_umpire
python cricket_umpire.py
```

The agent will:
1. Create a video call session
2. Open a browser UI
3. Join and start watching the video feed
4. Analyze cricket scenarios in real-time
5. Announce decisions via voice and text

## How It Works

```
Live Cricket Video
      ↓
YOLO Object Detection
(players, stumps, ball positions)
      ↓
Gemini Live Analysis
(scene understanding + reasoning)
      ↓
Third Umpire Decision
(OUT / NOT OUT + explanation)
```

## Project Structure

```
09_cricket_umpire/
├── cricket_umpire.py     # Main agent code
├── cricket_umpire.md     # Agent instructions (the "brain")
├── pyproject.toml        # Dependencies
└── README.md             # This file
```

## Built By
Jaya — Vision AI Hackathon 2026