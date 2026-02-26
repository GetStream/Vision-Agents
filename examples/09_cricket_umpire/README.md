# 🏏 Cricket DRS AI — Third Umpire Decision Review System

> AI-powered Decision Review System for Women's Cricket using Gemini Live vision, YOLO pose detection, and real-time voice verdicts.

---

## 🎥 Demo Video

[▶ Watch Demo](YOUR_DEMO_VIDEO_LINK_HERE)

## 🌐 Deployment

| | Link |
|---|---|
| Frontend | [YOUR_NETLIFY_LINK_HERE](YOUR_NETLIFY_LINK_HERE) |
| Backend | Runs locally (see setup below) |
| GitHub | [github.com/jaya6400/Vision-Agents](https://github.com/jaya6400/Vision-Agents) |

> Backend requires a persistent Gemini Live WebSocket connection and cannot be hosted on free-tier platforms. Full local setup takes under 2 minutes.

---

## ✨ Features

- **Real-time video analysis** — Gemini Live watches your screen share and analyzes cricket footage frame by frame
- **Voice verdicts** — Third Umpire AI speaks the decision aloud (DECISION / REVIEW TYPE / REASON / CONFIDENCE)
- **YOLO pose detection** — Player body positions detected in real-time at 30 FPS
- **Two review types** — LBW and Run Out (the two most contested DRS decisions)
- **FastAPI trigger endpoint** — Button click sends review request directly to Gemini via REST API
- **Custom DRS UI** — Built with Stream Video SDK, dark cricket stadium aesthetic

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Vision AI | Gemini Live (google-genai) — real-time video + audio |
| Pose Detection | YOLO11n-pose (Ultralytics) — player skeleton tracking |
| Video Transport | Stream Video SDK (getstream) |
| Agent Framework | Vision Agents SDK (GetStream) |
| Backend API | FastAPI + Uvicorn |
| Frontend | React + Vite + Stream Video React SDK |
| Auth | JWT token server |

---

## 🚀 Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Google API Key (Gemini Live access)
- Stream API Key + Secret

### 1. Clone & Install

```bash
git clone https://github.com/jaya6400/Vision-Agents.git
cd Vision-Agents
pip install -e agents-core
pip install -e plugins/getstream
pip install -e plugins/ultralytics
cd examples/09_cricket_umpire
pip install -r requirements.txt
```

### 2. Environment Variables

Create `.env` in `examples/09_cricket_umpire/`:

```env
GOOGLE_API_KEY=your_google_api_key
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
```

Create `.env` in `examples/09_cricket_umpire/frontend/`:

```env
VITE_STREAM_API_KEY=your_stream_api_key
```

### 3. Run

```bash
cd examples/09_cricket_umpire
bash run.sh
```

This starts:
- Token server on `http://localhost:8001`
- DRS Agent (Gemini Live + YOLO)
- Review API on `http://localhost:8002`
- Frontend on `http://localhost:5173`

### 4. Usage

1. Open `http://localhost:5173`
2. Click **Start DRS Session**
3. Click **Share Screen** → select your cricket video tab
4. **Uncheck "Share tab audio"** in the Chrome dialog (important!)
5. Click **LBW Review** or **Run Out Review**
6. Hear the Third Umpire AI speak the verdict

---

## 🗂 Project Structure

```
examples/09_cricket_umpire/
├── cricket_umpire.py      # Main agent + FastAPI review endpoint
├── cricket_umpire.md      # Gemini instructions (DRS rules)
├── token_server.py        # JWT auth server
├── run.sh                 # One-command startup
├── requirements.txt       # Python dependencies
└── frontend/
    └── src/
        ├── App.jsx        # React UI
        └── App.css        # DRS styling
```

---

## 📸 Screenshots

<!-- Add your screenshots here -->
> _Add screenshots of the DRS UI, YOLO pose detection, and decision log_

---

## ⚠️ Known Issues & Fixes

| Issue | Root Cause | Fix Applied |
|---|---|---|
| `AudioQueue buffer limit exceeded` | `SCREEN_SHARE_AUDIO` track overwhelming WebRTC pipeline | Uncheck "Share tab audio" in Chrome screen share dialog |
| `Pose processing TIMEOUT 12s` | YOLO running at full 1920×1080 resolution on CPU | Reduced `imgsz=256`, dropped Gemini fps to 2 |
| `Edge connection is not set` | Screen share arrived before agent WebRTC fully connected | Agent joins before user, race condition handled by SDK retry |
| `Cannot handle offer in signaling state "closed"` | WebRTC renegotiation after audio track timeout | Resolved by disabling screen share audio |
| Agent giving verdict before screen share | Gemini responding to text prompt alone without video | Added video grounding check in instructions |

---

## 🏗 Architecture

```
┌─────────────────┐     screen share      ┌──────────────────────┐
│   Browser UI    │ ──────────────────────▶│  Stream Video SFU    │
│  (React + Vite) │                        └──────────┬───────────┘
│                 │                                   │ video frames
│  Click Review   │     POST /review/lbw              ▼
│  Button         │ ──────────────────────▶ ┌─────────────────────┐
└─────────────────┘                         │  cricket_umpire.py  │
                                            │                     │
                                            │  YOLO Pose (256px)  │
                                            │       ↓             │
                                            │  Gemini Live        │
                                            │  (fps=2, vision)    │
                                            │       ↓             │
                                            │  Speaks verdict     │
                                            │  via Stream audio   │
                                            └─────────────────────┘
```

---

## 📝 Blog Post

[Read the full writeup on Medium](YOUR_MEDIUM_LINK_HERE)

---

## 🏆 Built For

Vision AI Hackathon 2026 — GetStream Vision Agents Challenge

*Women's Cricket • Decision Review System • Real-time AI Umpire*