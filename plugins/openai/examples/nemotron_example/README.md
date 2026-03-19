# Fraud Detection with NVIDIA Nemotron

https://github.com/user-attachments/assets/d2ddfa95-aab2-4b7a-9be1-13e447ee0b34

A voice-powered fraud detection agent built with [Vision Agents](https://github.com/GetStream/Vision-Agents) and NVIDIA's Nemotron Super 3 LLM hosted on Baseten.

The agent joins a real-time voice call and helps customers investigate suspicious transactions, freeze compromised cards, cancel fraudulent charges, and issue replacements — all through natural conversation.

It uses function calls that send events instead of actually blocking a card etc. The agent is empowered to do this itself.

See the [demo on X](https://x.com/visionagents_ai/status/2032548219566141793).

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Create a `.env` file with your API keys:

```
BASETEN_API_KEY=...
STREAM_API_KEY=...
STREAM_API_SECRET=...
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
```

3. Run the agent:

```bash
uv run python nemotron_example.py
```
