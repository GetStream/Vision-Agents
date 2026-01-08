# Nemotron STT Example

Example demonstrating NVIDIA Nemotron Speech-to-Text with Vision Agents.

## Architecture

Due to dependency conflicts between NeMo toolkit and other Vision Agents dependencies,
Nemotron runs as a **separate server**:

```
┌─────────────────┐         HTTP         ┌─────────────────┐
│  Vision Agents  │ ──────────────────▶  │ Nemotron Server │
│  (this example) │                      │  (NeMo toolkit) │
└─────────────────┘                      └─────────────────┘
```

## Setup

### 1. Start the Nemotron Server

In a separate terminal/environment with NeMo installed:

```bash
cd plugins/nemotron/server

# Create a separate virtual environment for NeMo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the server
python nemotron_server.py
```

Or use Docker:

```bash
cd plugins/nemotron/server
docker build -t nemotron-server .
docker run -p 8765:8765 nemotron-server
```

### 2. Run the Example

```bash
cd plugins/nemotron/example

# Sync dependencies (this installs vision-agents and plugins)
uv sync

# Create .env with your API keys
cat > .env << EOF
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
GOOGLE_API_KEY=your_google_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
EOF

# Run the example
uv run python main.py
```

**Note:** If you get `ModuleNotFoundError`, make sure you've run `uv sync` from the example directory first. The example uses editable dependencies that point to the workspace.

## Configuration

```python
# Connect to a remote server
stt = nemotron.STT(server_url="http://your-server:8765")

# Custom timeout
stt = nemotron.STT(timeout=60.0)
```

## Server Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEMOTRON_DEVICE` | `cpu` | Device: cpu or cuda |
| `NEMOTRON_MODEL` | `nvidia/nemotron-speech-streaming-en-0.6b` | Model name |

## Links

- [Nemotron Speech on HuggingFace](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- [Vision Agents Documentation](https://visionagents.ai)
