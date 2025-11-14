# TTS Setup Guide

## Quick Start: ElevenLabs (Recommended)

1. **Get an API key** from [ElevenLabs](https://elevenlabs.io/)
2. **Add to your `.env` file**:
   ```bash
   ELEVENLABS_API_KEY=your_api_key_here
   ```
3. **Install dependencies** (if not already done):
   ```bash
   cd examples/other_examples/get-in-shAIpe
   uv sync
   ```
4. **Run your agent** - TTS is now enabled! 🎉

## Alternative TTS Providers

### Cartesia (Very Low Latency)

1. **Get an API key** from [Cartesia](https://cartesia.ai/)
2. **Update `pyproject.toml`**:
   ```toml
   "vision-agents-plugins-cartesia" = { workspace = true }
   ```
3. **Update `main.py`**:
   ```python
   from vision_agents.plugins import cartesia
   # ...
   tts=cartesia.TTS(),
   ```
4. **Add to `.env`**:
   ```bash
   CARTESIA_API_KEY=your_api_key_here
   ```

### Kokoro (No API Key Required)

1. **Install espeak-ng** (required at runtime):
   ```bash
   brew install espeak-ng  # macOS
   # or
   sudo apt-get install espeak-ng  # Linux
   ```
2. **Update `pyproject.toml`**:
   ```toml
   "vision-agents-plugins-kokoro" = { workspace = true }
   ```
3. **Update `main.py`**:
   ```python
   from vision_agents.plugins import kokoro
   # ...
   tts=kokoro.TTS(),
   ```
4. **No API key needed!** 🎉

## How It Works

- `agent.say(message)` will now use the TTS provider for fast voice output
- Latency: ~100-300ms (vs 1-3 seconds with `agent.llm.simple_response()`)
- The TTS provider automatically handles audio synthesis and streaming

## Troubleshooting

**"No TTS available" warning:**
- Make sure you've added the TTS provider to `pyproject.toml`
- Run `uv sync` to install dependencies
- Check that your API key is set in `.env` (for ElevenLabs/Cartesia)
- Verify the import statement matches the provider you're using

**API Key errors:**
- Ensure your `.env` file is in the same directory as `main.py`
- Check that `load_dotenv()` is called before creating the agent
- Verify the API key is valid and has sufficient credits

