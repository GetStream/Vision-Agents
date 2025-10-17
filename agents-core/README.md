# Open Vision Agents by Stream

Build Vision Agents quickly with any model or video provider.

-  **Video AI**: Built for real-time video AI. Combine Yolo, Roboflow and others with gemini/openai realtime
-  **Low Latency**: Join quickly (500ms) and low audio/video latency (30ms)
-  **Open**: Built by Stream, but use any video edge network that you like
-  **Native APIs**: Native SDK methods from OpenAI (create response), Gemini (generate) and Claude (create message). So you're never behind on the latest features
-  **SDKs**: SDKs for React, Android, iOS, Flutter, React, React Native and Unity.

Created by Stream, uses [Stream's edge network](https://getstream.io/video/) for ultra-low latency.

## Quick Start

```bash
# Install vision-agents
uv add vision-agents

# Import (automatically creates .env file)
python -c "from vision_agents import Agent"
```

The package automatically creates a `.env` file with example configuration. Edit it to add your API keys:

### Required API Keys
- **Stream** (required): `STREAM_API_KEY`, `STREAM_API_SECRET` - [Get keys](https://getstream.io/)
- **LLM** (choose one): `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`
- **STT** (choose one): `DEEPGRAM_API_KEY`, `MOONSHINE_API_KEY`, `WIZPER_API_KEY`
- **TTS** (choose one): `CARTESIA_API_KEY`, `ELEVENLABS_API_KEY`, `KOKORO_API_KEY`
- **Turn Detection**: `FAL_KEY` - [Get key](https://fal.ai/)

### Setup Commands
```bash
vision-agents-setup              # Create .env file
vision-agents-setup --guide      # Show setup guide
vision-agents-setup --force      # Overwrite existing .env
```

## Example Usage

```python
from vision_agents import Agent
from vision_agents.plugins import openai, deepgram, cartesia

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="AI Assistant", id="agent"),
    instructions="You're a helpful AI assistant.",
    llm=openai.LLM(model="gpt-4o-mini"),
    tts=cartesia.TTS(),
    stt=deepgram.STT(),
)
```

## Documentation

- 📚 [Full Documentation](https://visionagents.ai/)
- 💬 [Examples](https://github.com/GetStream/Vision-Agents/tree/main/examples)
- 🔧 [GitHub Repository](https://github.com/GetStream/Vision-Agents)