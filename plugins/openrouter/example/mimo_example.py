"""
Xiaomi MiMo-V2-Omni Example

Demonstrates using Xiaomi's MiMo-V2-Omni model via OpenRouter as a video-capable
assistant. MiMo-V2-Omni natively processes image, video, and audio inputs within
a unified architecture.

Since OpenRouter is OpenAI-compatible, we use the OpenAI plugin's
ChatCompletionsVLM pointed at OpenRouter's API.

Set these environment variables before running:
- OPENROUTER_API_KEY: Your OpenRouter API key
- STREAM_API_KEY: Your Stream API key
- STREAM_API_SECRET: Your Stream API secret
- DEEPGRAM_API_KEY: Your Deepgram API key
- ELEVENLABS_API_KEY: Your ElevenLabs API key
"""

import os

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, elevenlabs, getstream, openai

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create a video assistant powered by Xiaomi MiMo-V2-Omni."""
    llm = openai.ChatCompletionsVLM(
        model="xiaomi/mimo-v2-omni",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        frame_buffer_seconds=3,
        frame_width=512,
        frame_height=384,
    )
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="MiMo Video Assistant", id="agent"),
        instructions=(
            "You are a helpful video assistant powered by Xiaomi MiMo-V2-Omni. "
            "You can see the user's video feed and answer questions about what you see. "
            "Keep responses concise — one or two sentences max. "
            "Describe what you observe when asked, and be direct."
        ),
        llm=llm,
        stt=deepgram.STT(),
        tts=elevenlabs.TTS(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and run until it ends."""
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
