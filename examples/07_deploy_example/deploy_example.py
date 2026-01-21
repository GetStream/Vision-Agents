import logging
from typing import Any, Dict

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import deepgram, elevenlabs, gemini, getstream

logger = logging.getLogger(__name__)

load_dotenv()

"""
Deploy example - similar to 01_simple_agent_example but containerized.

Eager turn taking STT, LLM, TTS workflow
- deepgram for optimal latency
- eleven labs for TTS
- gemini-2.5-flash-lite for fast responses
- stream's edge network for video transport
"""


async def create_agent(**kwargs) -> Agent:
    llm = gemini.LLM("gemini-2.5-flash-lite")

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="My happy AI friend", id="agent"),
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        processors=[],
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> Dict[str, Any]:
        return await get_weather_by_location(location)

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.simple_response("tell me something interesting in a short sentence")
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
