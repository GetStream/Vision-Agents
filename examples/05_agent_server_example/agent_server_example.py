import logging
from typing import Any, Dict

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import deepgram, elevenlabs, gemini, getstream

logger = logging.getLogger(__name__)

load_dotenv()

"""
Agent example optimized for fast response time.

Eager turn taking STT, LLM, TTS workflow
- deepgram for optimal latency
- eleven labs for TTS
- gemini-2.5-flash-lite for fast responses
- stream's edge network for video transport

This example uses STT, for a realtime openAI/gemini example see 02_golf_coach_example
"""


async def create_agent(**kwargs) -> Agent:
    llm = gemini.LLM("gemini-2.5-flash-lite")

    agent = Agent(
        edge=getstream.Edge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=User(name="My happy AI friend", id="agent"),
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(
            eager_turn_detection=True
        ),  # eager_turn_detection -> lower latency (but higher token usage)
        # turn_detection=vogent.TurnDetection(), # smart turn and vogent are supported. not needed with deepgram (it has turn keeping)
        # realtime openai and gemini are supported (tts and stt not needed in that case)
        # llm=openai.Realtime()
    )

    # MCP and function calling are supported. see https://visionagents.ai/guides/mcp-tool-calling
    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> Dict[str, Any]:
        return await get_weather_by_location(location)

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    # Have the agent join the call/room
    async with agent.join(call):
        # Use agent.simple response or...
        await agent.simple_response("tell me something interesting in a short sentence")

        # run till the call ends
        await agent.finish()


if __name__ == "__main__":
    Runner(
        AgentLauncher(create_agent=create_agent, join_call=join_call),
    ).cli()
