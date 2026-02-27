"""
OpenRouter VLM Example

This example demonstrates how to use OpenRouter as a provider for a Vision Language
Model agent. Since OpenRouter is OpenAI-compatible, we use the OpenAI plugin's
ChatCompletionsVLM pointed at OpenRouter's API.

Set these environment variables before running:
- OPENROUTER_API_KEY: Your OpenRouter API key
"""

import asyncio
import os

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, elevenlabs, getstream, openai
from vision_agents.plugins.getstream import CallSessionParticipantJoinedEvent

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    # OpenRouter is OpenAI-compatible, so we use ChatCompletionsVLM directly
    llm = openai.ChatCompletionsVLM(
        model="qwen/qwen-vl-plus",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        frame_buffer_seconds=3,
        frame_width=512,
        frame_height=384,
    )
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Video Assistant", id="agent"),
        instructions="""
        - Your job is to roast the person in the video.
        - You must use the video. Be funny, concise and mean, meaner than you think you can get away with.
        - Analyze the video frames and respond to user questions about what you see to use in your roast.
        - Keep responses to one sentence. Be concise and direct.
        """,
        llm=llm,
        stt=deepgram.STT(),
        tts=elevenlabs.TTS(),
        processors=[],
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        if event.participant.user.id != "agent":
            await asyncio.sleep(2)
            # await agent.simple_response("Roast the person you see")

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
