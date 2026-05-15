"""
AWS STT - LLM - TTS Pipeline Example

Voice agent built entirely from AWS components:
- STT: AWS Transcribe streaming
- LLM: AWS Bedrock (Qwen)
- TTS: AWS Polly
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import aws, getstream

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AWS Voice Agent", id="agent"),
        instructions="You are a voice agent. Keep replies short and "
        "conversational. Do not use special characters or formatting.",
        llm=aws.LLM(model="qwen.qwen3-32b-v1:0", region_name="us-east-1"),
        stt=aws.STT(language_code="en-US", region_name="us-east-1"),
        tts=aws.TTS(region_name="us-east-1", voice_id="Joanna", engine="neural"),
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await asyncio.sleep(5)
        await agent.simple_response(text="Ask the user about their day.")

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
