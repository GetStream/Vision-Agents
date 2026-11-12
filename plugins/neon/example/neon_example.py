"""
Neon AI Gateway LLM Example

Set these environment variables before running:
- NEON_AI_GATEWAY_TOKEN: Neon credential with the ai_gateway:invoke scope
- NEON_AI_GATEWAY_BASE_URL: Your branch's AI Gateway host
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, elevenlabs, getstream, neon

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with a Neon AI Gateway LLM."""
    llm = neon.LLM(model="gpt-5-mini")

    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str):
        """Get the current weather for a location."""
        return {
            "location": location,
            "temperature": "22 degrees celsius",
            "condition": "Sunny",
        }

    return Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Neon AI", id="agent"),
        instructions="You are a helpful voice assistant. Keep replies short.",
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
    )


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.simple_response("Say hi to the user")
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
