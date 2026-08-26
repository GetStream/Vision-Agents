"""Voice bot running its whole pipeline on Telnyx.

Unlike the phone examples in this folder, this one needs no Telnyx phone number,
ngrok, or Call Control App. It joins a Stream video call and answers with a
pipeline that runs entirely on Telnyx: `telnyx.STT`, `telnyx.LLM`, and
`telnyx.TTS`, with `smart_turn` for turn detection since Telnyx STT does not
emit VAD signals.

Run it, join the call in your browser, and speak to the bot.

Usage::

    uv run plugins/telnyx/examples/voice_bot.py run

The script looks for the following env vars (see `.env.example`):
    STREAM_API_KEY / STREAM_API_SECRET
    TELNYX_API_KEY
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Runner
from vision_agents.core.agents import Agent, AgentLauncher
from vision_agents.core.edge.types import User
from vision_agents.plugins import getstream, smart_turn, telnyx

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create an agent that runs STT, LLM, and TTS on Telnyx."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Telnyx Voice Bot", id="agent"),
        instructions=(
            "You're a helpful voice AI assistant. "
            "Keep replies short and conversational."
        ),
        stt=telnyx.STT(),
        llm=telnyx.LLM(),
        tts=telnyx.TTS(),
        turn_detection=smart_turn.TurnDetection(),
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Telnyx voice bot")

    async with agent.join(call):
        logger.info("Joined call")
        await asyncio.sleep(3)
        await agent.simple_response(
            "Hello! I'm listening. What would you like to talk about?"
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
