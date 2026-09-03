import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent
from vision_agents.plugins import stream as acceleration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

"""
A customer support agent whose skills, knowledge and instructions live in this
directory. sync_agent pushes them to the acceleration server; a second run with
the same files does nothing.

Gemini transcribes and holds the conversation, Inworld TTS-2 Flash speaks it,
and refunds are handed to Sol.
"""


async def main() -> None:
    await acceleration.sync_agent("customer_support")

    agent = Agent(
        config="customer_support",
        llm=acceleration.Accelerated(
            config="customer_support",
            stt="gemini/gemini-3.5-transcribe-live",
            tts="inworld/inworld-tts-2-flash",
            model="gemini/gemini-3.5-flash-lite",
            subagent="openai/gpt-5.6-sol",
        ),
    )
    call = await agent.create_call("default", "customer-support")
    async with agent.join(call):
        await agent.simple_response(
            "greet the user and let them know you're a friendly AI agent"
        )
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(main())
