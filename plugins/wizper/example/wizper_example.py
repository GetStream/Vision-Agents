"""
Wizper STT Example

This example demonstrates Wizper STT integration with Vision Agents.

Wizper is a speech-to-text service provided by FAL.ai with translation capabilities.

This example creates an agent that uses:
- Wizper for speech-to-text (STT) with optional translation
- OpenAI for LLM
- GetStream for edge/real-time communication

Requirements:
- FAL_KEY environment variable (from fal.ai)
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- OPENAI_API_KEY environment variable
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.core.stt.events import STTErrorEvent, STTTranscriptEvent
from vision_agents.plugins import getstream, openai, wizper

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with Wizper STT."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Transcription Bot", id="transcription-bot"),
        instructions="I transcribe speech and can translate it to other languages.",
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=wizper.STT(target_language="fr"),  # Translate to French
    )

    @agent.subscribe
    async def handle_transcript(event: STTTranscriptEvent):
        user_info = "unknown"
        if event.participant:
            user = event.participant
            user_info = user.name if user.name else str(user)

        logger.info(f"[{event.timestamp}] {user_info}: {event.text}")
        if event.confidence:
            logger.info(f"    confidence: {event.confidence:.2%}")
        if event.processing_time_ms:
            logger.info(f"    processing time: {event.processing_time_ms:.1f}ms")

    @agent.subscribe
    async def handle_stt_error(event: STTErrorEvent):
        logger.error(f"STT Error: {event.error_message}")
        if event.context:
            logger.error(f"    context: {event.context}")

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start transcription."""
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        logger.info("Listening for audio...")
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
