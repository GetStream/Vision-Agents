"""
Grok TTS — Restaurant Host Example

A voice agent that acts as a warm, welcoming restaurant host.
Uses the 'ara' voice (warm, friendly) for a conversational hospitality feel.

Requirements (environment variables):
    XAI_API_KEY          — xAI / Grok API key
    DEEPGRAM_API_KEY     — Deepgram STT key
    GOOGLE_API_KEY       — Google Gemini key
    STREAM_API_KEY       — Stream API key
    STREAM_API_SECRET    — Stream API secret
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, gemini, getstream, smart_turn
from vision_agents.plugins import grok_tts

logger = logging.getLogger(__name__)

load_dotenv()

RESTAURANT_HOST_INSTRUCTIONS = """\
You are Sofia, the host at "Bella Sera," an upscale Italian restaurant.

Your personality:
- Warm, gracious, and genuinely delighted to welcome every guest
- Knowledgeable about the menu, wine pairings, and the chef's specialties
- Attentive to dietary restrictions and allergies without being asked twice

Your responsibilities:
- Greet callers warmly and help them make or modify reservations
- Describe tonight's specials with enthusiasm and vivid detail
- Recommend dishes based on guest preferences (vegetarian, gluten-free, etc.)
- Manage waitlist estimates during busy hours
- Confirm reservation details (date, time, party size, special requests)

Tone guidelines:
- Use expressive language: "Wonderful choice!", "You're going to love it!"
- Keep responses concise — diners are often calling on the go
- If unsure about availability, say you'll check with the floor manager
  rather than guessing

Sample menu highlights you may reference:
- Truffle risotto, house-made pappardelle, branzino al limone
- Chef's tasting menu (5 courses, optional wine pairing)
- Tiramisu and panna cotta for dessert
"""


async def create_agent(**kwargs) -> Agent:
    """Create a restaurant host agent with Grok TTS (ara voice)."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Sofia - Bella Sera Host", id="agent"),
        instructions=RESTAURANT_HOST_INSTRUCTIONS,
        tts=grok_tts.TTS(voice="ara"),
        stt=deepgram.STT(eager_turn_detection=True),
        llm=gemini.LLM(),
        turn_detection=smart_turn.TurnDetection(
            silence_duration_ms=2000,
            speech_probability_threshold=0.5,
        ),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and greet the caller as a restaurant host."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Restaurant Host Agent...")

    async with agent.join(call):
        logger.info("Agent joined call")

        await asyncio.sleep(3)
        await agent.llm.simple_response(
            text=(
                "Good evening! Thank you for calling Bella Sera. "
                "I'm Sofia, your host. How may I help you tonight — "
                "would you like to make a reservation or hear about our specials?"
            )
        )

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
