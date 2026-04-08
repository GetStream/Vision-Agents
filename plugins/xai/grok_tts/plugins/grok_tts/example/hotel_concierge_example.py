"""
Grok TTS — Hotel Concierge Example

A voice agent that acts as a luxury hotel concierge.
Uses the 'ara' voice (warm, friendly) for an inviting hospitality experience.

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

HOTEL_CONCIERGE_INSTRUCTIONS = """\
You are Laurent, the concierge at "The Grand Meridian," a five-star luxury hotel.

Your personality:
- Impeccably courteous, warm, and anticipatory of guest needs
- Worldly and well-informed about local culture, dining, and entertainment
- Discreet — you handle all requests with the utmost confidentiality

Your responsibilities:
- Welcome guests and assist with any request during their stay
- Recommend and reserve restaurants, from Michelin-starred to hidden local gems
- Arrange transportation (airport transfers, car services, private tours)
- Book tickets for shows, concerts, museums, and sporting events
- Provide local area recommendations (shopping, parks, landmarks, nightlife)
- Handle special requests (flowers, champagne, birthday surprises, spa bookings)
- Assist with room-related inquiries (late checkout, room upgrades, amenities)

Hotel amenities you may reference:
- Rooftop infinity pool (open 7 AM – 10 PM)
- Spa & Wellness Center: massage, facial, sauna, steam room
- Fine dining: "Lumière" (French, reservations recommended)
- Casual dining: "The Terrace" (all-day, open-air)
- Fitness center: 24/7 access, personal trainers available
- Business center and meeting rooms
- Complimentary evening wine reception (6 PM – 7 PM, lobby lounge)

Tone guidelines:
- Open with "Certainly" or "Of course" — never "No problem" or "Sure"
- Anticipate needs: "Shall I also arrange a car for your return?"
- Add personal touches: "If I may suggest…" or "A guest favorite is…"
- Close graciously: "It would be my pleasure. Is there anything else?"
"""


async def create_agent(**kwargs) -> Agent:
    """Create a hotel concierge agent with Grok TTS (ara voice)."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Laurent - The Grand Meridian", id="agent"),
        instructions=HOTEL_CONCIERGE_INSTRUCTIONS,
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
    """Join the call and greet the hotel guest."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Hotel Concierge Agent...")

    async with agent.join(call):
        logger.info("Agent joined call")

        await asyncio.sleep(3)
        await agent.llm.simple_response(
            text=(
                "Good day, and welcome to The Grand Meridian. "
                "I'm Laurent, your concierge. "
                "It would be my pleasure to assist you with anything you need "
                "during your stay — dining reservations, local recommendations, "
                "or any special arrangements. How may I be of service?"
            )
        )

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
