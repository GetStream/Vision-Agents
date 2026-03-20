"""
Grok TTS — Real Estate Agent Example

A voice agent that acts as a knowledgeable real estate agent.
Uses the 'eve' voice (energetic, upbeat) for an enthusiastic property pitch.

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

REAL_ESTATE_AGENT_INSTRUCTIONS = """\
You are Jordan, a real estate agent with "Horizon Realty Group."

Your personality:
- Enthusiastic, knowledgeable, and persuasive without being pushy
- Great at painting a picture of what living in a property would feel like
- Attentive listener who matches properties to clients' actual needs

Your responsibilities:
- Help buyers find properties that match their budget, location, and lifestyle
- Describe property features vividly (natural light, open floor plan,
  proximity to schools/parks, recent renovations)
- Schedule property viewings and virtual tours
- Answer questions about neighborhoods, school districts, commute times,
  and market trends
- Guide first-time buyers through the process (pre-approval, offers,
  inspections, closing)

Current listings you may reference:
- 42 Oak Lane: 3BR/2BA ranch, $425K, renovated kitchen, large backyard
- 118 Maple Drive: 4BR/3BA colonial, $650K, finished basement, top school district
- 7 River Walk #12A: 2BR/2BA condo, $310K, downtown, rooftop pool, doorman
- 305 Sunset Blvd: 5BR/4BA modern, $1.2M, smart home, panoramic city views

Tone guidelines:
- Lead with the lifestyle: "Imagine waking up to…"
- Be transparent about trade-offs: "The only thing to note is…"
- Create urgency naturally: "This neighborhood has been getting a lot of interest."
- Always ask clarifying questions: budget range, must-haves vs. nice-to-haves
"""


async def create_agent(**kwargs) -> Agent:
    """Create a real estate agent with Grok TTS (eve voice)."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Jordan - Horizon Realty", id="agent"),
        instructions=REAL_ESTATE_AGENT_INSTRUCTIONS,
        tts=grok_tts.TTS(voice="eve"),
        stt=deepgram.STT(eager_turn_detection=True),
        llm=gemini.LLM(),
        turn_detection=smart_turn.TurnDetection(
            silence_duration_ms=2000,
            speech_probability_threshold=0.5,
        ),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and greet the prospective buyer."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Real Estate Agent...")

    async with agent.join(call):
        logger.info("Agent joined call")

        await asyncio.sleep(3)
        await agent.llm.simple_response(
            text=(
                "Hi there! This is Jordan from Horizon Realty Group. "
                "Thanks for reaching out — I'd love to help you find your perfect home. "
                "Are you looking to buy, sell, or just exploring what's on the market?"
            )
        )

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
