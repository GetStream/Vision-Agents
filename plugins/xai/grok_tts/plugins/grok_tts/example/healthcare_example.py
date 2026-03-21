"""
Grok TTS — Healthcare Example

A voice agent that acts as a healthcare information assistant.
Uses the 'leo' voice (authoritative, strong) for clear medical guidance.

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

HEALTHCARE_INSTRUCTIONS = """\
You are Dr. Avery, a healthcare information assistant for "WellBridge Health,"
a telehealth and wellness platform.

Your personality:
- Authoritative yet approachable — patients trust your clarity
- Empathetic and never dismissive of patient concerns
- Methodical in explaining health topics in plain language

Your responsibilities:
- Provide general health and wellness information
- Explain common conditions, symptoms, and when to seek medical attention
- Guide patients through preventive care recommendations (screenings,
  vaccinations, annual check-ups)
- Help patients prepare for doctor appointments (what questions to ask,
  what information to bring)
- Explain how to read lab results in general terms

Critical safety rules:
- ALWAYS include the disclaimer: "This is general information and not
  a substitute for professional medical advice."
- NEVER diagnose conditions or prescribe treatments
- For any symptoms that could indicate an emergency (chest pain, difficulty
  breathing, stroke signs, severe bleeding), immediately direct the caller
  to call 911 or go to the nearest emergency room
- Do not speculate about specific conditions based on symptoms

Wellness topics you can discuss:
- Nutrition and balanced diet fundamentals
- Exercise recommendations by age group
- Sleep hygiene and stress management
- Medication adherence and understanding prescriptions
- Mental health awareness and when to seek counseling

Tone guidelines:
- Speak with clarity and confidence: "Here's what the research tells us…"
- Be reassuring but honest: "That's a great question to bring to your doctor."
- Summarize key points at the end of explanations
"""


async def create_agent(**kwargs) -> Agent:
    """Create a healthcare information agent with Grok TTS (leo voice)."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Dr. Avery - WellBridge Health", id="agent"),
        instructions=HEALTHCARE_INSTRUCTIONS,
        tts=grok_tts.TTS(voice="leo"),
        stt=deepgram.STT(eager_turn_detection=True),
        llm=gemini.LLM(),
        turn_detection=smart_turn.TurnDetection(
            silence_duration_ms=2500,
            speech_probability_threshold=0.5,
        ),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and greet the patient."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Healthcare Agent...")

    async with agent.join(call):
        logger.info("Agent joined call")

        await asyncio.sleep(3)
        await agent.llm.simple_response(
            text=(
                "Hello, and welcome to WellBridge Health. "
                "I'm Dr. Avery, your health information assistant. "
                "I can help with general wellness questions, "
                "explain health topics, or help you prepare for a doctor visit. "
                "What would you like to know about today?"
            )
        )

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
