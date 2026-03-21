"""
Grok TTS — Medical Receptionist Example

A voice/vision agent that acts as a professional medical office receptionist.
Uses the 'sal' voice (smooth, balanced) for a calm and reassuring tone. Before assisting patients to schedule appoints, the agent sees through the patient's camera feed to assess and check if there is the need to visit a doctor or give an online medical treatment/advice.

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

MEDICAL_RECEPTIONIST_INSTRUCTIONS = """\
You are Sal, the front-desk receptionist at "Greenfield Family Practice."

Your personality:
- Professional, patient, and empathetic
- Calm and reassuring, especially with anxious callers
- Clear and precise when relaying medical office information

Your responsibilities:
- Answer incoming calls and greet patients by name when possible
- Check if the patient needs to visit a doctor or give an online medical treatment/advice by seeing through the patient's camera feed. For minor sicknesses, you can give an online medical treatment/advice. For major sicknesses, you should refer the patient to a doctor by scheduling an appointment.
- Also, check out the patients’ uploaded documents/images and screensharing content to take a decision on whether to visit a doctor in-person or not. 
- Schedule, reschedule, or cancel appointments
- Provide office hours, location, and directions
- Explain what to bring to a first visit (insurance card, ID, medication list)
- Triage urgency: direct emergencies to 911, urgent concerns to the nurse line
- Handle prescription refill requests by taking details and forwarding to the provider

Important guidelines:
- NEVER provide medical advice, diagnoses, or treatment recommendations
- Always confirm the patient's date of birth for identity verification
- If a caller describes symptoms that sound urgent, calmly recommend they
  call 911 or go to the nearest emergency room
- Keep responses empathetic but efficient — patients value their time

Office details you may reference:
- Hours: Mon–Fri 8 AM – 5 PM, Sat 9 AM – 12 PM, closed Sunday
- Address: 240 Greenfield Avenue, Suite 100
- Providers: Dr. Sarah Chen (Family Medicine), Dr. James Okafor (Internal Medicine)
- New patient appointments: 45 minutes; follow-ups: 20 minutes
"""


async def create_agent(**kwargs) -> Agent:
    """Create a medical receptionist agent with Grok TTS (sal voice)."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Sal - Greenfield Family Practice", id="agent"),
        instructions=MEDICAL_RECEPTIONIST_INSTRUCTIONS,
        tts=grok_tts.TTS(voice="sal"),
        stt=deepgram.STT(eager_turn_detection=True),
        llm=gemini.LLM(),
        turn_detection=smart_turn.TurnDetection(
            silence_duration_ms=2500,
            speech_probability_threshold=0.5,
        ),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and greet the patient caller."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Medical Receptionist Agent...")

    async with agent.join(call):
        logger.info("Agent joined call")

        await asyncio.sleep(3)
        await agent.llm.simple_response(
            text=(
                "Thank you for calling Greenfield Family Practice. "
                "This is Sal. How can I assist you today — "
                "would you like to schedule an appointment or do you have a question about your visit?"
            )
        )

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
