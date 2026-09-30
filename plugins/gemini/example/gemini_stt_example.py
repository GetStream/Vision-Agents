"""
Gemini STT Example

Demonstrates Gemini Live speech-to-text with the rest of the Gemini suite:

- Gemini Live for speech-to-text (STT)
- Gemini for the text LLM
- ElevenLabs for text-to-speech (TTS)
- GetStream for edge/real-time communication

Requirements:
- GOOGLE_API_KEY or GEMINI_API_KEY environment variable
- ELEVENLABS_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import elevenlabs, gemini, getstream
from vision_agents.plugins.getstream import CallSessionParticipantJoinedEvent

logger = logging.getLogger(__name__)

load_dotenv()

em_vocab = [
    "Teenage Engineering",
    "OP-XY",
    "Logic Pro",
    "Dawesome",
    "ZYKLØP",
    "ZYKLOP",
    "MYTH",
    "oeksound",
    "Soothe3",
    "Bloom",
    "Cableguys",
    "ShaperBox 3",
    "Baby Audio",
    "Transit 2",
    "Ableton Move",
    "Ableton Link",
    "MIDI clock",
]


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Gemini STT Agent", id="gemini-stt-agent"),
        instructions="You're a helpful voice AI assistant. Keep replies short and conversational.",
        stt=gemini.STT(
            language_codes=["en-US", "de-DE"],
            custom_vocabulary=em_vocab,
        ),
        llm=gemini.LLM(model="gemini-3.1-flash-lite"),
        tts=elevenlabs.TTS(voice_id="7A85ufQZSEaTbZ5eQ4f4"),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        if event.participant.user.id != "gemini-stt-agent":
            await asyncio.sleep(2)
            await agent.simple_response("Hello! How can I help you today?")

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
