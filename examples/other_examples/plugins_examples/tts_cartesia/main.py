#!/usr/bin/env python3
"""
Example: Text-to-Speech with Cartesia using Agent class

This minimal example shows how to:
1. Create an Agent with TTS capabilities
2. Join a Stream video call
3. Greet users when they join

Run it, join the call in your browser, and hear the bot greet you 🗣️

Usage::
    python main.py

The script looks for the following env vars (see `env.example`):
    STREAM_API_KEY / STREAM_API_SECRET
    CARTESIA_API_KEY
"""

import asyncio
from uuid import uuid4
from dotenv import load_dotenv

from stream_agents.core.agents import Agent
from stream_agents.core.edge.types import User
from stream_agents.plugins import cartesia, getstream
from stream_agents.core.events import CallSessionParticipantJoinedEvent

load_dotenv()

async def main():
    # Create agent with TTS
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="TTS Bot", id="tts-bot"),
        instructions="I'm a TTS bot that greets users when they join.",
        tts=cartesia.TTS(),
    )

    # Subscribe to participant joined events
    @agent.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        await agent.simple_response(f"Hello {event.participant.user.name}! Welcome to the call.")

    # Create call and open demo
    call = agent.edge.client.video.call("default", str(uuid4()))
    agent.edge.open_demo(call)

    # Join call and wait
    with await agent.join(call):
        await agent.finish()

if __name__ == "__main__":
    asyncio.run(main())