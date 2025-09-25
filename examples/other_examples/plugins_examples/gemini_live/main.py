#!/usr/bin/env python3
"""
Example: Gemini Live using Agent class

This example demonstrates how to:
1. Create an Agent with Gemini Live realtime capabilities
2. Join a Stream video call
3. Enable realtime speech-to-speech conversation

Usage:
    python main.py

Requirements:
    - Create a .env file with your Stream and Gemini credentials
    - Install dependencies: pip install -e .
"""

import asyncio
from uuid import uuid4
from dotenv import load_dotenv

from stream_agents.core.agents import Agent
from stream_agents.core.edge.types import User
from stream_agents.plugins import gemini, getstream

load_dotenv()

async def main():
    # Create agent with Gemini Live realtime LLM
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Gemini Live AI", id="gemini-bot"),
        instructions="You are a helpful AI assistant with realtime capabilities powered by Gemini Live.",
        llm=gemini.Realtime(),
    )

    # Create call and open demo
    call = agent.edge.client.video.call("default", str(uuid4()))
    agent.edge.open_demo(call)

    # Join call and start realtime conversation
    with await agent.join(call):
        await agent.finish()

if __name__ == "__main__":
    asyncio.run(main())