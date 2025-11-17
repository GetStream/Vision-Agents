#!/usr/bin/env python3
"""
Example: MCP Integration using Agent class

This example demonstrates how to:
1. Create an Agent with MCP server capabilities
2. Join a Stream video call
3. Use MCP tools in conversation

Usage:
    python main.py

Requirements:
    - Create a .env file with your Stream, Deepgram, ElevenLabs, and OpenAI credentials
    - Install dependencies: pip install -e .
"""

import logging

from dotenv import load_dotenv

from vision_agents.core.agents import Agent, AgentLauncher
from vision_agents.core import cli
from vision_agents.core.edge.types import User
from vision_agents.plugins import deepgram, elevenlabs, openai, getstream
from vision_agents.core.mcp import MCPBaseServer

logger = logging.getLogger(__name__)


# Example MCP server for demonstration
class ExampleMCPServer(MCPBaseServer):
    """Example MCP server that provides weather information."""

    def __init__(self):
        super().__init__("example-server")

    async def get_tools(self):
        """Return available tools."""
        return [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            }
        ]

    async def call_tool(self, name: str, arguments: dict):
        """Execute a tool call."""
        if name == "get_weather":
            location = arguments.get("location", "Unknown")
            return f"The weather in {location} is sunny and 72°F"
        return "Tool not found"


load_dotenv()


async def create_agent(**kwargs) -> Agent:
    # Create agent with MCP servers
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="MCP Bot", id="mcp-bot"),
        instructions="I can use MCP tools to help you. I have access to weather information.",
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(),
        tts=elevenlabs.TTS(),
        mcp_servers=[ExampleMCPServer()],
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    # ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    # Join call and start MCP-enabled conversation
    with await agent.join(call):
        await agent.say(
            "Hello! I have access to MCP tools including weather information. How can I help you?"
        )
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
