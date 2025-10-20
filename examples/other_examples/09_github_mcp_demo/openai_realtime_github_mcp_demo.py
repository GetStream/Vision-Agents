"""OpenAI Realtime GitHub MCP Demo - Demonstrate function calling with OpenAI Realtime and GitHub MCP.

This demo shows how OpenAI Realtime can use GitHub MCP tools for real-time function calling
during live conversations. The agent can interact with GitHub repositories, issues, and more
using voice commands through the OpenAI Realtime API.
"""

import asyncio
import logging
import os
from uuid import uuid4
from dotenv import load_dotenv

from vision_agents.core.agents import Agent
from vision_agents.core.mcp import MCPServerRemote
from vision_agents.plugins.openai.openai_realtime import Realtime
from vision_agents.plugins import getstream
from vision_agents.core.events import CallSessionParticipantJoinedEvent
from vision_agents.core.edge.types import User

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def start_agent():
    """Demonstrate OpenAI Realtime with GitHub MCP server integration."""

    # Get GitHub PAT from environment
    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        logger.error("GITHUB_PAT environment variable not found!")
        logger.error("Please set GITHUB_PAT in your .env file or environment")
        return

    # Check OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not found!")
        logger.error("Please set OPENAI_API_KEY in your .env file or environment")
        return

    # Create GitHub MCP server
    github_server = MCPServerRemote(
        url="https://api.githubcopilot.com/mcp/",
        headers={"Authorization": f"Bearer {github_pat}"},
        timeout=10.0,  # Shorter connection timeout
        session_timeout=300.0,
    )

    # Create OpenAI Realtime LLM (uses OPENAI_API_KEY from environment)
    llm = Realtime(model="gpt-4o-realtime-preview-2024-12-17")

    # Create real edge transport and agent user
    edge = getstream.Edge()
    agent_user = User(name="GitHub AI Assistant", id="github-agent")

    # Create agent with GitHub MCP server and Gemini Realtime LLM
    agent = Agent(
        edge=edge,
        llm=llm,
        agent_user=agent_user,
        instructions="You are a helpful AI assistant with access to GitHub via MCP server. You can help with GitHub operations like creating issues, managing pull requests, searching repositories, and more. Keep responses conversational and helpful. When you need to perform GitHub operations, use the available MCP tools.",
        processors=[],
        mcp_servers=[github_server],
    )

    logger.info("Agent created with OpenAI Realtime and GitHub MCP server")
    logger.info(f"GitHub server: {github_server}")

    try:
        # Set up event handler for when participants join
        @agent.subscribe
        async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
            # Check MCP tools after connection
            available_functions = agent.llm.get_available_functions()
            mcp_functions = [
                f for f in available_functions if f["name"].startswith("mcp_")
            ]
            logger.info(
                f"✅ Found {len(mcp_functions)} MCP tools available for function calling"
            )
            await agent.say(
                f"Hello {event.participant.user.name}! I'm your GitHub AI assistant powered by OpenAI Realtime. I have access to {len(mcp_functions)} GitHub tools and can help you with repositories, issues, pull requests, and more through voice commands!"
            )

        # Create a call
        call = agent.edge.client.video.call("default", str(uuid4()))

        # Have the agent join the call/room
        logger.info("🎤 Agent joining call...")
        with await agent.join(call):
            # Open the demo UI
            logger.info("🌐 Opening browser with demo UI...")
            await agent.edge.open_demo(call)
            logger.info(
                "✅ Agent is now live with OpenAI Realtime! You can talk to it in the browser."
            )
            logger.info("Try asking:")
            logger.info("  - 'What repositories do I have?'")
            logger.info("  - 'Create a new issue in my repository'")
            logger.info("  - 'Search for issues with the label bug'")
            logger.info("  - 'Show me recent pull requests'")
            logger.info("")
            logger.info(
                "The agent will use OpenAI Realtime's real-time function calling to interact with GitHub!"
            )

            # Run until the call ends
            await agent.finish()

    except Exception as e:
        logger.error(f"Error with OpenAI Realtime GitHub MCP demo: {e}")
        logger.error("Make sure your GITHUB_PAT and OPENAI_API_KEY are valid")
        import traceback

        traceback.print_exc()

    # Clean up
    await agent.close()
    logger.info("Demo completed!")


if __name__ == "__main__":
    asyncio.run(start_agent())
