"""Gemini Realtime GitHub MCP Demo - Demonstrate function calling with Gemini Live and GitHub MCP.

This demo shows how Gemini Realtime can use GitHub MCP tools for real-time function calling
during live conversations. The agent can interact with GitHub repositories, issues, and more
using voice commands through the Gemini Live API.
"""

import logging
import os

from dotenv import load_dotenv

from vision_agents.core.agents import Agent, AgentLauncher
from vision_agents.core import cli
from vision_agents.core.mcp import MCPServerRemote
from vision_agents.plugins.gemini.gemini_realtime import Realtime
from vision_agents.plugins import getstream
from vision_agents.core.events import CallSessionParticipantJoinedEvent
from vision_agents.core.edge.types import User

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_agent(**kwargs) -> Agent:
    """Demonstrate Gemini Realtime with GitHub MCP server integration."""

    # Get GitHub PAT from environment
    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        logger.error("GITHUB_PAT environment variable not found!")
        logger.error("Please set GITHUB_PAT in your .env file or environment")
        raise ValueError("GITHUB_PAT environment variable not found")

    # Get Google API key from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("GOOGLE_API_KEY environment variable not found!")
        logger.error("Please set GOOGLE_API_KEY in your .env file or environment")
        raise ValueError("GOOGLE_API_KEY environment variable not found")

    # Create GitHub MCP server
    github_server = MCPServerRemote(
        url="https://api.githubcopilot.com/mcp/",
        headers={"Authorization": f"Bearer {github_pat}"},
        timeout=10.0,  # Shorter connection timeout
        session_timeout=300.0,
    )

    # Create Gemini Realtime LLM
    llm = Realtime(
        model="gemini-2.5-flash-native-audio-preview-09-2025", api_key=google_api_key
    )

    # Create agent with GitHub MCP server and Gemini Realtime LLM
    agent = Agent(
        edge=getstream.Edge(),
        llm=llm,
        agent_user=User(name="GitHub AI Assistant", id="github-agent"),
        instructions="You are a helpful AI assistant with access to GitHub via MCP server. You can help with GitHub operations like creating issues, managing pull requests, searching repositories, and more. Keep responses conversational and helpful. When you need to perform GitHub operations, use the available MCP tools.",
        processors=[],
        mcp_servers=[github_server],
    )

    logger.info("Agent created with Gemini Realtime and GitHub MCP server")
    logger.info(f"GitHub server: {github_server}")

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
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
            await agent.simple_response(
                f"Hello {event.participant.user.name}! I'm your GitHub AI assistant powered by Gemini Live. I have access to {len(mcp_functions)} GitHub tools and can help you with repositories, issues, pull requests, and more through voice commands!"
            )

        # ensure the agent user is created
        await agent.create_user()
        # Create a call
        call = await agent.create_call(call_type, call_id)

        # Have the agent join the call/room
        logger.info("🎤 Agent joining call...")
        with await agent.join(call):
            logger.info(
                "✅ Agent is now live with Gemini Realtime! You can talk to it in the browser."
            )
            logger.info("Try asking:")
            logger.info("  - 'What repositories do I have?'")
            logger.info("  - 'Create a new issue in my repository'")
            logger.info("  - 'Search for issues with the label bug'")
            logger.info("  - 'Show me recent pull requests'")
            logger.info("")
            logger.info(
                "The agent will use Gemini Live's real-time function calling to interact with GitHub!"
            )

            # Run until the call ends
            await agent.finish()

    except Exception as e:
        logger.error(f"Error with Gemini Realtime GitHub MCP demo: {e}")
        logger.error("Make sure your GITHUB_PAT and GOOGLE_API_KEY are valid")
        import traceback

        traceback.print_exc()

    # Clean up
    await agent.close()
    logger.info("Demo completed!")


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
