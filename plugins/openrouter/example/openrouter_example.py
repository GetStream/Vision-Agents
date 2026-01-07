"""
OpenRouter LLM Example

This example demonstrates how to use the OpenRouter plugin with a Vision Agent,
including function calling and MCP server integration.

Set these environment variables before running:
- OPENROUTER_API_KEY: Your OpenRouter API key
- GITHUB_PAT: (Optional) GitHub Personal Access Token for MCP integration
"""

import logging
import os

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.core.mcp import MCPBaseServer, MCPServerRemote
from vision_agents.plugins import (
    openrouter,
    getstream,
    elevenlabs,
    deepgram,
    smart_turn,
)


logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with OpenRouter LLM, function calling, and optional MCP."""
    # OpenRouter uses Chat Completions API for all models.
    # Any model available on OpenRouter can be used here.
    # For MCP/GitHub integration, Claude is recommended as it handles
    # multi-step tool reasoning well (e.g., call get_me first, then use the result)
    model = "openrouter/auto"

    llm = openrouter.LLM(model=model)

    # Register local functions that the LLM can call
    @llm.register_function(description="Get current weather for a location")
    def get_weather(location: str):
        """Get the current weather for a location."""
        return {
            "location": location,
            "temperature": "22 degrees celsius",
            "condition": "Sunny",
            "humidity": "65 percent",
        }

    @llm.register_function(description="Calculate the sum of two numbers")
    def calculate_sum(a: int, b: int):
        """Calculate the sum of two numbers."""
        return a + b

    # Optional: Set up GitHub MCP server if GITHUB_PAT is available
    mcp_servers: list[MCPBaseServer] = []
    github_pat = os.getenv("GITHUB_PAT")
    if github_pat:
        logger.info("GitHub PAT found, enabling GitHub MCP integration")
        github_server = MCPServerRemote(
            url="https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {github_pat}"},
            timeout=10.0,
            session_timeout=300.0,
        )
        mcp_servers.append(github_server)
    else:
        logger.info("No GITHUB_PAT found, running without GitHub MCP integration")

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="OpenRouter AI", id="agent"),
        instructions="""You are a helpful assistant. Answer concisely - give the final answer, not your reasoning.

TOOL USE RULES:
1. When you need info, call tools silently - don't narrate what you're doing
2. Don't ask the user for information you can look up yourself. e.g. for GitHub tasks requiring a username/owner: 
    ALWAYS call get_me first, then use the returned username
3. Chain multiple tool calls as needed - for example, if asked about user repositories 
   and you need a username, call get_me first, then use that for subsequent calls
4. Example: "How many PRs in my repo?" → call get_me → use username for list_pull_requests → report count

Available: get_weather, calculate_sum, get_me, list_pull_requests, search_repositories, and GitHub tools.""",
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        turn_detection=smart_turn.TurnDetection(
            pre_speech_buffer_ms=2000, speech_probability_threshold=0.9
        ),
        mcp_servers=mcp_servers,
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    # Ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting OpenRouter Agent...")

    # Log available functions (including MCP tools if connected)
    available_functions = agent.llm.get_available_functions()
    logger.info(f"Available functions: {len(available_functions)}")
    for func in available_functions:
        logger.info(f"  - {func['name']}: {func.get('description', '')[:50]}...")

    # Have the agent join the call/room
    async with agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
