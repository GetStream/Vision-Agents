"""GitHub MCP Demo - Demonstrate automatic MCP tool registration with function registry.

This demo shows how MCP tools are automatically registered with the LLM's function registry
when an agent connects to MCP servers. The tools become available for function calling
by the LLM without any manual registration required.
"""

import asyncio
import os
from uuid import uuid4
from dotenv import load_dotenv

from stream_agents.core.agents import Agent
from stream_agents.core.mcp import MCPServerRemote
from stream_agents.plugins.openai.openai_llm import OpenAILLM
from stream_agents.plugins import elevenlabs, deepgram, silero, getstream
from stream_agents.core import cli
from stream_agents.core.events import CallSessionParticipantJoinedEvent
from stream_agents.core.edge.types import User

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate GitHub MCP server integration."""
    
    # Get GitHub PAT from environment
    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        logger.error("GITHUB_PAT environment variable not found!")
        logger.error("Please set GITHUB_PAT in your .env file or environment")
        return
    
    # Create GitHub MCP server
    github_server = MCPServerRemote(
        url="https://api.githubcopilot.com/mcp/",
        headers={"Authorization": f"Bearer {github_pat}"},
        timeout=10.0,  # Shorter connection timeout
        session_timeout=300.0
    )
    
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not found!")
        logger.error("Please set OPENAI_API_KEY in your .env file or environment")
        return
    
    # Create OpenAI LLM
    llm = OpenAILLM(model="gpt-4o", api_key=openai_api_key)
    
    # Create real edge transport and agent user
    edge = getstream.Edge()
    agent_user = User(name="GitHub AI Assistant", id="github-agent")
    
    # Create agent with GitHub MCP server and OpenAI LLM
    agent = Agent(
        edge=edge,
        llm=llm,
        agent_user=agent_user,
        instructions="You are a helpful AI assistant with access to GitHub via MCP server. You can help with GitHub operations like creating issues, managing pull requests, searching repositories, and more. Keep responses conversational and helpful.",
        processors=[],
        mcp_servers=[github_server],
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        vad=silero.VAD()
    )
    
    logger.info("Agent created with GitHub MCP server")
    logger.info(f"GitHub server: {github_server}")
    
    try:
        # Connect to GitHub MCP server with timeout
        logger.info("Connecting to GitHub MCP server...")
        try:
            await asyncio.wait_for(agent._connect_mcp_servers(), timeout=30.0)
            logger.info("✅ Successfully connected to GitHub MCP server")
        except asyncio.TimeoutError:
            logger.error("❌ Connection to GitHub MCP server timed out after 30 seconds")
            logger.error("This might be due to network issues or server unavailability")
            return
        
        # Check if MCP tools were registered with the function registry
        logger.info("Checking function registry for MCP tools...")
        available_functions = agent.llm.get_available_functions()
        mcp_functions = [f for f in available_functions if f['name'].startswith('mcp_')]
        
        logger.info(f"✅ Found {len(mcp_functions)} MCP tools registered in function registry")
        logger.info("MCP tools are now available to the LLM for function calling!")
        
        # Create the agent user
        await agent.create_user()
        
        # Set up event handler for when participants join
        @agent.subscribe
        async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
            await agent.say(f"Hello {event.participant.user.name}! I'm your GitHub AI assistant with access to {len(mcp_functions)} GitHub tools. I can help you with repositories, issues, pull requests, and more!")
        
        # Create a call
        call = agent.edge.client.video.call("default", str(uuid4()))
        
        # Open the demo UI
        logger.info("🌐 Opening browser with demo UI...")
        agent.edge.open_demo(call)
        
        # Have the agent join the call/room
        logger.info("🎤 Agent joining call...")
        with await agent.join(call):
            logger.info("✅ Agent is now live! You can talk to it in the browser.")
            logger.info("Try asking: 'What repositories do I have?' or 'Create a new issue'")
            
            # Run until the call ends
            await agent.finish()
        
    except Exception as e:
        logger.error(f"Error with GitHub MCP server: {e}")
        logger.error("Make sure your GITHUB_PAT and OPENAI_API_KEY are valid")
        import traceback
        traceback.print_exc()
    
    # Clean up
    await agent.close()
    logger.info("Demo completed!")


if __name__ == "__main__":
    asyncio.run(cli.start_dispatcher(main))
