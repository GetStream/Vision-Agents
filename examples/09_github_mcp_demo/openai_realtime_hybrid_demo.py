"""OpenAI Realtime Hybrid Demo - Custom Functions + GitHub MCP Integration.

This demo shows how OpenAI Realtime can use both custom functions and GitHub MCP tools
for real-time function calling during live conversations. The agent can perform
calculations, get weather info, and interact with GitHub repositories.
"""

import asyncio
import logging
import os
from uuid import uuid4
from dotenv import load_dotenv

from stream_agents.core.agents import Agent
from stream_agents.core.mcp import MCPServerRemote
from stream_agents.plugins.openai.openai_realtime import Realtime
from stream_agents.plugins import getstream
from stream_agents.core import cli
from stream_agents.core.events import CallSessionParticipantJoinedEvent
from stream_agents.core.edge.types import User

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate OpenAI Realtime with both custom functions and GitHub MCP server."""
    
    # Get GitHub PAT from environment
    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        logger.error("GITHUB_PAT environment variable not found!")
        logger.error("Please set GITHUB_PAT in your .env file or environment")
        return
    
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not found!")
        logger.error("Please set OPENAI_API_KEY in your .env file or environment")
        return
    
    # Create GitHub MCP server
    github_server = MCPServerRemote(
        url="https://api.githubcopilot.com/mcp/",
        headers={"Authorization": f"Bearer {github_pat}"},
        timeout=10.0,
        session_timeout=300.0
    )
    
    # Create OpenAI Realtime LLM
    llm = Realtime(
        model="gpt-realtime",
        voice="marin"
    )
    
    # Register custom functions
    @llm.register_function(description="Get current weather for a location")
    def get_weather(location: str) -> dict:
        """Get the current weather for a location."""
        # Mock weather data - in real implementation, call weather API
        weather_data = {
            "New York": {"temperature": "22°C", "condition": "Sunny", "humidity": "65%"},
            "London": {"temperature": "15°C", "condition": "Cloudy", "humidity": "80%"},
            "Tokyo": {"temperature": "18°C", "condition": "Rainy", "humidity": "75%"},
            "San Francisco": {"temperature": "20°C", "condition": "Foggy", "humidity": "70%"}
        }
        return weather_data.get(location, {"temperature": "20°C", "condition": "Unknown", "humidity": "50%"})
    
    @llm.register_function(description="Calculate mathematical expressions")
    def calculate(expression: str) -> dict:
        """Calculate a mathematical expression safely."""
        try:
            # Simple safe evaluation for basic math
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return {"error": "Invalid characters in expression", "result": None}
            
            result = eval(expression)
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": str(e), "result": None}
    
    @llm.register_function(description="Get current time and date")
    def get_current_time() -> dict:
        """Get the current time and date."""
        from datetime import datetime
        now = datetime.now()
        return {
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "day": now.strftime("%A"),
            "timezone": "UTC"
        }
    
    # Create real edge transport and agent user
    edge = getstream.Edge()
    agent_user = User(name="Hybrid AI Assistant", id="hybrid-agent")
    
    # Create agent with both custom functions and GitHub MCP server
    agent = Agent(
        edge=edge,
        llm=llm,
        agent_user=agent_user,
        instructions="""You are a helpful AI assistant with access to both custom functions and GitHub via MCP server. 

Custom functions available:
- get_weather: Get weather information for any location
- calculate: Perform mathematical calculations
- get_current_time: Get current time and date

GitHub MCP tools available:
- Search and manage repositories
- Create and manage issues
- Handle pull requests
- Access GitHub data

Keep responses conversational and helpful. Use the appropriate tools based on what the user asks for.""",
        processors=[],
        mcp_servers=[github_server],
    )
    
    logger.info("Agent created with OpenAI Realtime, custom functions, and GitHub MCP server")
    logger.info(f"GitHub server: {github_server}")
    
    # Log available functions
    available_functions = llm.get_available_functions()
    custom_functions = [f for f in available_functions if not f['name'].startswith('mcp_')]
    logger.info(f"✅ Custom functions registered: {[f['name'] for f in custom_functions]}")
    
    try:
        # Create the agent user
        await agent.create_user()
        
        # Set up event handler for when participants join
        @agent.subscribe
        async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
            # Check all available functions
            available_functions = agent.llm.get_available_functions()
            mcp_functions = [f for f in available_functions if f['name'].startswith('mcp_')]
            custom_functions = [f for f in available_functions if not f['name'].startswith('mcp_')]
            
            logger.info(f"✅ Found {len(custom_functions)} custom functions and {len(mcp_functions)} MCP tools")
            
            await agent.simple_response(
                f"Hello {event.participant.user.name}! I'm your hybrid AI assistant powered by OpenAI Realtime. "
                f"I have {len(custom_functions)} custom functions for weather, calculations, and time, plus "
                f"{len(mcp_functions)} GitHub tools for repository management. What would you like to do?"
            )
        
        # Create a call
        call = agent.edge.client.video.call("default", str(uuid4()))
        
        # Open the demo UI
        logger.info("🌐 Opening browser with demo UI...")
        agent.edge.open_demo(call)
        
        # Have the agent join the call/room
        logger.info("🎤 Agent joining call...")
        with await agent.join(call):
            logger.info("✅ Agent is now live with OpenAI Realtime! You can talk to it in the browser.")
            logger.info("Try asking:")
            logger.info("  CUSTOM FUNCTIONS:")
            logger.info("    - 'What's the weather like in New York?'")
            logger.info("    - 'Calculate 15 * 27 + 100'")
            logger.info("    - 'What time is it?'")
            logger.info("  GITHUB MCP:")
            logger.info("    - 'What repositories do I have?'")
            logger.info("    - 'Create a new issue in my repository'")
            logger.info("    - 'Search for issues with the label bug'")
            logger.info("  COMBINED:")
            logger.info("    - 'What's the weather in London and show me my GitHub repos?'")
            logger.info("")
            logger.info("The agent will use OpenAI Realtime's function calling for both custom and MCP tools!")
            
            # Run until the call ends
            await agent.finish()
        
    except Exception as e:
        logger.error(f"Error with OpenAI Realtime hybrid demo: {e}")
        logger.error("Make sure your GITHUB_PAT and OPENAI_API_KEY are valid")
        import traceback
        traceback.print_exc()
    
    # Clean up
    await agent.close()
    logger.info("Demo completed!")


if __name__ == "__main__":
    asyncio.run(cli.start_dispatcher(main))
