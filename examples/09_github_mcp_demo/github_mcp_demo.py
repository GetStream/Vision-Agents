"""GitHub MCP Demo - Connect to GitHub MCP server and demonstrate functionality."""

import asyncio
import logging
import os
from dotenv import load_dotenv

from stream_agents.core.agents import Agent
from stream_agents.core.mcp import MCPServerRemote

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
    
    # Create a mock processor to satisfy validation requirements
    from unittest.mock import MagicMock
    mock_processor = MagicMock()
    mock_processor.process_video = MagicMock()
    
    # Create agent with GitHub MCP server
    agent = Agent(
        instructions="You are a helpful AI assistant with access to GitHub via MCP server.",
        processors=[mock_processor],
        mcp_servers=[github_server]
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
        
        # Get available tools from GitHub MCP server
        logger.info("Fetching available tools from GitHub MCP server...")
        try:
            tools = await asyncio.wait_for(agent.get_mcp_tools(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.error("❌ Tool listing timed out after 15 seconds")
            logger.error("The GitHub MCP server might be slow or unresponsive")
            return
        
        if tools:
            logger.info(f"✅ Found {len(tools)} available tools:")
            for i, tool in enumerate(tools, 1):
                logger.info(f"  {i}. {tool.name}: {getattr(tool, 'description', 'No description')}")
            
            # Try to call a simple tool if available
            try:
                logger.info("\n🔍 Attempting to call a tool...")
                # Look for a simple tool to call (like list_repositories or get_user_info)
                simple_tools = [tool for tool in tools if any(keyword in tool.name.lower() 
                                for keyword in ['list', 'get', 'user', 'repo', 'info'])]
                
                if simple_tools:
                    tool_to_call = simple_tools[0]
                    logger.info(f"Calling tool: {tool_to_call.name}")
                    
                    # Call the tool with empty arguments (most GitHub tools don't require args)
                    result = await agent.call_mcp_tool(0, tool_to_call.name, {})
                    logger.info("✅ Tool call successful!")
                    logger.info(f"Result: {result}")
                else:
                    logger.info("No simple tools found to call")
                    
            except Exception as e:
                logger.warning(f"Tool call failed: {e}")
                logger.info("This might be expected if the tool requires specific arguments")
        
        else:
            logger.warning("No tools available from GitHub MCP server")
        
        # Disconnect from MCP servers
        await agent._disconnect_mcp_servers()
        logger.info("Disconnected from GitHub MCP server")
        
    except Exception as e:
        logger.error(f"Error with GitHub MCP server: {e}")
        logger.error("Make sure your GITHUB_PAT is valid and has the necessary permissions")
    
    # Clean up
    await agent.close()
    logger.info("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
