#!/usr/bin/env python3
"""
GitHub MCP Integration Example

This example demonstrates how to integrate GitHub's official MCP server
with Stream Agents, allowing LLMs to interact with GitHub repositories,
issues, pull requests, and more.

Prerequisites:
1. Install GitHub MCP server: https://github.com/github/github-mcp-server
2. Set up GitHub Personal Access Token with appropriate permissions
3. Configure environment variables

This example shows:
- Adding GitHub MCP server to an LLM
- Using GitHub tools through MCP (repos, issues, PRs, etc.)
- Mixing local functions with GitHub MCP tools
- Real-time GitHub operations in video calls
"""

import asyncio
import os
import tempfile
from uuid import uuid4

from dotenv import load_dotenv
from getstream import Stream
from getstream.plugins import DeepgramSTT, ElevenLabsTTS
from stream_agents.turn_detection import FalTurnDetection
from stream_agents.agents import Agent
from stream_agents.edge import StreamEdge
from stream_agents.llm import OpenAILLM
from stream_agents.llm.mcp_integration import MCPServerConfig
from stream_agents import open_demo, start_dispatcher

# Load environment variables
load_dotenv()


class GitHubMCPAgent(Agent):
    """Custom agent with GitHub MCP integration for repository operations."""
    
    def __init__(self, edge, llm, stt, tts, turn_detection, processors=None):
        super().__init__(edge, llm, stt, tts, turn_detection, processors)
        
        # Add local functions
        self._register_local_functions()
    
    def _register_local_functions(self):
        """Register local functions alongside GitHub MCP tools."""
        
        @self.llm.function("Get system information")
        async def get_system_info() -> str:
            """Get basic system information."""
            import platform
            return f"System: {platform.system()} {platform.release()}, Python {platform.python_version()}"
        
        @self.llm.function("Format GitHub URL")
        async def format_github_url(owner: str, repo: str, path: str = "") -> str:
            """Format a GitHub URL for a repository or specific path."""
            base_url = f"https://github.com/{owner}/{repo}"
            if path:
                return f"{base_url}/blob/main/{path}"
            return base_url


async def setup_github_mcp_server() -> MCPServerConfig:
    """Set up GitHub MCP server configuration."""
    
    # Check for required environment variables
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_token:
        raise ValueError(
            "GITHUB_PERSONAL_ACCESS_TOKEN environment variable is required. "
            "Please set it with your GitHub Personal Access Token."
        )
    
    # GitHub MCP server is a remote HTTP service
    return MCPServerConfig(
        name="github",
        command="http",
        args=["https://api.githubcopilot.com/mcp/"],
        env={
            "Authorization": f"Bearer {github_token}"
        }
    )








async def main():
    """Main function to run the GitHub MCP integration example."""
    
    print("🚀 GitHub MCP Integration Example")
    print("=" * 60)
    print("This example demonstrates GitHub integration through MCP.")
    print("=" * 60)
    
    # Check environment setup
    if not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
        print("❌ Error: GITHUB_PERSONAL_ACCESS_TOKEN environment variable is required.")
        print("Please set it with your GitHub Personal Access Token.")
        print("You can create one at: https://github.com/settings/tokens")
        return
    
    print("🎥 Starting video call with GitHub MCP integration...")
    
    # Create Stream client
    client = Stream.from_env()
    
    # Create OpenAI LLM with GitHub MCP integration
    llm = OpenAILLM(
        name="gpt-4o-mini",
        instructions="""You are a helpful GitHub assistant with access to GitHub operations through MCP.

You can help users with:
- Repository operations (search, browse, read files)
- Issue and pull request management
- Code search and user search
- GitHub Actions and security features
- Repository statistics and information

Always explain what you're doing when working with GitHub and be helpful and friendly."""
    )
    
    # Add GitHub MCP server
    github_config = await setup_github_mcp_server()
    llm.add_mcp_server(github_config)
    
    # Connect to MCP servers
    await llm.connect_mcp_servers()
    
    print("🔧 GitHub MCP server connected!")
    print(f"📋 Available functions: {len(llm.function_registry.list_functions())} functions")
    
    # Create the GitHub MCP-enabled agent
    agent = GitHubMCPAgent(
        edge=StreamEdge(),
        llm=llm,
        stt=DeepgramSTT(),
        tts=ElevenLabsTTS(),
        turn_detection=FalTurnDetection()
    )
    
    # Create a call
    call = client.video.call("default", str(uuid4()))
    
    # Open the demo UI
    open_demo(call)
    
    print("📞 Call created!")
    print("🎤 You can now talk to the agent and ask it to:")
    print("   - Search GitHub repositories")
    print("   - Read repository information")
    print("   - List issues and pull requests")
    print("   - Search for code or users")
    print("   - Get repository statistics")
    print("\n💡 Try saying: 'Search for Python repositories' or 'Show me the latest issues in GetStream/stream-agents'")
    
    # Have the agent join the call
    with await agent.join(call):
        # Send initial greeting
        await agent.create_response("Hello! I'm your GitHub assistant with access to GitHub operations through MCP. You can ask me to search repositories, read repository information, manage issues, or search for code and users. What would you like me to help you with?")
        
        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    asyncio.run(start_dispatcher(main))
