#!/usr/bin/env python3
"""
Simple GitHub MCP Integration Test
Tests GitHub MCP integration without video calls.
"""

import asyncio
import os
from dotenv import load_dotenv
from stream_agents.llm import OpenAILLM
from stream_agents.llm.mcp_integration import MCPServerConfig

# Load environment variables
load_dotenv()

async def test_github_mcp():
    """Test GitHub MCP integration with basic operations."""
    print("🚀 GitHub MCP Integration Test")
    print("=" * 50)
    
    # Check for required environment variables
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_token:
        print("❌ Error: GITHUB_PERSONAL_ACCESS_TOKEN environment variable is required.")
        print("Please set it with your GitHub Personal Access Token.")
        print("You can create one at: https://github.com/settings/tokens")
        print("")
        print("To set the token, run:")
        print("export GITHUB_PERSONAL_ACCESS_TOKEN=your_token_here")
        print("")
        print("Or create a .env file with:")
        print("GITHUB_PERSONAL_ACCESS_TOKEN=your_token_here")
        return
    
    try:
        # Create LLM with GitHub MCP integration
        llm = OpenAILLM(
            name="gpt-4o-mini",
            instructions="""You are a helpful GitHub assistant with access to GitHub operations through MCP.

You can:
- Search and browse GitHub repositories
- Read repository information, issues, and pull requests
- Search code and users on GitHub

When working with GitHub, always be helpful and explain what you're doing."""
        )
        
        # Add GitHub MCP server
        github_config = MCPServerConfig(
            name="github",
            command="http",
            args=["https://api.githubcopilot.com/mcp/"],
            env={
                "Authorization": f"Bearer {github_token}"
            }
        )
        
        llm.add_mcp_server(github_config)
        print("🔧 GitHub MCP server configured")
        
        # Connect to MCP servers
        await llm.connect_mcp_servers()
        print("🔧 GitHub MCP server connected!")
        
        # List available functions
        all_functions = llm.function_registry.list_functions()
        github_functions = [f for f in all_functions if f.startswith("mcp_github_")]
        print(f"📋 Available GitHub functions: {len(github_functions)} functions")
        
        # Test cases
        test_cases = [
            "Search for repositories related to 'stream-agents' on GitHub",
            "Get basic information about the 'GetStream/stream-agents' repository",
            "Search for users named 'stream' on GitHub"
        ]
        
        for i, message in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: {message}")
            try:
                response = await llm.generate(message)
                print(f"🤖 Response: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
            
            # Small delay between requests
            await asyncio.sleep(2)
        
        print("\n🎉 GitHub MCP integration test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Disconnect MCP servers to clean up connections
        try:
            if 'llm' in locals():
                await llm.function_registry.mcp_manager.disconnect_all()
        except Exception as e:
            print(f"Warning: Error disconnecting MCP servers: {e}")

if __name__ == "__main__":
    asyncio.run(test_github_mcp())
