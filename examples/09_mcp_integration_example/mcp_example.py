"""
MCP Integration Example

This example demonstrates how to integrate MCP (Model Context Protocol) servers
with Stream Agents, allowing LLMs to use external tools and data sources.

Prerequisites:
1. Install MCP package: pip install mcp
2. Install MCP filesystem server: npm install -g @modelcontextprotocol/server-filesystem
3. Create a test directory for file operations

This example shows:
- Adding MCP servers to an LLM
- Using filesystem tools through MCP
- Mixing local functions with MCP tools
- Real-time function calling in video calls
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

# Load environment variables
load_dotenv()


class MCPAgent(Agent):
    """Custom agent with MCP integration for file operations."""
    
    def __init__(self, edge, llm, stt, tts, turn_detection, processors=None):
        super().__init__(edge, llm, stt, tts, turn_detection, processors)
        
        # Add local functions
        self._register_local_functions()
    
    def _register_local_functions(self):
        """Register local functions alongside MCP tools."""
        
        @self.llm.function("Get system information")
        async def get_system_info() -> str:
            """Get basic system information."""
            import platform
            return f"System: {platform.system()} {platform.release()}, Python {platform.python_version()}"
        
        @self.llm.function("Calculate mathematical expression")
        async def calculate(expression: str) -> str:
            """Safely evaluate a mathematical expression."""
            try:
                # Only allow basic math operations for safety
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    return "Error: Only basic math operations are allowed"
                
                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Error calculating '{expression}': {str(e)}"


async def setup_mcp_filesystem_server(test_dir: str) -> MCPServerConfig:
    """Set up MCP filesystem server configuration."""
    return MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", test_dir],
        env={"NODE_ENV": "production"}
    )


async def test_mcp_integration():
    """Test MCP integration with a simple conversation."""
    
    # Create a temporary directory for file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Create some test files
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Hello from MCP filesystem server!")
        
        # Create OpenAI LLM with MCP integration
        llm = OpenAILLM(
            name="gpt-4o-mini",  # Use cheaper model for demo
            instructions="""You are a helpful assistant with access to file system operations through MCP.
            
You can:
- Read and write files using MCP filesystem tools
- Perform calculations using the local calculate function
- Get system information using the local get_system_info function

When working with files, always be careful and explain what you're doing.
Use the MCP filesystem tools for file operations and local functions for other tasks."""
        )
        
        # Add MCP filesystem server
        mcp_config = await setup_mcp_filesystem_server(temp_dir)
        llm.add_mcp_server(mcp_config)
        
        # Connect to MCP servers
        await llm.connect_mcp_servers()
        
        print("🔧 MCP servers connected!")
        print(f"📋 Available functions: {llm.function_registry.list_functions()}")
        
        # Test conversations
        test_cases = [
            "What files are in the current directory?",
            "Read the content of test.txt",
            "Create a new file called 'mcp_demo.txt' with the content 'MCP integration works!'",
            "Calculate 15 * 7 + 23",
            "What system are we running on?",
            "List all files again to see the new file"
        ]
        
        for i, message in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: {message}")
            try:
                response = await llm.generate(message)
                print(f"🤖 Response: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Small delay between requests
            await asyncio.sleep(1)


async def create_mcp_agent_with_functions():
    """Create an agent with MCP integration for video calls."""
    
    # Create a temporary directory for file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Create some test files
        test_file = os.path.join(temp_dir, "welcome.txt")
        with open(test_file, "w") as f:
            f.write("Welcome to MCP-powered Stream Agents!")
        
        # Create OpenAI LLM with MCP integration
        llm = OpenAILLM(
            name="gpt-4o-mini",
            instructions="""You are a helpful assistant with access to file system operations through MCP.

You can help users with:
- File operations (read, write, list files) using MCP filesystem tools
- Mathematical calculations using the local calculate function
- System information using the local get_system_info function

Always explain what you're doing when working with files and be helpful and friendly."""
        )
        
        # Add MCP filesystem server
        mcp_config = await setup_mcp_filesystem_server(temp_dir)
        llm.add_mcp_server(mcp_config)
        
        # Connect to MCP servers
        await llm.connect_mcp_servers()
        
        # Create the custom agent
        agent = MCPAgent(
            edge=StreamEdge(),
            llm=llm,
            stt=DeepgramSTT(),
            tts=ElevenLabsTTS(),
            turn_detection=FalTurnDetection()
        )
        
        return agent


async def main():
    """Main function to run the MCP integration example."""
    
    print("🚀 MCP Integration Example")
    print("=" * 50)
    
    # Test MCP integration first
    print("🧪 Testing MCP integration...")
    await test_mcp_integration()
    
    print("\n" + "=" * 50)
    print("🎥 Starting video call with MCP integration...")
    
    # Create Stream client
    client = Stream.from_env()
    
    # Create the MCP-enabled agent
    agent = await create_mcp_agent_with_functions()
    
    # Create a call
    call = client.video.call(
        type="default",
        id=str(uuid4()),
    )
    
    # Join the call
    await call.join(create=True)
    
    print("📞 Call created and joined!")
    print("🎤 You can now talk to the agent and ask it to:")
    print("   - Read and write files")
    print("   - Perform calculations")
    print("   - Get system information")
    print("   - List directory contents")
    print("\n💡 Try saying: 'What files are in the directory?' or 'Calculate 25 * 4'")
    
    # Start the agent
    await agent.start(call)
    
    print("🤖 Agent started! Press Ctrl+C to stop.")
    
    try:
        # Keep the agent running
        await asyncio.sleep(3600)  # Run for 1 hour
    except KeyboardInterrupt:
        print("\n🛑 Stopping agent...")
    finally:
        await agent.stop()
        await call.leave()


if __name__ == "__main__":
    asyncio.run(main())
