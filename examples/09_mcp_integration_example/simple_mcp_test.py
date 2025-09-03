#!/usr/bin/env python3
"""
Simple MCP Integration Test
Tests MCP filesystem integration without video calls.
"""

import asyncio
import os
import tempfile
from dotenv import load_dotenv
from stream_agents.llm import OpenAILLM
from stream_agents.llm.mcp_integration import MCPServerConfig

# Load environment variables
load_dotenv()

async def main():
    print("🚀 Simple MCP Integration Test")
    print("=" * 50)
    
    # Create a persistent temporary directory
    temp_dir = tempfile.mkdtemp(prefix="mcp_test_")
    # Resolve the path to avoid symlink issues
    temp_dir = os.path.realpath(temp_dir)
    print(f"📁 Using temporary directory: {temp_dir}")
    
    # Create some test files
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("Hello from MCP filesystem!")
    
    print(f"📄 Created test file: {test_file}")
    
    try:
        # Create LLM with MCP integration
        llm = OpenAILLM(name="mcp_test_llm")
        
        # Add MCP filesystem server
        mcp_config = MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", temp_dir]
        )
        
        llm.add_mcp_server(mcp_config)
        print("🔧 MCP server configured")
        
        # Connect to MCP servers
        await llm.connect_mcp_servers()
        print("🔧 MCP servers connected!")
        
        # List available functions
        all_functions = llm.function_registry.list_functions()
        mcp_functions = [f for f in all_functions if f.startswith("mcp_")]
        print(f"📋 Available MCP functions: {mcp_functions}")
        
        # Test 1: List directory contents
        print("\n🧪 Test 1: List directory contents")
        try:
            result = await llm.function_registry.call_function(
                "mcp_filesystem_list_directory", 
                {"path": temp_dir}
            )
            print(f"✅ Directory listing: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 2: Read the test file
        print("\n🧪 Test 2: Read test file")
        try:
            result = await llm.function_registry.call_function(
                "mcp_filesystem_read_text_file", 
                {"path": test_file}
            )
            print(f"✅ File content: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 3: Write a new file
        print("\n🧪 Test 3: Write new file")
        new_file = os.path.join(temp_dir, "mcp_created.txt")
        try:
            result = await llm.function_registry.call_function(
                "mcp_filesystem_write_file", 
                {
                    "path": new_file,
                    "content": "This file was created via MCP!"
                }
            )
            print(f"✅ File written: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 4: List directory again to see new file
        print("\n🧪 Test 4: List directory again")
        try:
            result = await llm.function_registry.call_function(
                "mcp_filesystem_list_directory", 
                {"path": temp_dir}
            )
            print(f"✅ Directory listing: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
            
        print("\n🎉 MCP integration test completed!")
        
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
        
        # Keep temp directory for inspection
        print(f"📁 Temporary directory preserved: {temp_dir}")
        print("You can inspect the files manually if needed")

if __name__ == "__main__":
    asyncio.run(main())
