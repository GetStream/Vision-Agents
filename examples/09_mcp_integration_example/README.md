# MCP Integration Example

This example demonstrates how to integrate MCP (Model Context Protocol) servers with Stream Agents, allowing LLMs to use external tools and data sources through a standardized protocol.

## What is MCP?

MCP (Model Context Protocol) is a standardized framework for AI systems to interact with external tools, data sources, and applications. It enables:

- **Standardized Integration**: Use any MCP-compatible tool with any LLM
- **Ecosystem Access**: Leverage existing MCP servers (filesystem, databases, APIs)
- **Security**: Tools run in separate processes with their own authentication
- **Interoperability**: Works across different AI providers and platforms

## Prerequisites

1. **Install MCP package**:
   ```bash
   pip install mcp
   ```

2. **Install MCP filesystem server**:
   ```bash
   npm install -g @modelcontextprotocol/server-filesystem
   ```

3. **Set up environment variables** in `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key
   STREAM_API_KEY=your_stream_api_key
   STREAM_API_SECRET=your_stream_api_secret
   DEEPGRAM_API_KEY=your_deepgram_api_key
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   FAL_KEY=your_fal_key
   ```

## Features Demonstrated

### 🔧 **MCP Server Integration**
- Connect to MCP filesystem server
- Automatically discover available tools
- Register MCP tools as LLM functions

### 📁 **File Operations via MCP**
- Read files using MCP filesystem tools
- Write files using MCP filesystem tools
- List directory contents
- File management operations

### 🧮 **Mixed Function Types**
- Local functions (calculations, system info)
- MCP tools (file operations)
- Seamless integration between both types

### 🎥 **Real-time Video Calls**
- Use MCP tools in live video conversations
- Voice-activated file operations
- Real-time function calling

## Usage

### Basic MCP Integration

```python
from stream_agents.llm import OpenAILLM
from stream_agents.llm.mcp_integration import MCPServerConfig

# Create LLM
llm = OpenAILLM(name="gpt-4o-mini")

# Add MCP filesystem server
config = MCPServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
)
llm.add_mcp_server(config)

# Connect to MCP servers
await llm.connect_mcp_servers()

# Now you can use filesystem tools!
response = await llm.generate("What files are in the directory?")
```

### Custom Agent with MCP

```python
class MCPAgent(Agent):
    def __init__(self, edge, llm, stt, tts, turn_detection):
        super().__init__(edge, llm, stt, tts, turn_detection)
        
        # Add local functions
        @self.llm.function("Calculate math expression")
        async def calculate(expression: str) -> str:
            return f"Result: {eval(expression)}"
```

## Available MCP Tools

When using the filesystem MCP server, the following tools are available:

- `mcp_filesystem_read_file`: Read file contents
- `mcp_filesystem_write_file`: Write content to file
- `mcp_filesystem_list_directory`: List directory contents
- `mcp_filesystem_create_directory`: Create new directory
- `mcp_filesystem_remove_file`: Delete file
- `mcp_filesystem_search_files`: Search for files

## Running the Example

1. **Install dependencies**:
   ```bash
   cd examples/09_mcp_integration_example
   uv sync
   ```

2. **Run the example**:
   ```bash
   uv run python mcp_example.py
   ```

3. **Test MCP integration**:
   - The example will first test MCP integration with file operations
   - Then start a video call where you can use voice commands
   - Try saying: "What files are in the directory?" or "Calculate 25 * 4"

## Example Conversations

### File Operations
- "What files are in the current directory?"
- "Read the content of test.txt"
- "Create a new file called 'demo.txt' with the content 'Hello MCP!'"
- "List all files again to see the new file"

### Mixed Operations
- "Calculate 15 * 7 + 23"
- "What system are we running on?"
- "Read the welcome.txt file and then calculate the length of its content"

## Security Considerations

- **File Access**: MCP filesystem server is restricted to the specified directory
- **Math Operations**: Local calculate function only allows basic math operations
- **Process Isolation**: MCP tools run in separate processes
- **Authentication**: MCP servers can use their own authentication methods

## Extending the Example

You can extend this example by:

1. **Adding More MCP Servers**:
   ```python
   # Database server
   db_config = MCPServerConfig(
       name="database",
       command="python",
       args=["-m", "mcp_server_database", "--connection-string", "sqlite:///test.db"]
   )
   llm.add_mcp_server(db_config)
   ```

2. **Custom MCP Servers**: Create your own MCP servers for specific use cases

3. **Tool Filtering**: Select which MCP tools to expose to the LLM

4. **Authentication**: Add authentication tokens for MCP servers

## Troubleshooting

### Common Issues

1. **MCP package not found**:
   ```bash
   pip install mcp
   ```

2. **Filesystem server not found**:
   ```bash
   npm install -g @modelcontextprotocol/server-filesystem
   ```

3. **Permission errors**: Ensure the MCP server has access to the specified directory

4. **Connection timeouts**: Check that the MCP server command is correct and accessible

### Debug Mode

Enable debug logging to see MCP communication:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Learn More

- [MCP Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Available MCP Servers](https://github.com/modelcontextprotocol/servers)
- [Stream Agents Documentation](https://getstream.io/video/docs/python-ai/)
