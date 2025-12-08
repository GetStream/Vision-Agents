# GitHub MCP Demo

This example demonstrates how to connect to the GitHub MCP server using Stream Agents.

## Overview

The GitHub MCP server provides access to GitHub's API through the Model Context Protocol, allowing agents to interact with repositories, issues, pull requests, and other GitHub resources.

## Prerequisites

1. **GitHub Personal Access Token (PAT)**: You need a GitHub PAT with appropriate permissions
2. **Environment Setup**: Set up your environment variables

## Setup

### 1. Get a GitHub Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a new token with the following scopes:
   - `repo` (Full control of private repositories)
   - `read:user` (Read user profile data)
   - `read:org` (Read org and team membership)
   - `read:project` (Read project data)

### 2. Set Environment Variables

Create a `.env` file in the project root with your GitHub PAT:

```bash
# .env file
GITHUB_PAT=your_github_personal_access_token_here
```

Or set it as an environment variable:

```bash
export GITHUB_PAT=your_github_personal_access_token_here
```

## Running the Demo

1. Go to the example's directory
    ```bash
    cd examples/other_examples/09_github_mcp_demo
    ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Run the agent
```bash
uv run python github_mcp_demo.py
```

### Gemini Realtime Version
```bash
cd examples/other_examples/09_github_mcp_demo
uv run python gemini_realtime_github_mcp_demo.py
```

**Note**: The Gemini Realtime version requires `GOOGLE_API_KEY` in your `.env` file in addition to `GITHUB_PAT`.


### OpenAI Realtime Version

```bash
cd examples/other_examples/09_github_mcp_demo
uv run python openai_realtime_github_mcp_demo.py
```

**Note**: The OpenAI Realtime version requires `OPENAI_API_KEY` in your `.env` file in addition to `GITHUB_PAT`.

## What the Demo Does


1. **Connects to GitHub MCP Server**: Establishes connection to `https://api.githubcopilot.com/mcp/`
2. **Lists Available Tools**: Shows all available GitHub MCP tools
3. **Demonstrates Tool Calling**: Attempts to call a simple tool
4. **Handles Errors Gracefully**: Provides helpful error messages

### Gemini Realtime Version
1. **Connects to GitHub MCP Server**: Establishes connection to `https://api.githubcopilot.com/mcp/`
2. **Registers MCP Tools**: Automatically registers GitHub MCP tools with Gemini Live
3. **Real-time Function Calling**: Uses Gemini Live's function calling during voice conversations
4. **Interactive Demo**: Opens a browser UI where you can talk to the agent
5. **Voice Commands**: Try saying "What repositories do I have?" or "Create a new issue"


## Expected Output

```
INFO:__main__:Agent created with GitHub MCP server
INFO:__main__:GitHub server: MCPServerRemote(url='https://api.githubcopilot.com/mcp/', connected=False)
INFO:__main__:Connecting to GitHub MCP server...
INFO:MCPServerRemote:Connecting to remote MCP server at https://api.githubcopilot.com/mcp/
INFO:MCPServerRemote:Successfully connected to remote MCP server at https://api.githubcopilot.com/mcp/ (session: xyz123)
INFO:__main__:Fetching available tools from GitHub MCP server...
INFO:__main__:✅ Found 5 available tools:
INFO:__main__:  1. list_repositories: List user repositories
INFO:__main__:  2. get_user_info: Get current user information
INFO:__main__:  3. create_issue: Create a new issue
INFO:__main__:  4. list_issues: List repository issues
INFO:__main__:  5. get_repository: Get repository information
```