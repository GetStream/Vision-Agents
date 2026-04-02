"""
Transformers MCP Tool Calling Example

Demonstrates tool calling with a local HuggingFace model using MCP servers.
The model runs on your hardware (MPS/CUDA/CPU) and can invoke tools exposed
by any MCP server — this example connects the GitHub MCP server so the
agent can look up your profile and starred repos (requires 2+ tool calls).

Creates an agent that uses:
- TransformersLLM for local inference with tool calling
- GitHub MCP server for repository / user queries
- Deepgram for speech-to-text (STT)
- Deepgram for text-to-speech (TTS)
- GetStream for edge/real-time communication

Requirements:
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- DEEPGRAM_API_KEY environment variable
- GITHUB_PAT environment variable (GitHub personal access token)
- npx available on PATH (ships with Node.js)

First run will download the model (~3 GB for Qwen2.5-1.5B-Instruct).
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.core.mcp import MCPServerLocal
from vision_agents.plugins import deepgram, getstream
from vision_agents.plugins.huggingface.transformers_llm import TransformersLLM

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with a local LLM and GitHub MCP tools."""
    llm = TransformersLLM(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=200,
    )

    github_pat = os.environ.get("GITHUB_PAT", "")
    if not github_pat:
        raise ValueError("GITHUB_PAT environment variable is required")

    github_mcp = MCPServerLocal(
        command="npx -y @modelcontextprotocol/server-github",
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": github_pat},
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="GitHub Agent", id="agent"),
        instructions=(
            "You are a helpful voice assistant with access to GitHub tools. "
            "When the user asks about their GitHub profile, first retrieve "
            "their username with get_authenticated_user, then use it to "
            "look up their starred repos with list_starred_repos. "
            "Always call the tools — never guess. "
            "Keep responses short and conversational."
        ),
        llm=llm,
        tts=deepgram.TTS(),
        stt=deepgram.STT(),
        mcp_servers=[github_mcp],
        streaming_tts=True,
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting GitHub MCP Agent...")

    async with agent.join(call):
        logger.info("Joining call")

        await asyncio.sleep(2)
        await agent.llm.simple_response(
            text=(
                "Greet the user and let them know you can look up their "
                "GitHub profile and starred repositories."
            ),
        )

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
