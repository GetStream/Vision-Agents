"""ByteDance Live Interpretation Example.

Runs an interpreter agent that translates speech in real time using ByteDance
Seed Speech AST 2.0 (Live Interpretation). Join the call and speak in the
source language; the agent speaks the translation and emits subtitles.

This model is a translator, not a chat model, so there is no STT/LLM/TTS
pipeline — just `llm=bytedance.Realtime(...)`.

Requirements:
- BYTEDANCE_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
"""

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import bytedance, getstream

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="ByteDance Interpreter", id="agent"),
        instructions="",
        llm=bytedance.Realtime(source_language="zh", target_language="en"),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.edge.open_demo(call)
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
