"""
Computer use example — the agent sees your screen share and can control your desktop.

Uses:
- Gemini Realtime for live screen-share vision + tool calling
- Stream's edge network for video transport
- Computer-use plugin for desktop actions (click, type, scroll, etc.)
- Grid overlay processor so the LLM can reference labeled cells

Share your screen in the call, then ask the agent to perform actions
like "open my Downloads folder" or "click on the Safari icon".
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.plugins import computer_use, gemini, getstream

logger = logging.getLogger(__name__)

load_dotenv()


grid = computer_use.Grid(cols=15, rows=15)


def setup_llm() -> gemini.Realtime:
    llm = gemini.Realtime(fps=2)
    computer_use.register(llm, grid=grid)
    return llm


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Desktop Assistant", id="desktop-agent"),
        instructions="Read @examples/10_computer_use_example/instructions.md",
        llm=setup_llm(),
        processors=[computer_use.GridOverlayProcessor(grid=grid, fps=2)],
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.llm.simple_response(
            text="Say hi and let the user know they can share their screen and ask you to perform actions on their computer."
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
