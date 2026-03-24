"""
Monty Agent Example — Gemini Realtime + sandboxed Python execution.

A voice AI agent that can write and execute Python code in a secure
sandbox (Monty) to answer complex questions. The sandbox has access to
`fetch()` for HTTP requests, `json_parse()`/`json_dumps()` for JSON,
and `web_search()` for DuckDuckGo search.

Usage:
    uv run python monty_agent_example.py run

Try asking:
    - "What is 2 to the power of 100?"
    - "Fetch the top story from Hacker News and tell me about it"
    - "What's the current Bitcoin price in USD?"
    - "Search for the latest James Webb telescope discoveries"
"""

import logging
from typing import Any

import pydantic_monty
from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.plugins import gemini, getstream

from instructions import AGENT_INSTRUCTIONS, TOOL_DESCRIPTION
from sandbox import EXTERNAL_FUNCTIONS, TYPE_STUBS, DuckDuckGoLite, set_search_engine

logger = logging.getLogger(__name__)

load_dotenv()

set_search_engine(DuckDuckGoLite())

# Gemini closes the WebSocket (1008/1011) if the tool response is too large.
MAX_TOOL_RESPONSE_SIZE = 10_000


async def run_python_code(python_code: str = "") -> str:
    """Execute Python code in the Monty sandbox."""
    logger.warning(f"Executing code:\n{python_code}")

    m = pydantic_monty.Monty(
        python_code,
        script_name="agent_code.py",
        type_check=True,
        type_check_stubs=TYPE_STUBS,
    )

    stdout_parts: list[str] = []

    def capture_print(stream: str, text: str) -> None:
        stdout_parts.append(text)

    output = await pydantic_monty.run_monty_async(
        m,
        external_functions=EXTERNAL_FUNCTIONS,
        print_callback=capture_print,
    )

    stdout = "".join(stdout_parts)
    result = stdout if stdout else str(output)

    if len(result) > MAX_TOOL_RESPONSE_SIZE:
        result = result[:max_result_size] + "\n... (truncated)"

    logger.warning(f"Code result: {result}")
    return result


def setup_llm() -> gemini.Realtime:
    llm = gemini.Realtime()
    llm.register_function(description=TOOL_DESCRIPTION)(run_python_code)
    return llm


async def create_agent(**kwargs: Any) -> Agent:
    llm = setup_llm()

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Monty Agent", id="agent"),
        instructions=AGENT_INSTRUCTIONS,
        llm=llm,
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs: Any) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.llm.simple_response("Say hi and ask what the user needs help with.")
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
