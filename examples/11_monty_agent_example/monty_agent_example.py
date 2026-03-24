"""
Monty Agent Example — Gemini Realtime + sandboxed Python execution.

A voice AI agent that can write and execute Python code in a secure
sandbox (Monty) to answer complex questions. The sandbox has access to
`fetch()` for HTTP requests, `json_parse()`/`json_dumps()` for JSON,
and `web_search()` for web search.

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
from vision_agents.core.llm.events import (
    RealtimeAgentSpeechTranscriptionEvent,
    RealtimeUserSpeechTranscriptionEvent,
)
from vision_agents.plugins import gemini, getstream

from display import TranscriptAccumulator, log_code, log_result, thinking_spinner
from instructions import AGENT_INSTRUCTIONS, TOOL_DESCRIPTION
from sandbox import EXTERNAL_FUNCTIONS, TYPE_STUBS, SearXNG, set_search_engine

logger = logging.getLogger(__name__)

load_dotenv()

NOISY_LOGGERS = [
    "vision_agents.plugins.getstream",
    "vision_agents.plugins.gemini",
    "vision_agents.core.agents",
    "httpx",
    "httpcore",
]


def configure_logging(verbosity: int = 0) -> None:
    """Set log levels based on verbosity. Called before agent starts.

    -v   → INFO (noisy loggers stay WARNING)
    -vv  → INFO everywhere
    -vvv → DEBUG everywhere
    """
    if verbosity >= 3:
        level = logging.DEBUG
    elif verbosity >= 2:
        level = logging.INFO
    else:
        level = logging.WARNING

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)

set_search_engine(SearXNG())

# Gemini closes the WebSocket (1008/1011) if the tool response is too large.
MAX_TOOL_RESPONSE_SIZE = 10_000


async def run_python_code(python_code: str = "") -> str:
    """Execute Python code in the Monty sandbox."""
    log_code(python_code)

    m = pydantic_monty.Monty(
        python_code,
        script_name="agent_code.py",
        type_check=True,
        type_check_stubs=TYPE_STUBS,
    )

    stdout_parts: list[str] = []

    def capture_print(stream: str, text: str) -> None:
        stdout_parts.append(text)

    with thinking_spinner("Running code"):
        output = await pydantic_monty.run_monty_async(
            m,
            external_functions=EXTERNAL_FUNCTIONS,
            print_callback=capture_print,
        )

    stdout = "".join(stdout_parts)
    result = stdout if stdout else str(output)

    if len(result) > MAX_TOOL_RESPONSE_SIZE:
        result = result[:MAX_TOOL_RESPONSE_SIZE] + "\n... (truncated)"

    log_result(result)
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

    user_acc = TranscriptAccumulator("User")
    agent_acc = TranscriptAccumulator("Agent")

    @agent.events.subscribe
    async def on_user_transcript(event: RealtimeUserSpeechTranscriptionEvent) -> None:
        user_acc.push(event.text, event.mode)

    @agent.events.subscribe
    async def on_agent_transcript(event: RealtimeAgentSpeechTranscriptionEvent) -> None:
        agent_acc.push(event.text, event.mode)

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs: Any) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        # Get the user's name from the first participant that joins
        user_name = None
        if agent._connection is not None:
            participants = agent._connection.participants.get_participants()
            for p in participants:
                if p.user_id != agent.agent_user.id:
                    user_name = p.name or p.user_id.replace("-", " ").title()
                    break

        if user_name:
            greeting = f"Say hi to {user_name} by name and ask what they need help with."
        else:
            greeting = "Say hi and ask what the user needs help with."

        await agent.llm.simple_response(greeting)
        await agent.finish()


if __name__ == "__main__":
    import sys

    # Parse -v flags before Runner takes over
    verbosity = sys.argv.count("-v") + sys.argv.count("-vv") * 2 + sys.argv.count("-vvv") * 3
    sys.argv = [a for a in sys.argv if a not in ("-v", "-vv", "-vvv")]
    configure_logging(verbosity)

    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
