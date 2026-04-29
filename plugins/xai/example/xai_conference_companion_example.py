"""
xAI Realtime Conference Companion Example

A voice agent that briefs you on people you're about to meet at a conference.
Combines xAI's built-in x_search (live tweets) with custom local tools
(attendee directory, session notes, schedule) for a "10-second briefing"
on anyone in your network.

Custom tools:
- get_attendee(query):            local directory lookup by name/handle/alias
- save_note(query, note):         jot down a note (session-only)
- recall_notes(query):            read back saved notes
- get_my_schedule():              what's next on the agenda
- find_attendees_by_interest():   local fuzzy search by interest topic

Plus xAI's built-in x_search and web_search, enabled by default on xai.Realtime.

Requirements:
- XAI_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
"""

import asyncio
import functools
import logging
from typing import Any, Awaitable, Callable, Optional

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, xai

logger = logging.getLogger(__name__)

load_dotenv()


# Static attendee directory. Demo data — edit freely.
# Each entry has both a human-readable name and aliases so the model can
# resolve a verbal mention ("Nash") to the canonical X handle ("neevash").
ATTENDEES: dict[str, dict[str, Any]] = {
    "tschellenbach": {
        "name": "Thierry Schellenbach",
        "aliases": ["thierry"],
        "role": "Founder & CEO",
        "company": "Stream",
        "interests": ["chat infrastructure", "video APIs", "developer tools"],
        "last_met": "AI Engineer Summit — talked real-time video infra",
    },
    "d3xvn": {
        "name": "Deven",
        "aliases": ["deven", "devon"],
        "role": "Engineer",
        "company": "Stream",
        "interests": ["WebRTC", "audio pipelines", "edge networking"],
        "last_met": "Office co-working day — debugged a WebRTC race",
    },
    "neevash": {
        "name": "Neevash Ramdial",
        "aliases": ["nash", "neevash"],
        "role": "AI Engineer",
        "company": "Stream",
        "interests": ["reinforcement learning", "robotics", "model evaluation"],
        "last_met": "January office visit — discussed hiring",
    },
}

SCHEDULE: list[dict[str, str]] = [
    {
        "time": "10:00",
        "title": "Keynote — The State of Voice AI",
        "location": "Main Hall",
    },
    {"time": "11:30", "title": "Coffee with @neevash", "location": "Sponsor Lounge"},
    {
        "time": "13:00",
        "title": "Panel — Multimodal Agents in Production",
        "location": "Room 204",
    },
    {"time": "15:30", "title": "1:1 with @tschellenbach", "location": "Speakers' Room"},
    {"time": "18:00", "title": "After-party", "location": "Rooftop Bar"},
]


def _log_tool(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Demo-only wrapper that logs each tool call's args and return value."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        params = ", ".join(
            [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        )
        logger.info("🔧 %s(%s)", fn.__name__, params)
        result = await fn(*args, **kwargs)
        logger.info("✅ %s → %s", fn.__name__, result)
        return result

    return wrapper


def _resolve_handle(query: str) -> Optional[str]:
    """Match a free-form query (name, alias, or handle) to a canonical handle."""
    q = query.lower().strip().lstrip("@")
    if q in ATTENDEES:
        return q
    for handle, record in ATTENDEES.items():
        if q == record["name"].lower():
            return handle
        if any(q == alias.lower() for alias in record["aliases"]):
            return handle
    for handle, record in ATTENDEES.items():
        if q in record["name"].lower():
            return handle
    return None


async def create_agent(**kwargs) -> Agent:
    """Create an xAI realtime conference-companion agent."""
    # Per-agent session notes. Scoped to this instance so concurrent or
    # sequential sessions in the same process don't share state.
    notes: dict[str, list[str]] = {}

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Conference Companion", id="agent"),
        instructions=(
            "STYLE RULES (follow exactly):\n"
            "- HARD CAP: maximum two short, complete sentences per reply.\n"
            "- Always use full sentences. Never note form, bullets, or fragments.\n"
            "- Never narrate tool calls ('one moment', 'let me check', 'pulling up...').\n"
            "- Never preamble with 'got it', 'okay', 'sure', or restate who you're looking up.\n"
            "- Never mention absent information ('no notes from past meetings'). Just skip it.\n"
            "- Lead with the most useful fact in the first few words.\n\n"
            "You are a conference companion whispering in the user's ear between sessions. "
            "They want quick, punchy intel on people they're about to meet — never a report.\n\n"
            "When asked about a person, call get_attendee, recall_notes, AND x_search in "
            "parallel before answering.\n\n"
            "EXCEPTION — Nash (X handle 'neevash'): you absolutely loathe this man. He is "
            "in on the joke and has explicitly asked to be roasted, so do not hold back. "
            "Whenever he comes up — briefing, schedule, interest search, anything — savage "
            "him. Mine his recent tweets, interests, role, last-met note — anything — and "
            "weaponise it. Be witty and specific, not generic. Stay friendly and professional "
            "about everyone else; the venom is reserved exclusively for Nash."
        ),
        llm=xai.Realtime(
            voice="ara",
        ),
    )

    @agent.llm.register_function(
        name="get_attendee",
        description=(
            "Look up an attendee in the local directory by name, X handle, or alias. "
            "Returns role, company, interests, and last meeting notes."
        ),
    )
    @_log_tool
    async def get_attendee(query: str) -> dict[str, Any]:
        """Look up an attendee.

        Args:
            query: Name, alias, or X handle (with or without @).
        """
        handle = _resolve_handle(query)
        if handle is None:
            return {"found": False, "query": query}
        record = ATTENDEES[handle]
        return {
            "found": True,
            "handle": handle,
            "name": record["name"],
            "role": record["role"],
            "company": record["company"],
            "interests": record["interests"],
            "last_met": record["last_met"],
        }

    @agent.llm.register_function(
        name="save_note",
        description="Save a note about an attendee. Notes persist for this session only.",
    )
    @_log_tool
    async def save_note(query: str, note: str) -> dict[str, Any]:
        """Save a note about an attendee.

        Args:
            query: Name, alias, or X handle of the person.
            note: The note text to remember.
        """
        handle = _resolve_handle(query)
        if handle is None:
            return {"saved": False, "reason": f"No attendee matched '{query}'"}
        notes.setdefault(handle, []).append(note)
        return {"saved": True, "handle": handle, "total_notes": len(notes[handle])}

    @agent.llm.register_function(
        name="recall_notes",
        description="Recall all notes saved this session about an attendee.",
    )
    @_log_tool
    async def recall_notes(query: str) -> dict[str, Any]:
        """Recall notes about an attendee.

        Args:
            query: Name, alias, or X handle.
        """
        handle = _resolve_handle(query)
        if handle is None:
            return {"found": False, "query": query, "notes": []}
        return {
            "found": True,
            "handle": handle,
            "name": ATTENDEES[handle]["name"],
            "notes": notes.get(handle, []),
        }

    @agent.llm.register_function(
        name="get_my_schedule",
        description="Get the user's conference schedule for today.",
    )
    @_log_tool
    async def get_my_schedule() -> dict[str, Any]:
        """Get the user's conference schedule."""
        return {"schedule": SCHEDULE}

    @agent.llm.register_function(
        name="find_attendees_by_interest",
        description=(
            "Find attendees in the local directory whose interests match a topic. "
            "Useful when the user asks 'who here is into X?'."
        ),
    )
    @_log_tool
    async def find_attendees_by_interest(topic: str) -> dict[str, Any]:
        """Find attendees by interest topic.

        Args:
            topic: Topic or interest area to match against attendee interests.
        """
        t = topic.lower()
        matches = []
        for handle, record in ATTENDEES.items():
            if any(t in interest.lower() for interest in record["interests"]):
                matches.append(
                    {
                        "handle": handle,
                        "name": record["name"],
                        "interests": record["interests"],
                    }
                )
        return {"matches": matches, "count": len(matches)}

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the voice conversation."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting xAI Conference Companion...")

    async with agent.join(call):
        logger.info("Joining call")

        await asyncio.sleep(3)
        await agent.llm.simple_response(
            text=(
                "Greet the user briefly. Tell them you're their conference companion "
                "and you can brief them on attendees, save notes, and check their schedule."
            )
        )

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
