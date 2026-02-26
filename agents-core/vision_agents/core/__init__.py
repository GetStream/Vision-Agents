from vision_agents.core.agents import Agent
from vision_agents.core.agents.agent_launcher import AgentLauncher, AgentSession
from vision_agents.core.agents.session_registry import (
    InMemorySessionKVStore,
    SessionInfo,
    SessionKVStore,
    SessionRegistry,
)
from vision_agents.core.edge.types import User
from vision_agents.core.runner import Runner, ServeOptions

__all__ = [
    "Agent",
    "AgentLauncher",
    "AgentSession",
    "InMemorySessionKVStore",
    "Runner",
    "ServeOptions",
    "SessionInfo",
    "SessionKVStore",
    "SessionRegistry",
    "User",
]

try:
    from vision_agents.core.agents.session_registry import RedisSessionKVStore

    __all__ += ["RedisSessionKVStore"]
except ModuleNotFoundError as exc:
    # Only swallow a missing `redis` package; re-raise anything else
    # so real import errors in redis_store.py surface immediately.
    if not exc.name or not exc.name.startswith("redis"):
        raise
