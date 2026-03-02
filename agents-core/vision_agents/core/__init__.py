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
except ImportError as e:
    import warnings

    if e.name and e.name.startswith("redis") or "RedisSessionKVStore" in str(e):
        warnings.warn(
            "Optional dependency 'redis' is not installed. "
            "Install the [redis] extra to enable RedisSessionKVStore.",
            stacklevel=2,
        )
    else:
        raise
