from .in_memory_store import InMemorySessionKVStore as InMemorySessionKVStore
from .registry import SessionRegistry as SessionRegistry
from .store import SessionKVStore as SessionKVStore
from .types import SessionInfo as SessionInfo

__all__ = [
    "InMemorySessionKVStore",
    "SessionInfo",
    "SessionKVStore",
    "SessionRegistry",
]

try:
    from .redis_store import RedisSessionKVStore as RedisSessionKVStore

    __all__ += ["RedisSessionKVStore"]
except ModuleNotFoundError as exc:
    # Only swallow a missing `redis` package; re-raise anything else
    # so real import errors in redis_store.py surface immediately.
    if not exc.name or not exc.name.startswith("redis"):
        raise
