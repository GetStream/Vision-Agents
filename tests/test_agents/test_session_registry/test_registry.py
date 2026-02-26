import asyncio

import pytest
import redis.asyncio as redis
from testcontainers.redis import RedisContainer
from vision_agents.core.agents.session_registry import SessionKVStore, SessionRegistry
from vision_agents.core.agents.session_registry.in_memory_store import (
    InMemorySessionKVStore,
)
from vision_agents.core.agents.session_registry.redis_store import RedisSessionKVStore


@pytest.fixture(scope="module")
def redis_url():
    with RedisContainer() as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"


@pytest.fixture()
async def in_memory_store():
    yield InMemorySessionKVStore()


@pytest.fixture()
async def redis_store(redis_url):
    client = redis.from_url(redis_url)
    store = RedisSessionKVStore(client=client, key_prefix="test_reg:")
    try:
        yield store
    finally:
        keys = await store.keys("")
        if keys:
            await store.delete(keys)
        await client.aclose()


@pytest.fixture(params=["in_memory", "redis"])
async def registry(request, in_memory_store, redis_store):
    if request.param == "in_memory":
        store: SessionKVStore = in_memory_store
    elif request.param == "redis":
        store: SessionKVStore = redis_store
    else:
        raise ValueError(f"Invalid param {request.param}")

    reg = SessionRegistry(store=store, ttl=5.0)
    await reg.start()
    try:
        yield reg
    finally:
        await reg.stop()


class TestSessionRegistry:
    async def test_register_and_get(self, registry: SessionRegistry) -> None:
        await registry.register("sess-1", "call-1")
        info = await registry.get("sess-1")
        assert info is not None
        assert info.session_id == "sess-1"
        assert info.call_id == "call-1"
        assert info.node_id == registry.node_id

    async def test_get_for_call(self, registry: SessionRegistry) -> None:
        await registry.register("s1", "call-multi")
        await registry.register("s2", "call-multi")
        sessions = await registry.get_for_call("call-multi")
        session_ids = {s.session_id for s in sessions}
        assert session_ids == {"s1", "s2"}

    async def test_remove(self, registry: SessionRegistry) -> None:
        await registry.register("to-remove", "call-r")
        await registry.remove("to-remove")
        assert await registry.get("to-remove") is None

    async def test_refresh_extends_ttl(self, registry: SessionRegistry) -> None:
        await registry.register("sess-r", "call-r")
        await asyncio.sleep(3.0)
        await registry.refresh({"sess-r": "call-r"})
        await asyncio.sleep(3.0)
        info = await registry.get("sess-r")
        assert info is not None

    async def test_request_close_and_get_close_requests(
        self, registry: SessionRegistry
    ) -> None:
        await registry.register("sess-close", "call-c")
        await registry.request_close("sess-close")
        flagged = await registry.get_close_requests(["sess-close", "other"])
        assert flagged == ["sess-close"]

    async def test_update_metrics(self, registry: SessionRegistry) -> None:
        await registry.register("sess-m", "call-m")
        await registry.update_metrics("sess-m", {"latency_ms": 42.0})
        info = await registry.get("sess-m")
        assert info is not None
        assert info.metrics["latency_ms"] == 42.0

    async def test_session_expires_without_refresh(
        self, registry: SessionRegistry
    ) -> None:
        short_registry = SessionRegistry(store=registry._store, ttl=1.0)
        await short_registry.register("sess-expire", "call-e")
        await asyncio.sleep(1.5)
        assert await short_registry.get("sess-expire") is None
