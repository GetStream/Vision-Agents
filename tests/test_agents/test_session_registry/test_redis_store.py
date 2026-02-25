import asyncio

import pytest
import redis.asyncio as redis
from testcontainers.redis import RedisContainer
from vision_agents.core.agents.session_registry import SessionRegistry
from vision_agents.core.agents.session_registry.redis_store import (
    RedisSessionKVStore,
)


@pytest.fixture(scope="module")
def redis_url():
    with RedisContainer() as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"


@pytest.fixture()
async def redis_store(redis_url):
    client = redis.from_url(redis_url)
    store = RedisSessionKVStore(client=client, key_prefix="test:")
    await store.start()
    try:
        yield store
    finally:
        keys = await store.keys("")
        if keys:
            await store.delete(keys)
        await client.aclose()


@pytest.fixture()
async def registry(redis_store):
    reg = SessionRegistry(store=redis_store, ttl=5.0)
    await reg.start()
    try:
        yield reg
    finally:
        await reg.stop()


class TestRedisSessionKVStore:
    async def test_set_and_get(self, redis_store: RedisSessionKVStore) -> None:
        await redis_store.set("k1", b"hello", ttl=10.0)
        assert await redis_store.get("k1") == b"hello"

    async def test_get_missing_key(self, redis_store: RedisSessionKVStore) -> None:
        assert await redis_store.get("nonexistent") is None

    async def test_set_overwrites(self, redis_store: RedisSessionKVStore) -> None:
        await redis_store.set("k1", b"first", ttl=10.0)
        await redis_store.set("k1", b"second", ttl=10.0)
        assert await redis_store.get("k1") == b"second"

    async def test_ttl_expiry(self, redis_store: RedisSessionKVStore) -> None:
        await redis_store.set("ephemeral", b"bye", ttl=0.5)
        await asyncio.sleep(0.7)
        assert await redis_store.get("ephemeral") is None

    async def test_mset_and_mget(self, redis_store: RedisSessionKVStore) -> None:
        await redis_store.mset(
            [
                ("a", b"1", 10.0),
                ("b", b"2", 10.0),
                ("c", b"3", 10.0),
            ]
        )
        result = await redis_store.mget(["a", "b", "c"])
        assert result == [b"1", b"2", b"3"]

    async def test_mget_partial_missing(self, redis_store: RedisSessionKVStore) -> None:
        await redis_store.mset([("x", b"1", 10.0), ("y", b"2", 10.0)])
        result = await redis_store.mget(["x", "y", "z"])
        assert result == [b"1", b"2", None]

    async def test_mget_empty(self, redis_store: RedisSessionKVStore) -> None:
        assert await redis_store.mget([]) == []

    async def test_expire_refreshes_ttl(self, redis_store: RedisSessionKVStore) -> None:
        await redis_store.set("refresh_me", b"val", ttl=1.0)
        await asyncio.sleep(0.5)
        await redis_store.expire("refresh_me", ttl=2.0)
        await asyncio.sleep(1.0)
        assert await redis_store.get("refresh_me") == b"val"

    async def test_expire_nonexistent_key(
        self, redis_store: RedisSessionKVStore
    ) -> None:
        await redis_store.expire("ghost", ttl=5.0)

    async def test_expire_multiple_keys(self, redis_store: RedisSessionKVStore) -> None:
        await redis_store.mset([("m1", b"a", 1.0), ("m2", b"b", 1.0)])
        await asyncio.sleep(0.5)
        await redis_store.expire("m1", "m2", ttl=2.0)
        await asyncio.sleep(1.0)
        assert await redis_store.get("m1") == b"a"
        assert await redis_store.get("m2") == b"b"

    async def test_keys_with_prefix(self, redis_store: RedisSessionKVStore) -> None:
        await redis_store.mset(
            [
                ("sessions/s1", b"a", 10.0),
                ("sessions/s2", b"b", 10.0),
                ("other/x", b"c", 10.0),
            ]
        )
        matched = await redis_store.keys("sessions/")
        assert sorted(matched) == ["sessions/s1", "sessions/s2"]

    async def test_delete(self, redis_store: RedisSessionKVStore) -> None:
        await redis_store.mset([("d1", b"a", 10.0), ("d2", b"b", 10.0)])
        await redis_store.delete(["d1"])
        assert await redis_store.get("d1") is None
        assert await redis_store.get("d2") == b"b"

    async def test_delete_nonexistent(self, redis_store: RedisSessionKVStore) -> None:
        await redis_store.delete(["does_not_exist"])

    async def test_delete_empty(self, redis_store: RedisSessionKVStore) -> None:
        await redis_store.delete([])

    async def test_publish_subscribe(self, redis_store: RedisSessionKVStore) -> None:
        received: list[bytes] = []

        async def listener():
            async for msg in redis_store.subscribe("chan"):
                received.append(msg)
                break

        task = asyncio.create_task(listener())
        await asyncio.sleep(0.1)
        await redis_store.publish("chan", b"ping")
        await asyncio.wait_for(task, timeout=2.0)
        assert received == [b"ping"]

    async def test_publish_no_subscriber(
        self, redis_store: RedisSessionKVStore
    ) -> None:
        await redis_store.publish("nobody_listening", b"hello")


class TestSessionRegistryWithRedis:
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
