import asyncio

import pytest
import redis.asyncio as redis
from testcontainers.redis import RedisContainer
from vision_agents.core.agents.session_registry.redis_store import RedisSessionKVStore


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
        await asyncio.sleep(2)
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
