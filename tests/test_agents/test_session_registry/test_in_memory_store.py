import asyncio

import pytest
from vision_agents.core.agents.session_registry.in_memory_store import (
    InMemorySessionKVStore,
)


@pytest.fixture()
async def store():
    s = InMemorySessionKVStore()
    await s.start()
    try:
        yield s
    finally:
        await s.close()


class TestInMemorySessionKVStore:
    async def test_set_and_get(self, store: InMemorySessionKVStore) -> None:
        await store.set("k1", b"hello", ttl=10.0)
        assert await store.get("k1") == b"hello"

    async def test_get_missing_key(self, store: InMemorySessionKVStore) -> None:
        assert await store.get("nonexistent") is None

    async def test_set_overwrites(self, store: InMemorySessionKVStore) -> None:
        await store.set("k1", b"first", ttl=10.0)
        await store.set("k1", b"second", ttl=10.0)
        assert await store.get("k1") == b"second"

    async def test_ttl_expiry(self, store: InMemorySessionKVStore) -> None:
        await store.set("ephemeral", b"bye", ttl=0.5)
        await asyncio.sleep(0.7)
        assert await store.get("ephemeral") is None

    async def test_mset_and_mget(self, store: InMemorySessionKVStore) -> None:
        await store.mset(
            [
                ("a", b"1", 10.0),
                ("b", b"2", 10.0),
                ("c", b"3", 10.0),
            ]
        )
        result = await store.mget(["a", "b", "c"])
        assert result == [b"1", b"2", b"3"]

    async def test_mget_partial_missing(self, store: InMemorySessionKVStore) -> None:
        await store.mset([("x", b"1", 10.0), ("y", b"2", 10.0)])
        result = await store.mget(["x", "y", "z"])
        assert result == [b"1", b"2", None]

    async def test_mget_empty(self, store: InMemorySessionKVStore) -> None:
        assert await store.mget([]) == []

    async def test_expire_refreshes_ttl(self, store: InMemorySessionKVStore) -> None:
        await store.set("refresh_me", b"val", ttl=1.0)
        await asyncio.sleep(0.5)
        await store.expire("refresh_me", ttl=2.0)
        await asyncio.sleep(1.0)
        assert await store.get("refresh_me") == b"val"

    async def test_expire_nonexistent_key(self, store: InMemorySessionKVStore) -> None:
        await store.expire("ghost", ttl=5.0)

    async def test_expire_multiple_keys(self, store: InMemorySessionKVStore) -> None:
        await store.mset([("m1", b"a", 1.0), ("m2", b"b", 1.0)])
        await asyncio.sleep(0.5)
        await store.expire("m1", "m2", ttl=2.0)
        await asyncio.sleep(1.0)
        assert await store.get("m1") == b"a"
        assert await store.get("m2") == b"b"

    async def test_keys_with_prefix(self, store: InMemorySessionKVStore) -> None:
        await store.mset(
            [
                ("sessions/s1", b"a", 10.0),
                ("sessions/s2", b"b", 10.0),
                ("other/x", b"c", 10.0),
            ]
        )
        matched = await store.keys("sessions/")
        assert sorted(matched) == ["sessions/s1", "sessions/s2"]

    async def test_delete(self, store: InMemorySessionKVStore) -> None:
        await store.mset([("d1", b"a", 10.0), ("d2", b"b", 10.0)])
        await store.delete(["d1"])
        assert await store.get("d1") is None
        assert await store.get("d2") == b"b"

    async def test_delete_nonexistent(self, store: InMemorySessionKVStore) -> None:
        await store.delete(["does_not_exist"])

    async def test_delete_empty(self, store: InMemorySessionKVStore) -> None:
        await store.delete([])
