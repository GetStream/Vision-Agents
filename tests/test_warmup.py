import asyncio
import random

from vision_agents.core.warmup import Warmable, WarmupCache


class Resource: ...


class DummyWarmable(Warmable[Resource]):
    def __init__(self):
        self.on_warmup_calls_count: int = 0
        self.on_warmed_up_calls_count: int = 0
        self.resource = None

    async def on_warmup(self):
        self.on_warmup_calls_count += 1
        # Sleep random time to simulate some heavy background work
        await asyncio.sleep(random.random())
        return Resource()

    def on_warmed_up(self, resource) -> None:
        self.on_warmed_up_calls_count += 1
        self.resource = resource


class TestWarmable:
    async def test_warmup_cached(self):
        cache = WarmupCache()
        warmable = DummyWarmable()
        await warmable.warmup(cache)
        resource = warmable.resource
        assert resource is not None
        assert warmable.on_warmup_calls_count == 1
        assert warmable.on_warmed_up_calls_count == 1

        # Trigger load() again with the same cache
        await warmable.warmup(cache)
        # Verify that "resource" is the same
        assert warmable.resource is resource
        # on_warmup() must be called only once
        assert warmable.on_warmup_calls_count == 1
        # on_warmed_up() sets the resource to the instance and must be called again
        assert warmable.on_warmed_up_calls_count == 2

    async def test_warmup_no_cache(self):
        warmable = DummyWarmable()
        await warmable.warmup()
        resource = warmable.resource
        assert resource is not None
        assert warmable.on_warmup_calls_count == 1
        assert warmable.on_warmed_up_calls_count == 1

        # Trigger load() again without cache
        await warmable.warmup()
        # Verify that the "resource" is not the same
        assert warmable.resource is not None
        assert warmable.resource is not resource
        # on_warmup() must be called again
        assert warmable.on_warmup_calls_count == 2
        # on_warmed_up() must be called again too
        assert warmable.on_warmed_up_calls_count == 2

    async def test_warmup_concurrently(self):
        cache = WarmupCache()

        # Create multiple instances of the same type and load all of them
        warmable1 = DummyWarmable()
        warmable2 = DummyWarmable()
        warmable3 = DummyWarmable()
        await asyncio.gather(
            warmable1.warmup(cache),
            warmable2.warmup(cache),
            warmable3.warmup(cache),
        )

        # Verify that the same resource is shared by all instances
        resource = warmable1.resource
        assert resource is not None
        assert warmable2.resource is resource
        assert warmable3.resource is resource

        # Verify that on_warmup() was called only for the first instance
        assert warmable1.on_warmup_calls_count == 1
        assert warmable2.on_warmup_calls_count == 0
        assert warmable3.on_warmup_calls_count == 0
