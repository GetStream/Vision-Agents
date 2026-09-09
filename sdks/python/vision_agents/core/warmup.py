import abc
import asyncio
from typing import Any, Generic, Type, TypeVar

from vision_agents.core.utils.utils import await_or_run

__all__ = (
    "Warmable",
    "WarmupCache",
)

T = TypeVar("T")


class WarmupCache:
    """
    A cache to keep track of resources being loaded by Warmable objects.

    Currently, the cache key is the actual class of the Warmable object.
    """

    def __init__(self):
        self._cache: dict[Type["Warmable"], Any] = {}
        self._locks: dict[Type["Warmable"], asyncio.Lock] = {}

    async def warmup(self, warmable: "Warmable"):
        warmable_cls = type(warmable)
        if (resource := self._cache.get(warmable_cls)) is not None:
            # The resource is already loaded.
            # Set it to the warmable instance and exit
            warmable.on_warmed_up(resource)
            return

        # When the resource is not loaded yet, use the lock to avoid loading it multiple times in parallel.
        lock = self._locks.setdefault(warmable_cls, asyncio.Lock())
        async with lock:
            # Check if the resource was loaded while we were waiting for the lock
            if (resource := self._cache.get(warmable_cls)) is None:
                # Load the resource by triggering `warmable.on_warmup()`
                resource = await warmable.on_warmup()
                # Store the result
                self._cache[warmable_cls] = resource
            # Set the resource back to the warmable instance.
            await await_or_run(warmable.on_warmed_up, resource)


class Warmable(abc.ABC, Generic[T]):
    """
    A base class for components that need to do some expensive resource loading before being used,
    like VAD plugins, YOLO-based processors, etc.

    It delegates storage of these resources to `WarmupCache`, so they can be re-used.

    Methods to implement:

    - `on_warmup() -> T` - must load the resource and return it so `WarmupCache` can store it.
        Avoid mutating the instance in this method.
    - `on_warmed_up(resource: T)` - must set the resource back to the instance.
        The resource is taken from the `WarmupCache` instance provided via `.warmup(cache: WarmupCache)` method.


    Example:
        class SomeWarmable(Warmable[dict]):
            def __init__(self):
                self._resource: dict | None = None

            async def on_warmup(self):
                # Do some loading work here
                resource = {}
                return resource

            def on_warmed_up(self, resource: dict) -> None:
                # Set the resource back to the instance.
                self._resource = resource
    """

    @abc.abstractmethod
    async def on_warmup(self) -> T:
        """
        A method to load required resources (e.g., download models) before performing any work.

        It's called once during start up of the application, and it must return the loaded resource.
        """
        ...

    @abc.abstractmethod
    def on_warmed_up(self, resource: T) -> None:
        """
        A method to set the loaded resource back to the object instance.
        This method is called every time an Agent starts.

        Args:
            resource:

        Returns:
        """

    async def warmup(self, cache: WarmupCache | None = None) -> None:
        """
        Perform the actual loading if it's not done yet based on the passed `cache` object.

        It's safe to call this method multiple times with the same `cache` instance.
        If `cache` is None, the loaded resources won't be cached between calls.
        """
        # If the class is already loaded
        cache = cache or WarmupCache()
        await cache.warmup(warmable=self)
