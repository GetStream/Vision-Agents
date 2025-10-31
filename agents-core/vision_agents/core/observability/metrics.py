from __future__ import annotations

import functools
import inspect
from typing import Dict, Any, Optional, Mapping, Callable, Awaitable, TypeVar, Union

from opentelemetry import trace, metrics
from opentelemetry.metrics import Histogram
import time

R = TypeVar("R")

# Get tracer and meter using the library name
# These will use whatever providers the application has configured
# If no providers are configured, they will be no-ops
tracer = trace.get_tracer("vision_agents.core")
meter = metrics.get_meter("vision_agents.core")

stt_latency_ms = meter.create_histogram(
    "stt.latency.ms", unit="ms", description="Total STT latency"
)
stt_first_byte_ms = meter.create_histogram(
    "stt.first_byte.ms", unit="ms", description="STT time to first token/byte"
)
stt_bytes_streamed = meter.create_counter(
    "stt.bytes.streamed", unit="By", description="Bytes received from STT"
)
stt_errors = meter.create_counter("stt.errors", description="STT errors")

tts_latency_ms = meter.create_histogram(
    "tts.latency.ms", unit="ms", description="Total TTS latency"
)
tts_errors = meter.create_counter("tts.errors", description="TTS errors")
tts_events_emitted = meter.create_counter(
    "tts.events.emitted", description="Number of TTS events emitted"
)

inflight_ops = meter.create_up_down_counter(
    "voice.ops.inflight", description="Inflight voice ops"
)

turn_detection_latency_ms = meter.create_histogram(
    "turn.detection.latency.ms",
    unit="ms",
)


class Timer:
    """
    Can be used as:
        done = Timer(hist, {"attr": 1})
        ...
        done({"phase": "init"})

        with Timer(hist, {"attr": 1}) as timer:
            timer.attributes["dynamic_key"] = "dynamic_value"
            ...

        @Timer(hist, {"route": "/join"})
        def handler(...): ...

        @Timer(hist)
        async def async_handler(...): ...

    If decorating a method, automatically adds {"class": <cls.__name__>} to attributes.

    When used as a context manager, you can add attributes dynamically via the
    `attributes` property, which will be merged with base attributes when recording.
    """

    def __init__(
        self,
        hist: Histogram,
        attributes: Optional[Mapping[str, Any]] = None,
        *,
        unit: str = "ms",
        record_exceptions: bool = True,
    ) -> None:
        self._hist = hist
        self._base_attrs: Dict[str, Any] = dict(attributes or {})
        self._unit = unit
        self._record_exceptions = record_exceptions

        self._start_ns = time.perf_counter_ns()
        self._stopped = False
        self.last_elapsed_ms: Optional[float] = None

        # Public attributes dictionary that can be modified during context manager usage
        self.attributes: Dict[str, Any] = {}

    def __call__(self, *args, **kwargs):
        """If called with a function, act as a decorator; else record."""
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            func = args[0]
            return self._decorate(func)
        extra_attrs = args[0] if args else None
        return self.stop(extra_attrs)

    def __enter__(self) -> "Timer":
        self._restart()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        attrs: Dict[str, Any] = {}
        if self._record_exceptions:
            attrs["exception"] = "true" if exc_type else "false"
            if exc_type:
                attrs["exception_type"] = getattr(exc_type, "__name__", str(exc_type))
        self.stop(attrs)

    def stop(self, extra_attributes: Optional[Mapping[str, Any]] = None) -> float:
        """Idempotent: records only once per start."""
        if not self._stopped:
            self._stopped = True
            elapsed = self.elapsed_ms()
            self.last_elapsed_ms = elapsed

            attrs = {**self._base_attrs}
            # Merge the dynamic attributes set during context manager usage
            attrs.update(self.attributes)
            if extra_attributes:
                attrs.update(dict(extra_attributes))

            value = elapsed if self._unit == "ms" else elapsed / 1000.0
            self._hist.record(value, attributes=attrs)

        return self.last_elapsed_ms or 0.0

    def elapsed_ms(self) -> float:
        return (time.perf_counter_ns() - self._start_ns) / 1_000_000.0

    def _restart(self) -> None:
        self._start_ns = time.perf_counter_ns()
        self._stopped = False
        self.last_elapsed_ms = None
        self.attributes = {}  # Reset dynamic attributes on restart

    def _decorate(
        self, func: Union[Callable[..., R], Callable[..., Awaitable[R]]]
    ) -> Union[Callable[..., R], Callable[..., Awaitable[R]]]:
        """
        Decorate a function or method.
        Automatically adds {"class": <ClassName>} if decorating a bound method.
        """

        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            # Type-cast func as async for type checker
            async_func: Callable[..., Awaitable[R]] = func  # type: ignore[assignment]

            @functools.wraps(async_func)
            async def async_wrapper(*args, **kwargs) -> R:
                class_name = _get_class_name_from_args(async_func, args)
                attrs = {**self._base_attrs}
                if class_name:
                    attrs["class"] = class_name
                with Timer(
                    self._hist,
                    attrs,
                    unit=self._unit,
                    record_exceptions=self._record_exceptions,
                ):
                    return await async_func(*args, **kwargs)

            return async_wrapper
        else:
            # Type-cast func as sync for type checker
            sync_func: Callable[..., R] = func  # type: ignore[assignment]

            @functools.wraps(sync_func)
            def sync_wrapper(*args, **kwargs) -> R:
                class_name = _get_class_name_from_args(sync_func, args)
                attrs = {**self._base_attrs}
                if class_name:
                    attrs["class"] = class_name
                with Timer(
                    self._hist,
                    attrs,
                    unit=self._unit,
                    record_exceptions=self._record_exceptions,
                ):
                    return sync_func(*args, **kwargs)

            return sync_wrapper


def _get_class_name_from_args(
    func: Callable[..., Any], args: tuple[Any, ...]
) -> Optional[str]:
    """Return fully qualified class path if first arg looks like a bound method (self or cls).

    For instance methods (self), we return the runtime class path (module.Class), not just
    the class name. This provides better identification in metrics, especially when multiple
    plugins use the same class name (e.g., TTS).

    Returns:
        Fully qualified class path like "vision_agents.plugins.cartesia.tts.TTS"
        or None if not a method call.
    """
    if not args:
        return None

    first = args[0]

    # Check if this looks like an instance method call (self parameter)
    if hasattr(first, "__class__") and not inspect.isclass(first):
        # Verify it's actually a method by checking the function's qualname contains a dot
        if "." in func.__qualname__:
            # Return the fully qualified class path
            return f"{first.__class__.__module__}.{first.__class__.__qualname__}"

    # Check if this looks like a class method call (cls parameter)
    if inspect.isclass(first) and func.__qualname__.startswith(first.__name__ + "."):
        return f"{first.__module__}.{first.__qualname__}"

    return None
