from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import List

from . import TTS
from .events import (
    TTSAudioEvent,
    TTSErrorEvent,
    TTSSynthesisStartEvent,
    TTSSynthesisCompleteEvent,
)

from getstream.video.rtc import PcmData


@dataclass
class TTSResult:
    speeches: List[PcmData] = field(default_factory=list)
    errors: List[Exception] = field(default_factory=list)
    started: bool = False
    completed: bool = False


class TTSSession:
    """Test helper to collect TTS events and wait for outcomes.

    Usage:
        session = TTSSession(tts)
        await tts.send(text)
        result = await session.wait_for_result(timeout=10.0)
        assert not result.errors
        assert result.speeches[0]
    """

    def __init__(self, tts: TTS):
        self._tts = tts
        self._speeches: List[PcmData] = []
        self._errors: List[Exception] = []
        self._started = False
        self._completed = False
        self._first_event = asyncio.Event()

        @tts.events.subscribe
        async def _on_start(ev: TTSSynthesisStartEvent):  # type: ignore[name-defined]
            self._started = True

        @tts.events.subscribe
        async def _on_audio(ev: TTSAudioEvent):  # type: ignore[name-defined]
            if ev.data:
                self._speeches.append(ev.data)
            self._first_event.set()

        @tts.events.subscribe
        async def _on_error(ev: TTSErrorEvent):  # type: ignore[name-defined]
            if ev.error:
                self._errors.append(ev.error)
            self._first_event.set()

        @tts.events.subscribe
        async def _on_complete(ev: TTSSynthesisCompleteEvent):  # type: ignore[name-defined]
            self._completed = True

    @property
    def speeches(self) -> List[PcmData]:
        return self._speeches

    @property
    def errors(self) -> List[Exception]:
        return self._errors

    async def wait_for_result(self, timeout: float = 10.0) -> TTSResult:
        try:
            await asyncio.wait_for(self._first_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Return whatever we have so far
            pass
        return TTSResult(
            speeches=list(self._speeches),
            errors=list(self._errors),
            started=self._started,
            completed=self._completed,
        )


@dataclass
class EventLoopProbeResult:
    ticks: int
    elapsed_ms: float
    max_gap_ms: float


async def _probe_event_loop_while(coro, interval: float = 0.01) -> EventLoopProbeResult:
    """Run a coroutine while probing event loop responsiveness.

    Spawns a ticker task that sleeps for `interval` and counts ticks,
    measuring the maximum observed gap between wakeups while `coro` runs.

    Returns probe metrics once `coro` completes.
    """
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    ticks = 0
    max_gap = 0.0
    last = loop.time()

    async def _ticker():
        nonlocal ticks, max_gap, last
        while not stop.is_set():
            await asyncio.sleep(interval)
            now = loop.time()
            gap = (now - last) * 1000.0
            if gap > max_gap:
                max_gap = gap
            ticks += 1
            last = now

    start = loop.time()
    ticker_task = asyncio.create_task(_ticker())
    try:
        await coro
    finally:
        stop.set()
        try:
            await asyncio.wait_for(ticker_task, timeout=1.0)
        except asyncio.TimeoutError:
            ticker_task.cancel()
    elapsed_ms = (loop.time() - start) * 1000.0
    return EventLoopProbeResult(ticks=ticks, elapsed_ms=elapsed_ms, max_gap_ms=max_gap)
