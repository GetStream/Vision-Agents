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


@dataclass
class TTSResult:
    speeches: List[bytes] = field(default_factory=list)
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
        self._speeches: List[bytes] = []
        self._errors: List[Exception] = []
        self._started = False
        self._completed = False
        self._first_event = asyncio.Event()

        @tts.events.subscribe
        async def _on_start(ev: TTSSynthesisStartEvent):  # type: ignore[name-defined]
            self._started = True

        @tts.events.subscribe
        async def _on_audio(ev: TTSAudioEvent):  # type: ignore[name-defined]
            if ev.audio_data:
                self._speeches.append(ev.audio_data)
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
    def speeches(self) -> List[bytes]:
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
