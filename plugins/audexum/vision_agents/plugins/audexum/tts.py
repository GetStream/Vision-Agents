import asyncio
import io
import logging
import os
import wave
from typing import AsyncIterator, Iterator, Optional

import aiohttp
import numpy as np
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts

logger = logging.getLogger(__name__)

BASE_URL = "https://audexum.com/v1"


class TTS(tts.TTS):
    """Text-to-Speech plugin backed by the Audexum API (https://audexum.com).

    Uses Audexum's OpenAI-compatible ``/v1/audio/speech`` endpoint and decodes
    the returned WAV into PCM.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "Nova",
        speed: float = 1.0,
        base_url: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        """Create a new Audexum TTS instance.

        Args:
            api_key: Audexum API key. Falls back to the ``AUDEXUM_API_KEY`` env var.
            voice: Voice name, e.g. ``Nova``, ``Atlas``, ``F1`` or ``M1``
                (list: ``GET https://audexum.com/v1/voices``).
            speed: Speaking-rate multiplier, 0.25 to 4.0. Applied to the
                ``F1``-``F5`` / ``M1``-``M5`` voices; named voices keep their
                natural pace.
            base_url: Override the API base URL.
            session: Optional pre-existing ``aiohttp.ClientSession``.
        """
        super().__init__(provider_name="audexum")

        self.api_key = api_key or os.getenv("AUDEXUM_API_KEY")
        if not self.api_key:
            raise ValueError("AUDEXUM_API_KEY env var or api_key parameter required")

        if not 0.25 <= speed <= 4.0:
            raise ValueError(f"speed must be between 0.25 and 4.0, got {speed}")

        self.voice = voice
        self.speed = speed
        self._base_url = (base_url or BASE_URL).rstrip("/")
        self._session = session
        self._own_session = session is None
        # Every in-flight synthesis, so stop_audio() can cancel concurrent calls.
        self._pending: set[asyncio.Task] = set()
        self._stopped: set[asyncio.Task] = set()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._own_session = True
        return self._session

    async def stream_audio(
        self, text: str, *_, **__
    ) -> PcmData | Iterator[PcmData] | AsyncIterator[PcmData]:
        """Synthesize ``text`` and return it as a single PcmData buffer.

        Returns an empty buffer if ``stop_audio()`` cancels the request.
        """
        task = asyncio.ensure_future(self._synthesize(text))
        self._pending.add(task)
        try:
            return await task
        except asyncio.CancelledError:
            # Swallow only a cancel that stop_audio() requested; the caller being
            # cancelled itself must still propagate (Task.cancelling: 3.11+).
            cancelling = getattr(asyncio.current_task(), "cancelling", None)
            if task not in self._stopped or (cancelling is not None and cancelling()):
                raise
            return PcmData.from_numpy(
                np.zeros(0, dtype=np.int16),
                sample_rate=24000,
                channels=1,
                format=AudioFormat.S16,
            )
        finally:
            self._pending.discard(task)
            self._stopped.discard(task)

    async def _synthesize(self, text: str) -> PcmData:
        session = await self._get_session()
        async with session.post(
            f"{self._base_url}/audio/speech",
            json={
                "model": "audexum",
                "input": text,
                "voice": self.voice,
                "speed": self.speed,
                "response_format": "wav",
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"Audexum TTS API error {resp.status}: {await resp.text()}"
                )
            audio = await resp.read()

        with wave.open(io.BytesIO(audio), "rb") as wf:
            if wf.getsampwidth() != 2:
                raise RuntimeError("Audexum TTS returned non 16-bit WAV audio")
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            return PcmData.from_numpy(
                samples,
                sample_rate=wf.getframerate(),
                channels=wf.getnchannels(),
                format=AudioFormat.S16,
            )

    async def stop_audio(self) -> None:
        """Cancel every in-flight synthesis request; their output is discarded."""
        for task in list(self._pending):
            if not task.done():
                self._stopped.add(task)
                task.cancel()
        logger.debug("Audexum TTS stop requested")

    async def close(self) -> None:
        """Stop pending requests, then close the HTTP session if this instance created it."""
        try:
            # Cancel in-flight requests first so their callers get empty audio
            # instead of a connection error from the session teardown.
            await self.stop_audio()
            if self._own_session and self._session and not self._session.closed:
                await self._session.close()
        finally:
            await super().close()
