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
            speed: Speaking-rate multiplier, 0.25 to 4.0.
            base_url: Override the API base URL.
            session: Optional pre-existing ``aiohttp.ClientSession``.
        """
        super().__init__(provider_name="audexum")

        self.api_key = api_key or os.getenv("AUDEXUM_API_KEY")
        if not self.api_key:
            raise ValueError("AUDEXUM_API_KEY env var or api_key parameter required")

        self.voice = voice
        self.speed = speed
        self._base_url = (base_url or BASE_URL).rstrip("/")
        self._session = session
        self._own_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._own_session = True
        return self._session

    async def stream_audio(
        self, text: str, *_, **__
    ) -> PcmData | Iterator[PcmData] | AsyncIterator[PcmData]:
        """Synthesize ``text`` and return it as a single PcmData buffer."""
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
        """No-op: each request is a single one-shot HTTP call."""
        logger.debug("Audexum TTS stop requested (no-op)")

    async def close(self) -> None:
        """Close the HTTP session if this instance created it."""
        try:
            if self._own_session and self._session and not self._session.closed:
                await self._session.close()
        finally:
            await super().close()
