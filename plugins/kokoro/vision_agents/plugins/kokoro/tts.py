from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Iterator, Optional

import numpy as np
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts
from vision_agents.core.warmup import Warmable

try:
    from kokoro import KPipeline  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – mocked during CI
    KPipeline = None  # type: ignore  # noqa: N816


logger = logging.getLogger(__name__)


class TTS(tts.TTS, Warmable[KPipeline]):
    """Text-to-Speech plugin backed by the Kokoro-82M model."""

    def __init__(
        self,
        lang_code: str = "a",  # American English
        voice: str = "af_heart",
        speed: float = 1.0,
        sample_rate: int = 24_000,
        device: Optional[str] = None,
        client: Optional[KPipeline] = None,
    ) -> None:
        super().__init__(provider_name="kokoro")

        if KPipeline is None:
            raise ImportError(
                "The 'kokoro' package is not installed. ``pip install kokoro`` first."
            )

        self.lang_code = lang_code
        self.device = device
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        self._pipeline = client
        self.client = client
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def on_warmup(self) -> KPipeline:
        if self._pipeline is not None:
            return self._pipeline

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: (
                KPipeline(lang_code=self.lang_code)
                if self.device is None
                else KPipeline(lang_code=self.lang_code, device=self.device)
            ),
        )

    def on_warmed_up(self, resource: KPipeline) -> None:
        self._pipeline = resource
        self.client = resource

    async def _ensure_loaded(self) -> KPipeline:
        if self._pipeline is None:
            resource = await self.on_warmup()
            self.on_warmed_up(resource)
        return self._pipeline

    async def stream_audio(
        self, text: str, *_, **__
    ) -> PcmData | Iterator[PcmData] | AsyncIterator[PcmData]:  # noqa: D401
        pipeline = await self._ensure_loaded()
        loop = asyncio.get_running_loop()
        done = object()

        async def _aiter():
            generator = self._generate_chunks(pipeline, text)
            while True:
                chunk = await loop.run_in_executor(
                    self._executor, next, generator, done
                )
                if chunk is done:
                    break
                yield PcmData.from_bytes(
                    chunk,
                    sample_rate=self.sample_rate,
                    channels=1,
                    format=AudioFormat.S16,
                )

        return _aiter()

    async def stop_audio(self) -> None:
        """
        Clears the queue and stops playing audio.

        """
        logger.info("Kokoro TTS stop requested (no-op)")

    def _generate_chunks(self, pipeline: KPipeline, text: str):
        for _gs, _ps, audio in pipeline(
            text, voice=self.voice, speed=self.speed, split_pattern=r"\n+"
        ):
            if not isinstance(audio, np.ndarray):
                audio = np.asarray(audio)
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
            yield pcm16.tobytes()

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            self._executor.shutdown(wait=False)
