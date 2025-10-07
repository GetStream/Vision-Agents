from __future__ import annotations

import asyncio
import logging

import numpy as np
from typing import AsyncIterator, List, Optional

from vision_agents.core import tts
from getstream.video.rtc.audio_track import AudioStreamTrack

try:
    from kokoro import KPipeline  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – mocked during CI
    KPipeline = None  # type: ignore  # noqa: N816


class TTS(tts.TTS):
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
        super().__init__()

        if KPipeline is None:
            raise ImportError(
                "The 'kokoro' package is not installed. ``pip install kokoro`` first."
            )

        self._pipeline = (
            KPipeline(lang_code=lang_code)
            if device is None
            else KPipeline(lang_code=lang_code, device=device)
        )
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        self.client = client if client is not None else self._pipeline

    def get_required_framerate(self) -> int:
        """Get the required framerate for Kokoro TTS."""
        return self.sample_rate

    def get_required_stereo(self) -> bool:
        """Get whether Kokoro TTS requires stereo audio."""
        return False  # Kokoro returns mono audio

    def set_output_track(self, track: AudioStreamTrack) -> None:  # noqa: D401
        if track.framerate != self.sample_rate:
            raise TypeError(
                f"Invalid framerate {track.framerate}, Kokoro requires {self.sample_rate} Hz"
            )
        super().set_output_track(track)

    async def stream_audio(self, text: str, *_, **__) -> AsyncIterator[bytes]:  # noqa: D401
        loop = asyncio.get_event_loop()
        chunks: List[bytes] = await loop.run_in_executor(
            None, lambda: list(self._generate_chunks(text))
        )

        async def _aiter():
            for chunk in chunks:
                yield chunk

        return _aiter()

    async def stop_audio(self) -> None:
        """
        Clears the queue and stops playing audio.

        """
        try:
            await self.track.flush()
            return
        except Exception as e:
            logging.error(f"Error flushing audio track: {e}")

    def _generate_chunks(self, text: str):
        for _gs, _ps, audio in self._pipeline(
            text, voice=self.voice, speed=self.speed, split_pattern=r"\n+"
        ):
            if not isinstance(audio, np.ndarray):
                audio = np.asarray(audio)
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
            yield pcm16.tobytes()
