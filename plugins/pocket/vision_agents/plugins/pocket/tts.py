import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Iterator, Literal

import numpy as np

from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts
from vision_agents.core.warmup import Warmable

from pocket_tts import TTSModel

logger = logging.getLogger(__name__)

Voice = Literal[
    "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"
]

VOICE_PATHS = {
    "alba": "hf://kyutai/tts-voices/alba-mackenna/casual.wav",
    "marius": "hf://kyutai/tts-voices/marius/casual.wav",
    "javert": "hf://kyutai/tts-voices/javert/casual.wav",
    "jean": "hf://kyutai/tts-voices/jean/casual.wav",
    "fantine": "hf://kyutai/tts-voices/fantine/casual.wav",
    "cosette": "hf://kyutai/tts-voices/cosette/casual.wav",
    "eponine": "hf://kyutai/tts-voices/eponine/casual.wav",
    "azelma": "hf://kyutai/tts-voices/azelma/casual.wav",
}


class TTS(tts.TTS, Warmable[tuple[TTSModel, Any]]):
    """
    Pocket TTS Text-to-Speech implementation.

    A lightweight CPU-based TTS model from Kyutai with ~200ms latency
    and voice cloning support.
    """

    def __init__(
        self,
        voice: Voice | str = "alba",
        client: TTSModel | None = None,
    ) -> None:
        """
        Initialize Pocket TTS.

        Args:
            voice: Built-in voice name or path to custom wav file for voice cloning.
            client: Optional pre-initialized TTSModel instance.
        """
        super().__init__(provider_name="pocket")

        self.voice = voice
        self._model: TTSModel | None = client
        self._voice_state = None
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def on_warmup(self) -> tuple[TTSModel, Any]:
        if self._model is not None and self._voice_state is not None:
            return (self._model, self._voice_state)

        loop = asyncio.get_running_loop()

        if self._model is not None:
            model = self._model
        else:
            logger.info("Loading Pocket TTS model...")
            model = await loop.run_in_executor(self._executor, TTSModel.load_model)
            logger.info("Pocket TTS model loaded")

        voice_path = VOICE_PATHS.get(self.voice, self.voice)
        logger.info(f"Loading voice state for: {self.voice}")
        voice_state = await loop.run_in_executor(
            self._executor,
            lambda: model.get_state_for_audio_prompt(voice_path),
        )
        logger.info("Voice state loaded")
        return (model, voice_state)

    def on_warmed_up(self, resource: tuple[TTSModel, Any]) -> None:
        self._model, self._voice_state = resource

    async def _ensure_loaded(self) -> None:
        """Ensure model and voice state are loaded."""
        if self._model is None or self._voice_state is None:
            resource = await self.on_warmup()
            self.on_warmed_up(resource)

    async def stream_audio(
        self, text: str, *_, **__
    ) -> PcmData | Iterator[PcmData] | AsyncIterator[PcmData]:
        """
        Convert text to speech using Pocket TTS.

        Args:
            text: The text to convert to speech.

        Returns:
            PcmData containing the synthesized audio.
        """
        await self._ensure_loaded()
        assert self._model is not None
        assert self._voice_state is not None

        model = self._model
        voice_state = self._voice_state

        def _generate():
            audio_tensor = model.generate_audio(voice_state, text)
            audio_np = audio_tensor.numpy()
            pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
            return pcm16, model.sample_rate

        loop = asyncio.get_running_loop()
        samples, sample_rate = await loop.run_in_executor(self._executor, _generate)

        return PcmData.from_numpy(
            samples, sample_rate=sample_rate, channels=1, format=AudioFormat.S16
        )

    async def stop_audio(self) -> None:
        """Stop audio playback (no-op for Pocket TTS)."""
        logger.info("Pocket TTS stop requested (no-op)")

    async def close(self) -> None:
        """Close the TTS and cleanup resources."""
        await super().close()
        self._executor.shutdown(wait=False)
