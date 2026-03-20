import asyncio
import io
import logging
import os
from typing import Any, AsyncIterator, Iterator, Literal, Optional

import aiohttp
import numpy as np

from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts

logger = logging.getLogger(__name__)

Voice = Literal["eve", "ara", "leo", "rex", "sal"]

VOICE_DESCRIPTIONS = {
    "eve": "Energetic, upbeat — engaging and enthusiastic (default)",
    "ara": "Warm, friendly — balanced and conversational",
    "leo": "Authoritative, strong — commanding, great for instructional content",
    "rex": "Confident, clear — professional, ideal for business",
    "sal": "Smooth, balanced — versatile for a wide range of contexts",
}

Codec = Literal["mp3", "wav", "pcm", "mulaw", "alaw"]
SampleRate = Literal[8000, 16000, 22050, 24000, 44100, 48000]


class TTS(tts.TTS):
    """
    Grok (xAI) Text-to-Speech integration for Vision Agents.

    Uses the xAI TTS REST API to convert text into spoken audio.
    Supports five expressive voices, inline speech tags for fine-grained
    delivery control, and multiple output formats.

    Speech tags supported in text:
        Inline:  [pause] [long-pause] [laugh] [chuckle] [giggle] [cry]
                 [tsk] [tongue-click] [lip-smack] [breath] [inhale]
                 [exhale] [sigh] [hum-tune]
        Wrapping: <whisper>text</whisper>  <shout>text</shout>
                  <slow>text</slow>  <fast>text</fast>
                  <soft>text</soft>  <loud>text</loud>
                  <high-pitch>text</high-pitch>  <low-pitch>text</low-pitch>
                  <sing>text</sing>
    """

    BASE_URL = "https://api.x.ai/v1/tts"

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: Voice = "eve",
        language: str = "en",
        codec: Codec = "pcm",
        sample_rate: SampleRate = 24000,
        bit_rate: Optional[int] = None,
        base_url: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        """
        Initialize the Grok TTS service.

        Args:
            api_key: xAI API key. Falls back to XAI_API_KEY env var.
            voice: Voice to use. One of "eve", "ara", "leo", "rex", "sal".
            language: BCP-47 language code (e.g. "en", "zh", "pt-BR") or "auto".
            codec: Audio codec. "pcm" is best for real-time pipelines.
            sample_rate: Output sample rate in Hz.
            bit_rate: MP3 bit rate (only used when codec is "mp3").
            base_url: Override the API endpoint URL.
            session: Optional pre-existing aiohttp.ClientSession.
        """
        super().__init__(provider_name="grok_tts")

        self._api_key = api_key or os.environ.get("XAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "xAI API key is required. Pass api_key or set XAI_API_KEY env var."
            )

        self.voice: Voice = voice
        self.language = language
        self.codec: Codec = codec
        self.sample_rate: SampleRate = sample_rate
        self.bit_rate = bit_rate
        self._base_url = base_url or self.BASE_URL
        self._own_session = session is None
        self._session = session
        self._current_task: Optional[asyncio.Task] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._own_session = True
        return self._session

    def _build_payload(self, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": text,
            "voice_id": self.voice,
            "language": self.language,
        }
        output_format: dict[str, Any] = {
            "codec": self.codec,
            "sample_rate": self.sample_rate,
        }
        if self.codec == "mp3" and self.bit_rate is not None:
            output_format["bit_rate"] = self.bit_rate
        payload["output_format"] = output_format
        return payload

    async def stream_audio(
        self, text: str, *_, **kwargs: Any
    ) -> PcmData | Iterator[PcmData] | AsyncIterator[PcmData]:
        """
        Convert text to speech using the Grok TTS API.

        Args:
            text: Text to synthesize (max 15,000 chars). Supports speech tags.
            **kwargs: Extra fields merged into the request payload.

        Returns:
            PcmData containing the synthesized audio.
        """
        session = await self._get_session()

        payload = self._build_payload(text)
        payload.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.post(
                    self._base_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        audio_bytes = await resp.read()
                        return self._decode_audio(audio_bytes)

                    if resp.status in (429, 500, 503):
                        wait = 2 ** attempt
                        logger.warning(
                            "Grok TTS returned %d, retrying in %ds (attempt %d/%d)",
                            resp.status, wait, attempt + 1, max_retries,
                        )
                        await asyncio.sleep(wait)
                        continue

                    error_body = await resp.text()
                    raise RuntimeError(
                        f"Grok TTS API error {resp.status}: {error_body}"
                    )
            except asyncio.CancelledError:
                raise
            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Network error: %s, retrying in %ds", e, wait)
                    await asyncio.sleep(wait)
                    continue
                raise

        raise RuntimeError("Grok TTS: max retries exceeded")

    def _decode_audio(self, audio_bytes: bytes) -> PcmData:
        """Decode raw audio bytes into PcmData based on the configured codec."""
        if self.codec == "pcm":
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            return PcmData.from_numpy(
                samples,
                sample_rate=self.sample_rate,
                channels=1,
                format=AudioFormat.S16,
            )

        if self.codec in ("mulaw", "alaw"):
            samples = self._decode_g711(audio_bytes, self.codec)
            return PcmData.from_numpy(
                samples,
                sample_rate=self.sample_rate,
                channels=1,
                format=AudioFormat.S16,
            )

        # MP3 and WAV: use the built-in decoder
        return self._decode_compressed(audio_bytes)

    @staticmethod
    def _decode_g711(data: bytes, law: str) -> np.ndarray:
        """Decode G.711 mu-law or A-law encoded bytes to 16-bit PCM."""
        raw = np.frombuffer(data, dtype=np.uint8)
        if law == "mulaw":
            return TTS._ulaw_decode(raw)
        return TTS._alaw_decode(raw)

    @staticmethod
    def _ulaw_decode(data: np.ndarray) -> np.ndarray:
        """Decode mu-law compressed audio to linear 16-bit PCM."""
        BIAS = 0x84
        CLIP = 32635

        sign = data & 0x80
        exponent = (data >> 4) & 0x07
        mantissa = data & 0x0F

        data = ~data
        sign = (data & 0x80).astype(np.int16)
        exponent = ((data >> 4) & 0x07).astype(np.int16)
        mantissa = (data & 0x0F).astype(np.int16)

        magnitude = ((mantissa << 1) | 0x21) << (exponent + 2)
        magnitude = magnitude - BIAS

        result = np.where(sign != 0, -magnitude, magnitude).astype(np.int16)
        return result

    @staticmethod
    def _alaw_decode(data: np.ndarray) -> np.ndarray:
        """Decode A-law compressed audio to linear 16-bit PCM."""
        data = data ^ 0x55
        sign = data & 0x80
        exponent = (data >> 4) & 0x07
        mantissa = data & 0x0F

        if_exp_zero = (mantissa << 1) | 1
        if_exp_nonzero = ((mantissa << 1) | 0x21) << (exponent - 1)

        magnitude = np.where(exponent == 0, if_exp_zero, if_exp_nonzero)
        result = np.where(sign != 0, -magnitude, magnitude).astype(np.int16)
        return result

    def _decode_compressed(self, audio_bytes: bytes) -> PcmData:
        """Decode MP3 or WAV audio to PcmData via the wave module (WAV) or pydub (MP3)."""
        if self.codec == "wav":
            import wave

            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
                samples = np.frombuffer(raw, dtype=np.int16)
                return PcmData.from_numpy(
                    samples,
                    sample_rate=sr,
                    channels=1,
                    format=AudioFormat.S16,
                )

        # MP3 fallback using pydub
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "pydub is required to decode MP3 audio. "
                "Install it with: pip install pydub"
            )

        segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        segment = segment.set_channels(1).set_sample_width(2)
        samples = np.frombuffer(segment.raw_data, dtype=np.int16)
        return PcmData.from_numpy(
            samples,
            sample_rate=segment.frame_rate,
            channels=1,
            format=AudioFormat.S16,
        )

    async def stop_audio(self) -> None:
        """Cancel any in-flight synthesis request."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        logger.info("Grok TTS stop requested")

    async def close(self) -> None:
        """Close the TTS provider and clean up resources."""
        await super().close()
        if self._own_session and self._session and not self._session.closed:
            await self._session.close()
