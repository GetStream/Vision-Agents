import base64
import json
import logging
import os
import uuid
from importlib.metadata import PackageNotFoundError, version
from typing import AsyncIterator, Iterator, Literal

import httpx
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts

logger = logging.getLogger(__name__)

INWORLD_API_BASE = "https://api.inworld.ai"
WAV_HEADER_SIZE = 44

try:
    _pkg_version = version("vision-agents-plugins-inworld")
except PackageNotFoundError:
    _pkg_version = "unknown"

USER_AGENT = f"vision-agents-plugins-inworld/{_pkg_version}"


def _strip_wav_header(data: bytes) -> bytes:
    """Strip the RIFF/WAV header from LINEAR16 audio if present."""
    if len(data) > WAV_HEADER_SIZE and data[:4] == b"RIFF":
        return data[WAV_HEADER_SIZE:]
    return data


class TTS(tts.TTS):
    """Inworld AI Text-to-Speech plugin."""

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str = "Sarah",
        model_id: Literal[
            "inworld-tts-1.5-max",
            "inworld-tts-1.5-mini",
            "inworld-tts-1",
            "inworld-tts-1-max",
            "inworld-tts-2",
        ] = "inworld-tts-2",
        sample_rate: int = 16000,
        temperature: float = 1.1,
        speaking_rate: float | None = None,
    ):
        """Initialize the Inworld AI TTS service.

        Args:
            api_key: Inworld AI API key – falls back to ``INWORLD_API_KEY`` env var.
            voice_id: The voice ID to use for synthesis.
            model_id: The model to use for synthesis.
            sample_rate: Desired PCM sample rate in Hz.
            temperature: Randomness when sampling audio tokens (0–2).
            speaking_rate: Speech speed multiplier (0.5–1.5). ``None`` uses the server default.
        """
        super().__init__(provider_name="inworld")

        api_key = api_key or os.getenv("INWORLD_API_KEY")
        if not api_key:
            raise ValueError(
                "INWORLD_API_KEY environment variable must be set or api_key must be provided"
            )

        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._sample_rate = sample_rate
        self._temperature = temperature
        self._speaking_rate = speaking_rate
        self._client = httpx.AsyncClient(timeout=60.0)

    async def stream_audio(
        self, text: str, *_, **__
    ) -> PcmData | Iterator[PcmData] | AsyncIterator[PcmData]:
        """Convert text to speech using Inworld AI streaming API.

        Args:
            text: The text to convert to speech (max 2,000 characters).
        """
        request_id = str(uuid.uuid4())
        headers = {
            "Authorization": f"Basic {self._api_key}",
            "Content-Type": "application/json",
            "X-User-Agent": USER_AGENT,
            "X-Request-Id": request_id,
        }

        audio_config: dict[str, object] = {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": self._sample_rate,
            "temperature": self._temperature,
        }
        if self._speaking_rate is not None:
            audio_config["speakingRate"] = self._speaking_rate

        payload = {
            "text": text,
            "voiceId": self._voice_id,
            "modelId": self._model_id,
            "audioConfig": audio_config,
        }

        return PcmData.from_response(
            self._stream_raw_pcm(headers, payload, request_id),
            sample_rate=self._sample_rate,
            channels=1,
            format=AudioFormat.S16,
        )

    async def _stream_raw_pcm(
        self,
        headers: dict[str, str],
        payload: dict[str, object],
        request_id: str,
    ) -> AsyncIterator[bytes]:
        """Yield raw LINEAR16 PCM bytes from the Inworld NDJSON stream."""
        url = f"{INWORLD_API_BASE}/tts/v1/voice:stream"

        async with self._client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code >= 400:
                error_text = (await response.aread()).decode()
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}: {error_text}",
                    request=response.request,
                    response=response,
                )

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping unparseable NDJSON line (request_id=%s)", request_id)
                    continue

                if "error" in data:
                    error_msg = data["error"].get("message", "Unknown error")
                    logger.error("Inworld API error: %s (request_id=%s)", error_msg, request_id)
                    continue

                result = data.get("result")
                if result and "audioContent" in result:
                    wav_bytes = base64.b64decode(result["audioContent"])
                    yield _strip_wav_header(wav_bytes)

    async def stop_audio(self) -> None:
        pass

    async def close(self) -> None:
        await self._client.aclose()
        await super().close()
