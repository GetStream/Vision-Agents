import asyncio
import base64
import binascii
import io
import json
import logging
import os
import struct
import wave
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path

import aiohttp
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.modelbest.cn/v1"
MAX_REFERENCE_AUDIO_BYTES = 5 * 1024 * 1024


class VoxCPMTTSError(Exception):
    """Raised when ModelBest cannot synthesize a VoxCPM response."""


class _WavStreamParser:
    """Extract PCM16 frames from a WAV file arriving in arbitrary chunks."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._cursor = 12
        self._data_started = False
        self._audio_bytes = 0
        self.sample_rate: int | None = None
        self.channels: int | None = None
        self._block_align: int | None = None

    def feed(self, chunk: bytes) -> bytes:
        """Consume a WAV fragment and return any complete PCM frames."""
        self._buffer.extend(chunk)
        if not self._data_started:
            self._parse_header()
        if not self._data_started:
            return b""
        return self._drain_frames()

    def finish(self) -> None:
        """Validate that the stream ended after complete audio frames."""
        if not self._data_started:
            raise VoxCPMTTSError("VoxCPM returned a WAV without a data chunk")
        if self._buffer:
            raise VoxCPMTTSError("VoxCPM returned a truncated PCM frame")
        if self._audio_bytes == 0:
            raise VoxCPMTTSError("VoxCPM returned an empty WAV")

    def _parse_header(self) -> None:
        if len(self._buffer) < 12:
            return
        if self._buffer[:4] != b"RIFF" or self._buffer[8:12] != b"WAVE":
            raise VoxCPMTTSError("VoxCPM returned data that is not a RIFF/WAVE file")

        while len(self._buffer) >= self._cursor + 8:
            chunk_id = bytes(self._buffer[self._cursor : self._cursor + 4])
            chunk_size = int.from_bytes(
                self._buffer[self._cursor + 4 : self._cursor + 8], "little"
            )
            chunk_start = self._cursor + 8

            if chunk_id == b"data":
                if self.sample_rate is None or self._block_align is None:
                    raise VoxCPMTTSError("VoxCPM returned a WAV without a PCM format")
                del self._buffer[:chunk_start]
                self._data_started = True
                return

            chunk_end = chunk_start + chunk_size
            if len(self._buffer) < chunk_end:
                return
            if chunk_id == b"fmt ":
                self._parse_format(bytes(self._buffer[chunk_start:chunk_end]))
            self._cursor = chunk_end + (chunk_size & 1)

    def _parse_format(self, payload: bytes) -> None:
        if len(payload) < 16:
            raise VoxCPMTTSError("VoxCPM returned an invalid WAV format chunk")
        encoding, channels, sample_rate, _, block_align, bits_per_sample = (
            struct.unpack("<HHIIHH", payload[:16])
        )
        if encoding != 1 or bits_per_sample != 16:
            raise VoxCPMTTSError("VoxCPM WAV output must use 16-bit PCM")
        if channels < 1 or sample_rate < 1 or block_align != channels * 2:
            raise VoxCPMTTSError("VoxCPM returned an invalid PCM format")
        self.channels = channels
        self.sample_rate = sample_rate
        self._block_align = block_align

    def _drain_frames(self) -> bytes:
        if self._block_align is None:
            return b""
        complete_size = len(self._buffer) - len(self._buffer) % self._block_align
        if complete_size == 0:
            return b""
        frames = bytes(self._buffer[:complete_size])
        del self._buffer[:complete_size]
        self._audio_bytes += len(frames)
        return frames


async def _iter_sse_events(
    lines: AsyncIterable[bytes],
) -> AsyncIterator[dict[str, object]]:
    data_lines: list[str] = []
    async for raw_line in lines:
        try:
            line = raw_line.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as exc:
            raise VoxCPMTTSError("VoxCPM returned invalid SSE text") from exc

        if not line:
            if data_lines:
                yield _decode_sse_event(data_lines)
                data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())

    if data_lines:
        yield _decode_sse_event(data_lines)


def _decode_sse_event(data_lines: list[str]) -> dict[str, object]:
    try:
        event = json.loads("\n".join(data_lines))
    except json.JSONDecodeError as exc:
        raise VoxCPMTTSError("VoxCPM returned invalid SSE event data") from exc
    if not isinstance(event, dict):
        raise VoxCPMTTSError("VoxCPM returned an invalid SSE event")
    return event


def _audio_data_uri(audio: bytes | str | Path | None, field_name: str) -> str | None:
    if audio is None:
        return None
    try:
        audio_bytes = (
            Path(audio).read_bytes() if isinstance(audio, (str, Path)) else audio
        )
    except OSError as exc:
        raise ValueError(f"Unable to read {field_name} WAV: {exc}") from exc
    if not audio_bytes:
        raise ValueError(f"{field_name} WAV is empty")
    if len(audio_bytes) > MAX_REFERENCE_AUDIO_BYTES:
        raise ValueError(f"{field_name} WAV exceeds ModelBest's 5 MiB limit")
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            if wav_file.getnframes() < 1:
                raise ValueError(f"{field_name} WAV is empty")
            if wav_file.getcomptype() != "NONE":
                raise ValueError(f"{field_name} must be an uncompressed PCM WAV")
    except wave.Error as exc:
        raise ValueError(f"{field_name} must be a valid WAV") from exc
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


class TTS(tts.TTS):
    """Streaming VoxCPM TTS through ModelBest's hosted Audio Speech API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        voice: str = "default",
        base_url: str = DEFAULT_BASE_URL,
        ref_audio: bytes | str | Path | None = None,
        prompt_audio: bytes | str | Path | None = None,
        prompt_text: str | None = None,
        request_timeout: float = 120.0,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Initialize the ModelBest VoxCPM client.

        Args:
            api_key: ModelBest API key. Defaults to ``MODELBEST_API_KEY``.
            model: ModelBest model ID with ``speech_synthesis`` capability.
            voice: Protocol voice placeholder. Speaker identity comes from audio.
            base_url: ModelBest API base URL.
            ref_audio: Optional WAV used for speaker identity cloning.
            prompt_audio: Optional WAV used for high-fidelity delivery cloning.
            prompt_text: Exact transcript of ``prompt_audio``.
            request_timeout: Maximum seconds without response data.
            session: Optional externally managed aiohttp session.
        """
        super().__init__(provider_name="voxcpm")

        self._api_key = api_key or os.getenv("MODELBEST_API_KEY")
        if not self._api_key:
            raise ValueError("MODELBEST_API_KEY env var or api_key parameter required")

        self.model = model or os.getenv("MODELBEST_VOXCPM_MODEL_ID") or ""
        if not self.model:
            raise ValueError(
                "MODELBEST_VOXCPM_MODEL_ID env var or model parameter required"
            )
        if bool(prompt_audio) != bool(prompt_text and prompt_text.strip()):
            raise ValueError("prompt_audio and prompt_text must be provided together")

        self.voice = voice
        self._base_url = base_url.rstrip("/")
        self._ref_audio = _audio_data_uri(ref_audio, "ref_audio")
        self._prompt_audio = _audio_data_uri(prompt_audio, "prompt_audio")
        self._prompt_text = prompt_text.strip() if prompt_text else None
        self._timeout = aiohttp.ClientTimeout(
            total=None, connect=10.0, sock_read=request_timeout
        )
        self._session = session
        self._owns_session = session is None
        self._response: aiohttp.ClientResponse | None = None
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    async def stream_audio(
        self, text: str, *_args: object, **_kwargs: object
    ) -> AsyncIterator[PcmData]:
        """Stream synthesized PCM chunks for a complete utterance."""

        async def _stream() -> AsyncIterator[PcmData]:
            async with self._lock:
                self._stop_event.clear()
                response = await self._request(text)
                self._response = response
                parser = _WavStreamParser()
                completed = False
                try:
                    async for event in _iter_sse_events(response.content):
                        if self._stop_event.is_set():
                            return
                        event_type = event.get("type")
                        if event_type == "speech.audio.delta":
                            encoded_audio = event.get("audio")
                            if not isinstance(encoded_audio, str) or not encoded_audio:
                                raise VoxCPMTTSError(
                                    "VoxCPM returned an empty audio chunk"
                                )
                            try:
                                wav_chunk = base64.b64decode(
                                    encoded_audio, validate=True
                                )
                            except (binascii.Error, ValueError) as exc:
                                raise VoxCPMTTSError(
                                    "VoxCPM returned invalid Base64 audio"
                                ) from exc
                            pcm = parser.feed(wav_chunk)
                            if pcm:
                                yield PcmData.from_bytes(
                                    pcm,
                                    sample_rate=parser.sample_rate or 48_000,
                                    channels=parser.channels or 1,
                                    format=AudioFormat.S16,
                                )
                        elif event_type == "speech.audio.done":
                            completed = True
                            break
                        elif event_type == "error":
                            raise VoxCPMTTSError(
                                str(event.get("error") or "VoxCPM returned an error")
                            )

                    if not self._stop_event.is_set():
                        if not completed:
                            raise VoxCPMTTSError(
                                "VoxCPM stream ended before speech.audio.done"
                            )
                        parser.finish()
                except aiohttp.ClientConnectionError:
                    if not self._stop_event.is_set():
                        raise
                finally:
                    response.close()
                    if self._response is response:
                        self._response = None

        return _stream()

    async def stop_audio(self) -> None:
        """Stop yielding audio and close the active HTTP response."""
        self._stop_event.set()
        if self._response is not None:
            self._response.close()

    async def close(self) -> None:
        """Cancel synthesis and release the owned HTTP session."""
        await super().close()
        if self._owns_session and self._session is not None:
            await self._session.close()
        self._session = None

    async def _request(self, text: str) -> aiohttp.ClientResponse:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True

        payload: dict[str, object] = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": "wav",
            "stream": True,
        }
        if self._ref_audio:
            payload["ref_audio"] = self._ref_audio
        if self._prompt_audio and self._prompt_text:
            payload["prompt_audio"] = self._prompt_audio
            payload["prompt_text"] = self._prompt_text

        response = await self._session.post(
            f"{self._base_url}/audio/speech",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "text/event-stream",
            },
            json=payload,
            timeout=self._timeout,
        )
        if response.status != 200:
            message = (await response.text())[:500]
            response.close()
            raise VoxCPMTTSError(
                f"VoxCPM request failed with HTTP {response.status}: {message}"
            )
        return response
