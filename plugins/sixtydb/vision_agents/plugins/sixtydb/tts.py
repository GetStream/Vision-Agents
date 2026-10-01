"""60db complete-utterance synthesis using workspace voice IDs."""

import asyncio
import base64
import io
import json
import math
import os
import time
import wave
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts

SAMPLE_RATE = 24000
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class TTS(tts.TTS):
    """Buffer each utterance as mono PCM16 at 24 kHz.

    Interrupts discard stale output through the base TTS epoch. An in-flight
    HTTP request still finishes or times out; there is no server cancellation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model: str | None = None,
        speed: float = 1.0,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(provider_name="sixtydb")
        self.api_key = api_key or os.getenv("SIXTYDB_API_KEY")
        self.voice_id = voice_id or os.getenv("SIXTYDB_VOICE_ID")
        for name, value in (("api_key", self.api_key), ("voice_id", self.voice_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Provide {name} or SIXTYDB_{name.upper()}")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("model must be a nonempty model ID")
        if isinstance(speed, bool) or not math.isfinite(speed) or not 0.5 <= speed <= 2:
            raise ValueError("speed must be between 0.5 and 2.0")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.model = model or ""
        self.speed = speed
        self.timeout = timeout
        self._endpoint = "https://api.60db.ai/tts-synthesize"

    async def stream_audio(self, text: str, *_, **__) -> PcmData:
        if not isinstance(text, str) or not text.strip() or len(text) > 5000:
            raise ValueError("text must contain 1 to 5000 characters")
        audio = await asyncio.to_thread(self._synthesize, text)
        return PcmData.from_bytes(
            audio, sample_rate=SAMPLE_RATE, channels=1, format=AudioFormat.S16
        )

    def _synthesize(self, text: str) -> bytes:
        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "speed": self.speed,
            "audio_config": {
                "audio_encoding": "LINEAR16",
                "sample_rate_hertz": SAMPLE_RATE,
            },
            "timestamp_type": "NONE",
        }
        if self.model:
            payload["model_id"] = self.model
        request = Request(
            self._endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        deadline = time.monotonic() + self.timeout
        try:
            with build_opener(_NoRedirect()).open(
                request, timeout=self.timeout
            ) as response:
                body = bytearray()
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("60db synthesis timed out")
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise ValueError("60db response exceeds 32 MiB")
                content_type = response.headers.get_content_type()
                self._validate(
                    {
                        key: int(response.headers[header])
                        for key, header in (
                            ("sample_rate", "X-Sample-Rate"),
                            ("channels", "X-Channels"),
                            ("bit_depth", "X-Bit-Depth"),
                        )
                        if response.headers.get(header) is not None
                    }
                )
        except HTTPError as exc:
            exc.close()
            raise RuntimeError(f"60db synthesis failed (HTTP {exc.code})") from None
        except URLError:
            raise RuntimeError("60db synthesis connection failed") from None
        if content_type == "application/json":
            audio = self._record_audio(json.loads(body))
        elif content_type in {
            "application/x-ndjson",
            "application/ndjson",
            "text/plain",
        }:
            audio = b"".join(
                self._record_audio(json.loads(line))
                for line in body.splitlines()
                if line.strip()
            )
        elif content_type in {
            "audio/pcm",
            "audio/wav",
            "audio/x-wav",
            "application/octet-stream",
        }:
            audio = bytes(body)
        else:
            raise ValueError("60db returned unsupported audio content type")
        return self._pcm(audio)

    @staticmethod
    def _validate(record: Any) -> None:
        if not isinstance(record, dict):
            raise TypeError("60db returned an invalid response object")
        if (
            record.get("success") is False
            or record.get("type") == "error"
            or record.get("error")
        ):
            raise ValueError("60db reported a synthesis error")
        for key in ("encoding", "audio_encoding", "output_format"):
            if key in record and str(record[key]).lower() not in {
                "linear16",
                "pcm",
                "pcm16",
                "wav",
            }:
                raise ValueError("60db returned incompatible audio encoding")
        for key, expected in (
            ("sample_rate", SAMPLE_RATE),
            ("sample_rate_hertz", SAMPLE_RATE),
            ("channels", 1),
            ("bit_depth", 16),
        ):
            if key in record and record[key] != expected:
                raise ValueError("60db returned incompatible audio metadata")
        if "audio_config" in record:
            TTS._validate(record["audio_config"])

    @classmethod
    def _record_audio(cls, record: Any) -> bytes:
        cls._validate(record)
        result = record.get("result", record.get("backendResponse", record))
        cls._validate(result)
        value = result.get("audioContent", result.get("audio_base64"))
        if value is None:
            return b""
        if not isinstance(value, str):
            raise TypeError("60db audio must be base64 text")
        audio = base64.b64decode(value, validate=True)
        if audio.startswith(b"{"):
            try:
                inner = json.loads(audio)
            except (ValueError, UnicodeDecodeError):
                return audio
            audio = cls._record_audio(inner)
        if audio.startswith((b"RIFF", b"ID3", b"OggS", b"fLaC")):
            return cls._pcm(audio)
        return audio

    @staticmethod
    def _pcm(audio: bytes) -> bytes:
        if audio.startswith(b"RIFF"):
            with wave.open(io.BytesIO(audio), "rb") as wav:
                if (
                    wav.getnchannels(),
                    wav.getsampwidth(),
                    wav.getframerate(),
                    wav.getcomptype(),
                ) != (1, 2, SAMPLE_RATE, "NONE"):
                    raise ValueError("60db WAV must be mono PCM16 at 24 kHz")
                frames = wav.getnframes()
                audio = wav.readframes(frames)
                if len(audio) != frames * 2:
                    raise ValueError("60db returned truncated WAV audio")
        elif audio.startswith((b"ID3", b"OggS", b"fLaC")):
            raise ValueError("60db returned compressed audio instead of PCM")
        if not audio or len(audio) % 2:
            raise ValueError("60db returned empty or incomplete PCM16 audio")
        return audio

    async def stop_audio(self) -> None:
        """The base interrupt epoch discards output from interrupted requests."""
