"""Telnyx Text-to-Speech via WebSocket streaming.

Docs: https://developers.telnyx.com/api/call-control/text-to-speech

Two properties of the wire protocol drive this implementation:

- A synthesis is primed with an init frame, then one or more text frames, then
  an empty-text stop frame. Telnyx closes the socket once the stop frame has
  been served, so a connection cannot be reused across ``stream_audio`` calls
  the way a persistent-socket provider allows.
- The audio frames carry slices of MP3, and a synthesis can span several
  concatenated MP3 files, each introduced by its own ID3v2 tag at the head of
  a WebSocket frame. Those tags have to be dropped before the bytes reach the
  decoder, otherwise decoding fails part way through the utterance.

The decoded sample rate depends on the voice (Polly voices return 24 kHz,
Kokoro voices 22.05 kHz), so the rate reported by the decoder is used rather
than a configured one.
"""

import asyncio
import base64
import binascii
import json
import logging
import os
from typing import Any, AsyncIterator, Optional, cast
from urllib.parse import urlencode

import aiohttp
import av
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts

logger = logging.getLogger(__name__)

WS_TTS_URL = "wss://api.telnyx.com/v2/text-to-speech/speech"

DEFAULT_VOICE = "Telnyx.KokoroTTS.af_heart"

ID3_HEADER_SIZE = 10


class TelnyxTTSError(Exception):
    """Raised when Telnyx TTS returns an error frame over WebSocket."""


class _Id3Stripper:
    """Removes the ID3v2 tag heading each MP3 file in the audio stream.

    Telnyx concatenates one MP3 file per synthesised segment. Each file starts
    with an ID3v2 tag at the head of a WebSocket frame, so only the head of a
    frame is inspected. A tag whose header or body spans frames is carried
    across calls.
    """

    def __init__(self) -> None:
        self._skip = 0
        self._pending = b""

    def feed(self, data: bytes) -> bytes:
        """Return ``data`` with any leading ID3v2 tag removed."""
        if self._skip:
            consumed = min(self._skip, len(data))
            data = data[consumed:]
            self._skip -= consumed
            if not data:
                return b""

        if self._pending:
            data = self._pending + data
            self._pending = b""

        if data[:3] != b"ID3":
            return data

        if len(data) < ID3_HEADER_SIZE:
            self._pending = data
            return b""

        # ID3v2 stores the tag size as four synchsafe bytes (7 bits each).
        size = data[6] << 21 | data[7] << 14 | data[8] << 7 | data[9]
        body = data[ID3_HEADER_SIZE:]
        consumed = min(size, len(body))
        self._skip = size - consumed
        return body[consumed:]


class TTS(tts.TTS):
    """Telnyx streaming Text-to-Speech.

    Opens one WebSocket per synthesis, streams MP3 slices back, and decodes
    them into ``PcmData`` as they arrive.

    Examples:

        from vision_agents.plugins import telnyx
        tts = telnyx.TTS(voice="AWS.Polly.Danielle-Neural")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = DEFAULT_VOICE,
        idle_timeout: float = 10.0,
    ) -> None:
        """Initialize Telnyx TTS.

        Args:
            api_key: Telnyx API key. Falls back to the ``TELNYX_API_KEY`` env var.
            voice: Voice id as listed by ``GET /v2/text-to-speech/voices``,
                for example ``Telnyx.KokoroTTS.af_heart`` or
                ``AWS.Polly.Danielle-Neural``.
            idle_timeout: Seconds of server silence before synthesis is treated
                as finished. Normally the server marks the last frame with
                ``isFinal``; this is a safety net.
        """
        super().__init__(provider_name="telnyx")

        self._api_key = api_key or os.environ.get("TELNYX_API_KEY")
        if not self._api_key:
            raise ValueError(
                "TELNYX_API_KEY env var or api_key parameter required for Telnyx TTS"
            )

        self.voice = voice
        self._idle_timeout = idle_timeout

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    async def close(self) -> None:
        """Close the current WebSocket and release the aiohttp session."""
        await super().close()
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        self._on_disconnected()

    async def stream_audio(
        self, text: str, *_: Any, **__: Any
    ) -> AsyncIterator[PcmData]:
        """Stream TTS audio chunks for ``text``.

        Returns:
            Async iterator yielding ``PcmData`` chunks.
        """

        async def _stream() -> AsyncIterator[PcmData]:
            async with self._lock:
                # Cleared under the lock so a stop_audio() aimed at an
                # in-flight synthesis cannot leave the event set for a call
                # queued behind it.
                self._stop_event.clear()
                try:
                    ws = await self._connect()
                    # Telnyx rejects a text frame that is not preceded by an
                    # init frame with "Invalid message".
                    await ws.send_str(json.dumps({"text": " "}))
                    await ws.send_str(json.dumps({"text": text}))
                    await ws.send_str(json.dumps({"text": ""}))
                    async for chunk in self._receive_audio(ws):
                        yield chunk
                except aiohttp.ClientConnectionError:
                    # stop_audio() closes the socket underneath us, which is a
                    # normal barge-in rather than a synthesis failure.
                    if not self._stop_event.is_set():
                        raise
                finally:
                    await self._close_ws()

        return _stream()

    async def stop_audio(self) -> None:
        """Cancel any in-flight synthesis and drop the connection."""
        self._stop_event.set()
        await self._close_ws()

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        url = f"{WS_TTS_URL}?{urlencode({'voice': self.voice})}"
        # aiohttp's default client timeout for the handshake is 300s, far past
        # the point where a caller waiting on speech should give up.
        ws = await asyncio.wait_for(
            self._session.ws_connect(
                url, headers={"Authorization": f"Bearer {self._api_key}"}
            ),
            timeout=self._idle_timeout,
        )
        self._ws = ws
        self._on_connected()
        logger.debug("Telnyx TTS websocket connected for voice %s", self.voice)
        return ws

    async def _close_ws(self) -> None:
        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.close()
            except (aiohttp.ClientError, ConnectionError):
                logger.debug("Error closing Telnyx TTS websocket", exc_info=True)
        self._ws = None

    async def _receive_audio(
        self, ws: aiohttp.ClientWebSocketResponse
    ) -> AsyncIterator[PcmData]:
        """Yield PcmData until the final frame, a stop, an idle timeout, or a close."""
        decoder = cast(av.AudioCodecContext, av.CodecContext.create("mp3", "r"))
        resampler = av.AudioResampler(format="s16", layout="mono")
        stripper = _Id3Stripper()

        while True:
            if self._stop_event.is_set():
                break
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=self._idle_timeout)
            except asyncio.TimeoutError:
                logger.debug("Telnyx TTS idle timeout, ending synthesis")
                break

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.ERROR,
            ):
                break
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning("Telnyx TTS sent non-JSON text: %s", msg.data)
                continue

            if not isinstance(data, dict):
                logger.warning("Telnyx TTS sent unexpected payload: %r", data)
                continue

            if data.get("error"):
                raise TelnyxTTSError(str(data["error"]))

            encoded = data.get("audio")
            if encoded:
                try:
                    raw = base64.b64decode(encoded, validate=True)
                except (binascii.Error, TypeError, ValueError):
                    logger.warning("Telnyx TTS sent audio that is not valid base64")
                    continue
                for pcm in self._decode(raw, decoder, resampler, stripper):
                    yield pcm

            if data.get("isFinal"):
                break

    def _decode(
        self,
        audio: bytes,
        decoder: av.AudioCodecContext,
        resampler: av.AudioResampler,
        stripper: _Id3Stripper,
    ) -> list[PcmData]:
        """Decode one WebSocket audio payload into PcmData chunks.

        A corrupt payload is dropped rather than allowed to abort the
        synthesis; the decoder recovers on the next packet boundary.
        """
        chunks: list[PcmData] = []
        try:
            for packet in decoder.parse(stripper.feed(audio)):
                for frame in decoder.decode(packet):
                    for resampled in resampler.resample(frame):
                        chunks.append(
                            PcmData(
                                samples=resampled.to_ndarray().reshape(-1),
                                sample_rate=resampled.sample_rate,
                                channels=1,
                                format=AudioFormat.S16,
                            )
                        )
        except av.FFmpegError:
            logger.warning("Telnyx TTS sent undecodable audio, dropping payload")
        return chunks
