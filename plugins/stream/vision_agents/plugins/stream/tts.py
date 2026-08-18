import asyncio
import logging
import struct
import uuid
from typing import Any, AsyncIterator, Optional, Union

import numpy as np
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts
from vision_agents.core.utils.utils import cancel_and_wait

from ._backend import Backend
from ._socket import Socket

logger = logging.getLogger(__name__)

# HEADER is the little-endian preamble on every audio frame: a uint32 sample rate, a uint16
# channel count, and two bytes held back so the samples that follow stay aligned.
HEADER = struct.Struct("<IHH")


class TTS(tts.TTS):
    """Speech routed through the acceleration backend.

    For a pipeline that stays in Python: the conversation is here and only the voice is
    somewhere else. Utterances are spoken one at a time over a socket that stays open, so
    a provider that streams is still streaming by the time the audio reaches you.
    """

    streaming: bool = True

    def __init__(
        self,
        target: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        url: Optional[str] = None,
        customer_id: Optional[str] = None,
    ):
        """Route speech to `target`.

        Args:
            target: A `provider/model` name or a capability shortcut such as
                `en-low-latency`.
            voice: A provider-specific voice id.
            language: A language hint, which narrows the candidates.
            tags: Cost labels carried onto every request.
            url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
            customer_id: Who the work is billed to. Defaults to
                `STREAM_ACCELERATION_CUSTOMER_ID`.
        """
        super().__init__(provider_name="stream")
        self.model = target
        self.voice = voice
        self.language = language
        self.tags = tags or {}
        self.backend = Backend(url=url, customer_id=customer_id)

        self._socket: Optional[Socket] = None
        self._reader: Optional[asyncio.Task] = None
        self._incoming: asyncio.Queue[Union[bytes, dict[str, Any]]] = asyncio.Queue()
        # One utterance at a time: audio comes back as bare frames, so two overlapping
        # syntheses would be indistinguishable on the way in.
        self._speaking = asyncio.Lock()

    async def start(self):
        """Open the socket and start reading audio off it."""
        self._socket = Socket(
            self.backend.socket("/v1/tts/stream"), self.backend.headers
        )
        await self._socket.connect()
        await self._socket.send(
            {
                "type": "start",
                "target": self.model,
                "voice": self.voice or "",
                "languages": [self.language] if self.language else [],
                "tags": self.tags,
            }
        )
        self._reader = asyncio.create_task(self._read())
        self._on_connected()

    async def stream_audio(self, text: str, *args, **kwargs) -> AsyncIterator[PcmData]:
        """Speak `text`, yielding audio as the provider produces it."""
        return self._synthesize(text)

    async def stop_audio(self) -> None:
        """Abandon what is being spoken and drop the audio already on its way."""
        if self._socket is not None and self._socket.open:
            await self._socket.send({"type": "interrupt"})
        while not self._incoming.empty():
            self._incoming.get_nowait()

    async def close(self):
        """Close the socket and stop reading it."""
        await super().close()
        if self._reader is not None:
            await cancel_and_wait(self._reader)
            self._reader = None
        if self._socket is not None:
            await self._socket.close()
            self._socket = None
            self._on_disconnected()

    async def _synthesize(self, text: str) -> AsyncIterator[PcmData]:
        if self._socket is None or not self._socket.open:
            raise RuntimeError("the text-to-speech socket is not open")

        async with self._speaking:
            await self._socket.send(
                {
                    "type": "speak",
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "voice": self.voice or "",
                    "language": self.language or "",
                    "final": True,
                }
            )

            while True:
                message = await self._incoming.get()
                if isinstance(message, bytes):
                    yield _pcm_of(message)
                    continue

                kind = message.get("type", "")
                if kind == "synthesis_complete":
                    return
                if kind == "error":
                    raise RuntimeError(message.get("error", "synthesis failed"))
                if kind == "closed":
                    return

    async def _read(self) -> None:
        """Hand everything the router sends to whoever is synthesizing."""
        if self._socket is None:
            return

        async for frame in self._socket.frames():
            await self._incoming.put(frame)


def _pcm_of(message: bytes) -> PcmData:
    """Read one audio frame, whose header says how to play what follows."""
    sample_rate, channels, _ = HEADER.unpack_from(message)
    samples = np.frombuffer(message[HEADER.size :], dtype="<i2").astype(np.int16)
    return PcmData(
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        format=AudioFormat.S16,
    )
