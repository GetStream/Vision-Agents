from collections import deque
from typing import AsyncIterator, Callable, Iterable, Optional, Self

import aiortc
from getstream.video.rtc import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm import LLM, Realtime
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.core.llm.realtime import (
    RealtimeAgentTranscript,
    RealtimeAudioOutput,
    RealtimeAudioOutputDone,
    RealtimeUserTranscript,
)
from vision_agents.core.stt import STT
from vision_agents.core.stt.stt import Transcript
from vision_agents.core.tts import TTS
from vision_agents.core.tts.tts import TTSOutputChunk
from vision_agents.core.turn_detection import TurnDetector, TurnEnded, TurnStarted
from vision_agents.core.utils.video_forwarder import VideoForwarder

__all__ = (
    "STTStub",
    "TTSStub",
    "LLMStub",
    "RealtimeStub",
    "TurnDetectorStub",
)


class STTStub(STT):
    def __init__(
        self,
        responses: Iterable[TurnStarted | TurnEnded | Transcript] = (),
        turn_detection: bool = True,
        factory: Callable[
            [PcmData, Participant],
            AsyncIterator[TurnStarted | TurnEnded | Transcript],
        ]
        | None = None,
    ) -> None:
        super().__init__()
        self._responses = deque(responses)
        self.turn_detection = turn_detection
        self._factory = factory

    @classmethod
    def from_iterable(
        cls, responses: Iterable[TurnStarted | TurnEnded | Transcript]
    ) -> Self:
        return cls(responses=responses)

    @classmethod
    def from_callable(
        cls,
        factory: Callable[
            [PcmData, Participant],
            AsyncIterator[TurnStarted | TurnEnded | Transcript],
        ],
    ) -> Self:
        return cls(factory=factory)

    async def process_audio(self, pcm_data: PcmData, participant: Participant) -> None:
        if self._factory is not None:
            async for item in self._factory(pcm_data, participant):
                self._output.send_nowait(item)
            return
        if len(self._responses):
            next_ = self._responses.popleft()
            self._output.send_nowait(next_)


class TurnDetectorStub(TurnDetector):
    def __init__(self, responses: Iterable[TurnStarted | TurnEnded] = ()) -> None:
        super().__init__()
        self._responses = deque(responses)

    async def process_audio(
        self, data: object, participant: Participant, conversation: object = None
    ) -> None:
        if len(self._responses):
            self._output.send_nowait(self._responses.popleft())


class LLMStub(LLM):
    def __init__(
        self,
        responses: Iterable[LLMResponseDelta | LLMResponseFinal] = (),
        factory: Callable[
            [str, Participant | None],
            AsyncIterator[LLMResponseDelta | LLMResponseFinal],
        ]
        | None = None,
    ) -> None:
        super().__init__()
        self._responses = list(responses)
        self._factory = factory

    async def simple_response(self, text, participant=None):
        if self._factory is not None:
            async for r in self._factory(text, participant):
                yield r
            return
        for r in self._responses:
            yield r

    @classmethod
    def from_iterable(cls, items: Iterable[LLMResponseDelta | LLMResponseFinal]):
        return cls(responses=list(items))

    @classmethod
    def from_callable(
        cls,
        factory: Callable[
            [str, Participant | None],
            AsyncIterator[LLMResponseDelta | LLMResponseFinal],
        ],
    ) -> Self:
        return cls(factory=factory)


RealtimeOutput = (
    RealtimeAudioOutput
    | RealtimeAudioOutputDone
    | RealtimeUserTranscript
    | RealtimeAgentTranscript
)


class RealtimeStub(Realtime):
    def __init__(
        self,
        audio_factory: Callable[
            [PcmData, Participant], AsyncIterator[RealtimeOutput]
        ]
        | None = None,
    ) -> None:
        super().__init__()
        self._audio_factory = audio_factory

    @classmethod
    def from_callable(
        cls,
        audio_factory: Callable[
            [PcmData, Participant], AsyncIterator[RealtimeOutput]
        ],
    ) -> Self:
        return cls(audio_factory=audio_factory)

    async def watch_video_track(
        self,
        track: aiortc.mediastreams.MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None: ...

    async def simple_response(
        self, text: str, participant: Optional[Participant] = None
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        for _ in ():
            yield _

    async def connect(self): ...

    async def simple_audio_response(
        self, pcm: PcmData, participant: Participant
    ) -> None:
        if self._audio_factory is not None:
            async for ev in self._audio_factory(pcm, participant):
                self._output.send_nowait(ev)

    async def close(self): ...


class TTSStub(TTS):
    def __init__(
        self,
        chunks: Iterable[TTSOutputChunk] = (),
        streaming: bool = False,
    ) -> None:
        super().__init__()
        self._chunks = list(chunks)
        # Per-instance override of the ClassVar so tests can toggle branch.
        self.streaming = streaming

    @classmethod
    def from_iterable(cls, chunks: Iterable[TTSOutputChunk]) -> Self:
        return cls(chunks=chunks)

    async def send_iter(self, text, participant=None, *args, **kwargs):
        if self._chunks:
            for chunk in self._chunks:
                yield chunk
        else:
            # Echo the received text so tests can observe which text
            # reached send_iter by inspecting the output stream.
            yield TTSOutputChunk(text=text)

    async def stream_audio(self, *_, **__):
        return b""

    async def stop_audio(self) -> None: ...
