from abc import ABC, abstractmethod
from typing import Protocol

from getstream.video.rtc.track_util import PcmData

from ..edge.types import Participant


class AudioInputHost(Protocol):
    """Structural contract that AudioInputSender needs from its host."""

    connected: bool

    async def simple_audio_response(
        self, pcm: PcmData, participant: Participant
    ) -> None: ...


class AudioInputSender(ABC):
    """Abstract handler for the input-audio path between Realtime.process_audio and the provider."""

    @abstractmethod
    async def send(self, pcm: PcmData, participant: Participant | None) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...
