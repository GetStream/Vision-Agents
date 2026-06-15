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


class DirectInput(AudioInputSender):
    """Forward every PCM chunk straight to the host's provider.

    Subclasses may override `send` to add buffering/pacing and call
    `super().send(...)` to deliver each chunk through the default path.
    """

    def __init__(self, host: AudioInputHost) -> None:
        self._host = host

    async def send(self, pcm: PcmData, participant: Participant | None) -> None:
        if not self._host.connected:
            return
        if participant is None:
            return
        await self._host.simple_audio_response(pcm, participant)

    def clear(self) -> None:
        pass

    async def close(self) -> None:
        pass
