from abc import ABC, abstractmethod

from getstream.video.rtc.track_util import PcmData

from ..edge.types import Participant


class AudioInputProcessor(ABC):
    """Abstract handler for the input-audio path between Realtime.process_audio and the provider."""

    @abstractmethod
    async def process_audio(
        self, pcm: PcmData, participant: Participant | None
    ) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...
