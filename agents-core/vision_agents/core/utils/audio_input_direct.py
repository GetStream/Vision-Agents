from getstream.video.rtc.track_util import PcmData

from ..edge.types import Participant
from ..llm.llm import AudioLLM
from .audio_input_processor import AudioInputProcessor


class DirectInput(AudioInputProcessor):
    """Forward every PCM chunk straight to the audio LLM's provider.

    Subclasses may override `process_audio` to add buffering/pacing and call
    `super().process_audio(...)` to deliver each chunk through the default path.
    """

    def __init__(self, audio_llm: AudioLLM) -> None:
        self._audio_llm = audio_llm

    async def process_audio(
        self, pcm: PcmData, participant: Participant | None
    ) -> None:
        if participant is None:
            return
        await self._audio_llm.simple_audio_response(pcm, participant)

    def clear(self) -> None:
        pass

    async def close(self) -> None:
        pass
