from getstream.video.rtc.track_util import PcmData

from ..edge.types import Participant
from ..llm.llm import AudioLLM
from .audio_input_sender import AudioInputSender


class DirectInput(AudioInputSender):
    """Forward every PCM chunk straight to the audio LLM's provider.

    Subclasses may override `send` to add buffering/pacing and call
    `super().send(...)` to deliver each chunk through the default path.
    """

    def __init__(self, audio_llm: AudioLLM) -> None:
        self._audio_llm = audio_llm

    async def send(self, pcm: PcmData, participant: Participant | None) -> None:
        if participant is None:
            return
        await self._audio_llm.simple_audio_response(pcm, participant)

    def clear(self) -> None:
        pass

    async def close(self) -> None:
        pass
