import abc
import logging
from typing import Optional

from getstream.video.rtc import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.vad.silero import SileroVADSession, SileroVADSessionPool
from vision_agents.core.warmup import Warmable

logger = logging.getLogger(__name__)


class AudioFilter(abc.ABC):
    """
    Built-in audio filter that gates incoming audio before it reaches the
    AudioProcessor pipeline. Unlike AudioProcessor (a user-facing plugin),
    AudioFilter is an internal Agent component.
    """

    @abc.abstractmethod
    async def process_audio(
        self, pcm: PcmData, participant: Participant
    ) -> Optional[PcmData]:
        """Filter audio. Return PcmData to pass through, or None to drop."""

    @abc.abstractmethod
    def clear(self, participant: Optional[Participant] = None) -> None:
        """Clear any held lock/state. Called by the Agent on turn end or participant disconnect.

        Args:
            participant: If provided, only clear if this participant holds the lock.
                         If None, clear unconditionally.
        """


class FirstSpeakerWinsFilter(AudioFilter, Warmable[SileroVADSessionPool]):
    """
    First-speaker-wins audio gate for multi-participant calls.

    When multiple participants are on a call, only the first person to start
    speaking gets their audio forwarded through the pipeline. Other participants'
    audio is dropped (returns None) until the active speaker's turn ends.

    Lock lifecycle:
        - Acquire: first participant whose VAD score exceeds the threshold.
        - Hold: only that participant's audio passes through.
        - Release: on TurnEndedEvent, silence timeout, or participant disconnect.
    """

    def __init__(
        self,
        speech_threshold: float = 0.5,
        silence_release_ms: float = 1500.0,
        model_dir: str = "/tmp/first_speaker_wins_model",
    ):
        """
        Args:
            speech_threshold: VAD score above which audio counts as speech (0.0-1.0).
            silence_release_ms: Release the lock after this many ms of continuous
                silence from the active speaker (fallback when no TurnEndedEvent fires).
            model_dir: Directory for the Silero VAD ONNX model.
        """
        self._speech_threshold = speech_threshold
        self._silence_release_ms = silence_release_ms
        self._model_dir = model_dir

        self._active_speaker_id: Optional[str] = None
        self._active_speaker_silent_ms: float = 0.0
        self._vad_session: Optional[SileroVADSession] = None

    @property
    def active_speaker_id(self) -> Optional[str]:
        return self._active_speaker_id

    async def process_audio(
        self, pcm: PcmData, participant: Participant
    ) -> Optional[PcmData]:
        if self._vad_session is None:
            raise RuntimeError("warmup() must be called before process_audio()")

        speaker_id = participant.id
        if self._active_speaker_id is None:
            # No lock held — allow all audio through.
            # Acquire the lock on the first participant with detected speech.
            score = self._vad_session.predict_speech(pcm)
            is_speech = score > self._speech_threshold
            if is_speech:
                self._acquire(speaker_id)
            return pcm
        elif self._active_speaker_id != speaker_id:
            # Somebody else is talking and the lock is still held, exit early without
            # running an expensive VAD model
            return None
        else:
            # This participant holds the lock.
            # Run the VAD model to check whether we're getting speech or silence
            score = self._vad_session.predict_speech(pcm)
            is_speech = score > self._speech_threshold
            if is_speech:
                self._active_speaker_silent_ms = 0.0
            else:
                self._active_speaker_silent_ms += pcm.duration_ms
                if self._active_speaker_silent_ms >= self._silence_release_ms:
                    self.clear()
            return pcm

    async def on_warmup(self) -> SileroVADSessionPool:
        return await SileroVADSessionPool.load(self._model_dir)

    def on_warmed_up(self, resource: SileroVADSessionPool) -> None:
        self._vad_session = resource.session(reset_interval_seconds=5.0)

    def _acquire(self, speaker_id: str) -> None:
        self._active_speaker_id = speaker_id
        self._active_speaker_silent_ms = 0.0
        logger.debug("Speaker lock acquired by %s", speaker_id)

    def clear(self, participant: Optional[Participant] = None) -> None:
        if self._active_speaker_id is None:
            return
        if participant is not None and participant.id != self._active_speaker_id:
            return
        logger.debug("Speaker lock released for %s", self._active_speaker_id)
        self._active_speaker_id = None
        self._active_speaker_silent_ms = 0.0
