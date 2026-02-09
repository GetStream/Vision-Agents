import pytest
from getstream.video.rtc import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.utils.audio_filter import FirstSpeakerWinsFilter


def _participant(user_id: str, id: str = "") -> Participant:
    return Participant(original=None, user_id=user_id, id=id or user_id)


def _silence_chunk(silence_1s_16khz: PcmData) -> PcmData:
    """Slice a 20ms chunk from the 1s silence fixture."""
    return PcmData(
        samples=silence_1s_16khz.samples[:320],
        sample_rate=silence_1s_16khz.sample_rate,
        format=silence_1s_16khz.format,
    )


@pytest.fixture
async def audio_filter(tmp_path):
    """Create a filter with a real Silero VAD model."""
    f = FirstSpeakerWinsFilter(
        speech_threshold=0.5,
        silence_release_ms=100.0,
        model_dir=str(tmp_path / "vad_model"),
    )
    await f.warmup()
    return f


@pytest.mark.skip_blockbuster
class TestFirstSpeakerWinsFilter:
    async def test_no_lock_on_silence(self, audio_filter, silence_1s_16khz):
        """Silence does not acquire the lock."""
        alice = _participant("alice")
        chunk = _silence_chunk(silence_1s_16khz)

        for _ in range(5):
            result = await audio_filter.process_audio(chunk, alice)
            assert result is not None

        assert audio_filter.active_speaker_id is None

    async def test_lock_acquired_on_speech(self, audio_filter, mia_audio_16khz_chunked):
        """First participant with speech acquires the lock."""
        alice = _participant("alice")

        for chunk in mia_audio_16khz_chunked:
            await audio_filter.process_audio(chunk, alice)
            if audio_filter.active_speaker_id is not None:
                break

        assert audio_filter.active_speaker_id == "alice"

    async def test_lock_blocks_other_participants(
        self, audio_filter, mia_audio_16khz_chunked, silence_1s_16khz
    ):
        """Once someone holds the lock, other participants' audio is dropped."""
        alice = _participant("alice")
        bob = _participant("bob")

        for chunk in mia_audio_16khz_chunked:
            await audio_filter.process_audio(chunk, alice)
            if audio_filter.active_speaker_id is not None:
                break
        assert audio_filter.active_speaker_id == "alice"

        result = await audio_filter.process_audio(_silence_chunk(silence_1s_16khz), bob)
        assert result is None

    async def test_active_speaker_audio_passes(
        self, audio_filter, mia_audio_16khz_chunked
    ):
        """The active speaker's audio continues to pass through."""
        alice = _participant("alice")
        chunks = iter(mia_audio_16khz_chunked)

        # Acquire the lock
        for chunk in chunks:
            await audio_filter.process_audio(chunk, alice)
            if audio_filter.active_speaker_id is not None:
                break
        assert audio_filter.active_speaker_id == "alice"

        # Next chunk from same speaker passes through
        result = await audio_filter.process_audio(next(chunks), alice)
        assert result is not None

    async def test_lock_released_on_silence_timeout(
        self, audio_filter, mia_audio_16khz_chunked, silence_1s_16khz
    ):
        """Lock is released after sufficient silence from the active speaker."""
        alice = _participant("alice")

        for chunk in mia_audio_16khz_chunked:
            await audio_filter.process_audio(chunk, alice)
            if audio_filter.active_speaker_id is not None:
                break
        assert audio_filter.active_speaker_id == "alice"

        # 100ms / 20ms = 5 silence chunks to trigger release
        silence = _silence_chunk(silence_1s_16khz)
        for _ in range(5):
            await audio_filter.process_audio(silence, alice)

        assert audio_filter.active_speaker_id is None

    async def test_clear_releases_lock(self, audio_filter, mia_audio_16khz_chunked):
        """clear() resets the active speaker lock."""
        alice = _participant("alice")

        for chunk in mia_audio_16khz_chunked:
            await audio_filter.process_audio(chunk, alice)
            if audio_filter.active_speaker_id is not None:
                break
        assert audio_filter.active_speaker_id == "alice"

        audio_filter.clear()
        assert audio_filter.active_speaker_id is None

    async def test_second_speaker_can_acquire_after_release(
        self, audio_filter, mia_audio_16khz_chunked, silence_1s_16khz
    ):
        """After release, another speaker can acquire the lock."""
        alice = _participant("alice")
        bob = _participant("bob")

        # Alice acquires
        for chunk in mia_audio_16khz_chunked:
            await audio_filter.process_audio(chunk, alice)
            if audio_filter.active_speaker_id is not None:
                break
        assert audio_filter.active_speaker_id == "alice"

        # Alice goes silent -> lock released
        silence = _silence_chunk(silence_1s_16khz)
        for _ in range(5):
            await audio_filter.process_audio(silence, alice)
        assert audio_filter.active_speaker_id is None

        # Bob can now acquire
        for chunk in mia_audio_16khz_chunked:
            await audio_filter.process_audio(chunk, bob)
            if audio_filter.active_speaker_id is not None:
                break
        assert audio_filter.active_speaker_id == "bob"

    async def test_raises_if_not_warmed_up(self, silence_1s_16khz):
        """process_audio raises RuntimeError if warmup() was not called."""
        f = FirstSpeakerWinsFilter()
        alice = _participant("alice")

        with pytest.raises(RuntimeError, match="warmup"):
            await f.process_audio(_silence_chunk(silence_1s_16khz), alice)
