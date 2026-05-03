import numpy as np
import pytest
from getstream.video.rtc import PcmData
from getstream.video.rtc.track_util import AudioFormat
from vision_agents.core.agents.inference.audio import (
    AudioOutputChunk,
    AudioOutputFlush,
    AudioOutputStream,
)


def make_pcm(ms: int, sample_rate: int = 16000, fill: int = 1) -> PcmData:
    num_samples = int(sample_rate * ms / 1000)
    samples = np.full(num_samples, fill, dtype=np.int16)
    return PcmData(samples=samples, sample_rate=sample_rate, format=AudioFormat.S16)


@pytest.fixture
def stream() -> AudioOutputStream:
    return AudioOutputStream()


class TestAudioOutputStream:
    async def test_exact_multiple_of_20ms_emits_that_many_chunks(
        self, stream: AudioOutputStream
    ):
        stream.send_nowait(AudioOutputChunk(data=make_pcm(40, fill=7)))
        items = stream.peek()
        assert len(items) == 2
        for item in items:
            assert isinstance(item, AudioOutputChunk)
            assert item.data is not None
            assert len(item.data.samples) == 320
            assert np.all(item.data.samples == 7)

    async def test_sub_20ms_input_emits_nothing(self, stream: AudioOutputStream):
        stream.send_nowait(AudioOutputChunk(data=make_pcm(10)))
        assert stream.empty()

    async def test_carry_is_prepended_on_next_send(self, stream: AudioOutputStream):
        stream.send_nowait(AudioOutputChunk(data=make_pcm(10)))
        stream.send_nowait(AudioOutputChunk(data=make_pcm(15)))
        items = stream.peek()
        assert len(items) == 1
        assert isinstance(items[0], AudioOutputChunk)
        assert items[0].data is not None
        assert len(items[0].data.samples) == 320

    async def test_chunk_size_tracks_sample_rate(self, stream: AudioOutputStream):
        stream.send_nowait(AudioOutputChunk(data=make_pcm(20, sample_rate=48000)))
        items = stream.peek()
        assert len(items) == 1
        assert isinstance(items[0], AudioOutputChunk)
        assert items[0].data is not None
        assert len(items[0].data.samples) == 960

    async def test_final_with_carry_pads_then_emits_terminal_marker(
        self, stream: AudioOutputStream
    ):
        stream.send_nowait(AudioOutputChunk(data=make_pcm(5, fill=100), final=True))
        items = stream.peek()
        assert len(items) == 2

        padded, terminal = items
        assert isinstance(padded, AudioOutputChunk)
        assert padded.final is False
        assert padded.data is not None
        assert len(padded.data.samples) == 320
        assert np.all(padded.data.samples[:80] == 100)
        assert np.all(padded.data.samples[80:] == 0)

        assert isinstance(terminal, AudioOutputChunk)
        assert terminal.final is True
        assert terminal.data is not None
        assert len(terminal.data.samples) == 0

    async def test_final_with_no_carry_emits_chunk_plus_marker(
        self, stream: AudioOutputStream
    ):
        stream.send_nowait(AudioOutputChunk(data=make_pcm(20, fill=9), final=True))
        items = stream.peek()
        assert len(items) == 2

        full, terminal = items
        assert isinstance(full, AudioOutputChunk)
        assert full.final is False
        assert full.data is not None
        assert len(full.data.samples) == 320
        assert np.all(full.data.samples == 9)

        assert isinstance(terminal, AudioOutputChunk)
        assert terminal.final is True
        assert terminal.data is not None
        assert len(terminal.data.samples) == 0

    async def test_carry_is_reset_after_final(self, stream: AudioOutputStream):
        stream.send_nowait(AudioOutputChunk(data=make_pcm(5), final=True))
        stream.clear()
        stream.send_nowait(AudioOutputChunk(data=make_pcm(10)))
        assert stream.empty()

    async def test_flush_passes_through_unchanged(self, stream: AudioOutputStream):
        flush = AudioOutputFlush()
        stream.send_nowait(flush)
        assert stream.peek() == [flush]

    async def test_chunk_with_none_data_passes_through_unchanged(
        self, stream: AudioOutputStream
    ):
        signal = AudioOutputChunk(data=None, final=True)
        stream.send_nowait(signal)
        assert stream.peek() == [signal]

    async def test_clear_drops_the_carry(self, stream: AudioOutputStream):
        stream.send_nowait(AudioOutputChunk(data=make_pcm(25)))
        stream.clear()
        stream.send_nowait(AudioOutputChunk(data=make_pcm(15)))
        assert stream.empty()

    async def test_buffered_reports_pending_seconds(self, stream: AudioOutputStream):
        assert stream.buffered == 0.0

        stream.send_nowait(AudioOutputChunk(data=make_pcm(40)))
        assert stream.buffered == pytest.approx(0.04)

        stream.send_nowait(AudioOutputChunk(data=make_pcm(20)))
        assert stream.buffered == pytest.approx(0.06)

    async def test_buffered_includes_carry(self, stream: AudioOutputStream):
        stream.send_nowait(AudioOutputChunk(data=make_pcm(10)))
        assert stream.empty()
        assert stream.buffered == pytest.approx(0.01)

    async def test_buffered_ignores_flush(self, stream: AudioOutputStream):
        stream.send_nowait(AudioOutputFlush())
        assert stream.buffered == 0.0

    async def test_buffered_ignores_chunk_with_none_data(
        self, stream: AudioOutputStream
    ):
        stream.send_nowait(AudioOutputChunk(data=None, final=True))
        assert stream.buffered == 0.0

    async def test_buffered_after_final_excludes_terminal_marker(
        self, stream: AudioOutputStream
    ):
        stream.send_nowait(AudioOutputChunk(data=make_pcm(20), final=True))
        # Stream now holds the real 20ms chunk plus a zero-sample terminal marker.
        assert len(stream.peek()) == 2
        # Only the real chunk contributes to buffered duration.
        assert stream.buffered == pytest.approx(0.02)
