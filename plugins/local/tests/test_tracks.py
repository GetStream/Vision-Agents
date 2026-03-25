"""Tests for local plugin audio tracks."""

import numpy as np
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.plugins.local.tracks import LocalOutputAudioTrack

from .conftest import _FakeAudioOutput


class TestLocalOutputAudioTrack:
    """Tests for LocalOutputAudioTrack class."""

    async def test_create_output_audio_track(self) -> None:
        output = _FakeAudioOutput(sample_rate=48000, channels=2)
        track = LocalOutputAudioTrack(audio_output=output)
        assert output.sample_rate == 48000
        assert output.channels == 2
        assert not track._running

    async def test_audio_track_start(self) -> None:
        output = _FakeAudioOutput(sample_rate=48000, channels=2)
        track = LocalOutputAudioTrack(audio_output=output)
        track.start()

        assert track._running
        assert track._playback_task is not None
        assert output.started

        track.stop()

    async def test_audio_track_write(self) -> None:
        output = _FakeAudioOutput(sample_rate=48000, channels=2)
        track = LocalOutputAudioTrack(audio_output=output)
        track.start()

        samples = np.array([100, 200, 300, 400], dtype=np.int16)
        pcm = PcmData(
            samples=samples,
            sample_rate=48000,
            format=AudioFormat.S16,
            channels=2,
        )

        await track.write(pcm)
        await output.wait_consumed()
        assert len(output.written) == 1

        track.stop()

    async def test_audio_track_stop(self) -> None:
        output = _FakeAudioOutput(sample_rate=48000, channels=2)
        track = LocalOutputAudioTrack(audio_output=output)
        track.start()
        track.stop()

        assert not track._running
        assert output.stopped

    async def test_audio_track_flush(self) -> None:
        output = _FakeAudioOutput(sample_rate=48000, channels=2)
        output._write_barrier.clear()
        track = LocalOutputAudioTrack(audio_output=output)
        track.start()

        samples = np.array([100, 200, 300, 400], dtype=np.int16)
        pcm = PcmData(
            samples=samples,
            sample_rate=48000,
            format=AudioFormat.S16,
            channels=2,
        )
        await track.write(pcm)
        await track.write(pcm)
        assert not track._queue.empty()

        await track.flush()
        assert track._queue.empty()

        output._write_barrier.set()
        track.stop()

    async def test_playback_task_processes_queue(self) -> None:
        output = _FakeAudioOutput(sample_rate=48000, channels=2)
        track = LocalOutputAudioTrack(audio_output=output)
        track.start()

        samples = np.array([100, 200, 300, 400], dtype=np.int16)
        pcm = PcmData(
            samples=samples,
            sample_rate=48000,
            format=AudioFormat.S16,
            channels=2,
        )
        await track.write(pcm)

        await output.wait_consumed()

        assert track._queue.empty()
        assert len(output.written) == 1

        track.stop()

    async def test_buffer_limit_configurable(self) -> None:
        output = _FakeAudioOutput(sample_rate=48000, channels=2)
        track = LocalOutputAudioTrack(audio_output=output, buffer_limit=5)
        assert track._queue.maxsize == 5

    async def test_resampling(self) -> None:
        output = _FakeAudioOutput(sample_rate=48000, channels=2)
        track = LocalOutputAudioTrack(audio_output=output)
        track.start()

        samples = np.array([100, 200, 300, 400], dtype=np.int16)
        pcm = PcmData(
            samples=samples,
            sample_rate=16000,
            format=AudioFormat.S16,
            channels=1,
        )

        await track.write(pcm)
        await output.wait_consumed()
        assert len(output.written) == 1

        track.stop()
