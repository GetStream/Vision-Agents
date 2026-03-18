"""Tests for local plugin audio tracks."""

import asyncio

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
        assert not track._stopped

    async def test_audio_track_start(self) -> None:
        output = _FakeAudioOutput(sample_rate=48000, channels=2)
        track = LocalOutputAudioTrack(audio_output=output)
        track.start()

        assert track._running
        assert track._playback_thread is not None
        assert track._playback_thread.is_alive()
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
        assert not track._buffer.empty()

    async def test_audio_track_stop(self) -> None:
        output = _FakeAudioOutput(sample_rate=48000, channels=2)
        track = LocalOutputAudioTrack(audio_output=output)
        track.start()
        track.stop()

        assert track._stopped
        assert not track._running
        assert output.stopped

    async def test_audio_track_flush(self) -> None:
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
        assert not track._buffer.empty()

        await track.flush()
        assert track._buffer.empty()

    async def test_playback_thread_processes_queue(self) -> None:
        output = _FakeAudioOutput(sample_rate=48000, channels=2)
        track = LocalOutputAudioTrack(audio_output=output)
        track.start()

        test_data = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=np.int16)
        track._buffer.put(test_data)

        await asyncio.sleep(0.2)

        assert track._buffer.empty()
        assert len(output.written) == 1

        track.stop()

    async def test_buffer_has_duration_limit(self) -> None:
        output = _FakeAudioOutput(sample_rate=48000, channels=2)
        track = LocalOutputAudioTrack(audio_output=output, buffer_limit_ms=2000)
        assert track._buffer._buffer_limit_ms == 2000

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
        assert not track._buffer.empty()
