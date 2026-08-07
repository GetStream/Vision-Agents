import time

import numpy as np
import pytest
from aiortc.mediastreams import AUDIO_PTIME
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.plugins.lemonslice.track import AvatarInputTrack

_SAMPLE_RATE = 16000
_CHANNELS = 1
_SAMPLES_PER_FRAME = int(AUDIO_PTIME * _SAMPLE_RATE)


def _audio(num_samples: int) -> PcmData:
    return PcmData(
        samples=np.zeros(num_samples, dtype=np.int16),
        sample_rate=_SAMPLE_RATE,
        format=AudioFormat.S16,
        channels=_CHANNELS,
    )


class TestStampedAudioTrack:
    @pytest.fixture
    def track(self) -> AvatarInputTrack:
        return AvatarInputTrack(sample_rate=_SAMPLE_RATE, channels=_CHANNELS)

    async def test_pts_zero_on_fresh_track(self, track: AvatarInputTrack) -> None:
        assert await track.pts() == 0

    async def test_pts_reflects_buffered_samples_before_any_recv(
        self, track: AvatarInputTrack
    ) -> None:
        await track.write(_audio(_SAMPLES_PER_FRAME * 3))
        assert await track.pts() == _SAMPLES_PER_FRAME * 3

    async def test_pts_after_emission_equals_last_buffered_frame_pts(
        self, track: AvatarInputTrack
    ) -> None:
        # Once recv() has begun, pts() = _timestamp + queued_samples = the wire
        # PTS the last buffered frame will carry when it's emitted.
        await track.write(_audio(_SAMPLES_PER_FRAME * 3))
        await track.recv()  # _timestamp=0, 2 frames queued
        assert await track.pts() == _SAMPLES_PER_FRAME * 2
        await track.recv()  # _timestamp=320, 1 frame queued
        assert await track.pts() == _SAMPLES_PER_FRAME * 2

    async def test_recv_frame_shape(self, track: AvatarInputTrack) -> None:
        await track.write(_audio(_SAMPLES_PER_FRAME))
        frame = await track.recv()
        assert frame.samples == _SAMPLES_PER_FRAME
        assert frame.sample_rate == _SAMPLE_RATE
        assert frame.pts == 0

    async def test_pts_advances_by_samples_per_frame(
        self, track: AvatarInputTrack
    ) -> None:
        await track.write(_audio(_SAMPLES_PER_FRAME * 5))
        pts_values = [(await track.recv()).pts for _ in range(5)]
        assert pts_values == [i * _SAMPLES_PER_FRAME for i in range(5)]

    async def test_burst_drains_without_pacing(self, track: AvatarInputTrack) -> None:
        # 1 second of audio = 50 frames; under wall-clock pacing this would take ~1s.
        await track.write(_audio(_SAMPLE_RATE))
        await track.recv()  # first recv initializes the timestamp baseline
        start = time.monotonic()
        for _ in range(49):
            await track.recv()
        elapsed = time.monotonic() - start
        assert elapsed < 49 * AUDIO_PTIME * 0.5, (
            f"burst was paced: {elapsed:.4f}s for 49 frames"
        )

    async def test_silence_paces_at_frame_interval(
        self, track: AvatarInputTrack
    ) -> None:
        await track.recv()  # init (first silence frame, also sleeps once)
        start = time.monotonic()
        await track.recv()
        elapsed = time.monotonic() - start
        assert AUDIO_PTIME * 0.5 < elapsed < AUDIO_PTIME * 3, (
            f"silence pacing off: {elapsed:.4f}s vs target {AUDIO_PTIME:.4f}s"
        )

    async def test_silence_after_burst_does_not_sleep_through_backlog(
        self, track: AvatarInputTrack
    ) -> None:
        # The drift edge case: PTS runs ~1s ahead of wall clock after the burst.
        # If we used the parent's wall-clock-anchored sleep, the next silence
        # frame would sleep ~1s to "catch up". We must sleep only AUDIO_PTIME.
        await track.write(_audio(_SAMPLE_RATE))  # 50 frames
        for _ in range(50):
            await track.recv()
        start = time.monotonic()
        frame = await track.recv()
        elapsed = time.monotonic() - start
        assert elapsed < AUDIO_PTIME * 5, (
            f"silence after burst slept through backlog: {elapsed:.4f}s"
        )
        assert frame.pts == 50 * _SAMPLES_PER_FRAME

    async def test_partial_frame_is_padded_and_bursted(
        self, track: AvatarInputTrack
    ) -> None:
        await track.write(_audio(_SAMPLES_PER_FRAME // 2))
        start = time.monotonic()
        frame = await track.recv()
        elapsed = time.monotonic() - start
        assert elapsed < AUDIO_PTIME * 0.5, f"partial frame was paced: {elapsed:.4f}s"
        assert frame.samples == _SAMPLES_PER_FRAME
        assert frame.pts == 0

    async def test_pts_monotonic_across_data_silence_data_transitions(
        self, track: AvatarInputTrack
    ) -> None:
        # data, then silence, then data — PTS continues advancing by samples_per_frame each step.
        await track.write(_audio(_SAMPLES_PER_FRAME * 2))
        f0 = await track.recv()
        f1 = await track.recv()
        f2 = await track.recv()  # silence
        await track.write(_audio(_SAMPLES_PER_FRAME))
        f3 = await track.recv()  # data again
        assert [f0.pts, f1.pts, f2.pts, f3.pts] == [
            i * _SAMPLES_PER_FRAME for i in range(4)
        ]

    async def test_long_silence_run_does_not_drift_to_zero_or_negative_sleep(
        self, track: AvatarInputTrack
    ) -> None:
        # Many silence frames in a row — each should still sleep ~AUDIO_PTIME,
        # not zero (no drift compensation collapsing the interval).
        await track.recv()  # init
        start = time.monotonic()
        for _ in range(5):
            await track.recv()
        elapsed = time.monotonic() - start
        assert elapsed > 5 * AUDIO_PTIME * 0.5, (
            f"silence loop did not pace: {elapsed:.4f}s for 5 frames"
        )
