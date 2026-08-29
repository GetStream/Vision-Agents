import asyncio
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

    async def test_pts_unchanged_by_draining(self, track: AvatarInputTrack) -> None:
        # pts() is the wire PTS the track will have reached once the buffer drains,
        # so writing moves it and recv() does not.
        await track.write(_audio(_SAMPLES_PER_FRAME * 3))
        await track.recv()
        assert await track.pts() == _SAMPLES_PER_FRAME * 3
        await track.recv()
        assert await track.pts() == _SAMPLES_PER_FRAME * 3

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
        start = time.monotonic()
        for _ in range(50):
            await track.recv()
        elapsed = time.monotonic() - start
        assert elapsed < 50 * AUDIO_PTIME * 0.5, (
            f"burst was paced: {elapsed:.4f}s for 50 frames"
        )

    async def test_recv_waits_for_audio_instead_of_emitting_silence(
        self, track: AvatarInputTrack
    ) -> None:
        pending = asyncio.create_task(track.recv())
        await asyncio.sleep(AUDIO_PTIME * 3)
        assert not pending.done(), "recv() emitted a frame while starved"

        await track.write(_audio(_SAMPLES_PER_FRAME))
        frame = await asyncio.wait_for(pending, timeout=1)
        assert frame.samples == _SAMPLES_PER_FRAME
        assert frame.pts == 0

    async def test_pts_contiguous_across_idle_gap(
        self, track: AvatarInputTrack
    ) -> None:
        # An idle gap costs no PTS: the next utterance continues the timeline.
        await track.write(_audio(_SAMPLES_PER_FRAME * 2))
        first = [(await track.recv()).pts for _ in range(2)]
        await asyncio.sleep(AUDIO_PTIME * 3)
        await track.write(_audio(_SAMPLES_PER_FRAME))
        resumed = await track.recv()
        assert first == [0, _SAMPLES_PER_FRAME]
        assert resumed.pts == _SAMPLES_PER_FRAME * 2

    async def test_partial_tail_is_emitted_unpadded(
        self, track: AvatarInputTrack
    ) -> None:
        await track.write(_audio(_SAMPLES_PER_FRAME // 2))
        frame = await track.recv()
        assert frame.samples == _SAMPLES_PER_FRAME // 2
        assert frame.pts == 0

    async def test_pts_advances_by_actual_samples_after_a_partial_tail(
        self, track: AvatarInputTrack
    ) -> None:
        # A short tail must not consume a full frame slot, or the timeline gains a hole.
        await track.write(_audio(_SAMPLES_PER_FRAME // 2))
        tail = await track.recv()
        await track.write(_audio(_SAMPLES_PER_FRAME))
        following = await track.recv()
        assert tail.pts == 0
        assert following.pts == _SAMPLES_PER_FRAME // 2

    async def test_interrupt_drops_pending_audio_and_keeps_waiting(
        self, track: AvatarInputTrack
    ) -> None:
        await track.write(_audio(_SAMPLES_PER_FRAME * 3))
        await track.recv()
        await track.flush()

        pending = asyncio.create_task(track.recv())
        await asyncio.sleep(AUDIO_PTIME * 3)
        assert not pending.done(), "flushed frames were still emitted"

        await track.write(_audio(_SAMPLES_PER_FRAME))
        frame = await asyncio.wait_for(pending, timeout=1)
        assert frame.pts == _SAMPLES_PER_FRAME
