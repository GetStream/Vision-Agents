"""Unit tests for `TencentAudioTrack` chunking behaviour."""

import numpy as np
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.plugins.tencent.tracks import (
    BYTES_PER_20MS,
    CHANNELS,
    SAMPLE_RATE,
    TencentAudioTrack,
)


def _pcm_of_size(num_bytes: int, sample_rate: int = SAMPLE_RATE) -> PcmData:
    """Build a PcmData buffer of exactly ``num_bytes`` bytes of int16 PCM."""
    samples = np.zeros(num_bytes // 2, dtype=np.int16)
    return PcmData.from_numpy(
        samples,
        sample_rate=sample_rate,
        channels=CHANNELS,
        format=AudioFormat.S16,
    )


class TestTencentAudioTrackWriteSync:
    def test_exactly_one_frame_lands_in_queue(self) -> None:
        track = TencentAudioTrack()
        track._write_sync(_pcm_of_size(BYTES_PER_20MS))
        assert len(track._queue) == 1
        assert len(track._queue[0]) == BYTES_PER_20MS
        assert track._remainder == bytearray()

    def test_partial_frame_goes_to_remainder(self) -> None:
        track = TencentAudioTrack()
        track._write_sync(_pcm_of_size(BYTES_PER_20MS + 360))
        assert len(track._queue) == 1
        assert len(track._remainder) == 360

    def test_remainder_carries_into_next_write(self) -> None:
        track = TencentAudioTrack()
        # First write: 1 full frame + 360 bytes remainder.
        track._write_sync(_pcm_of_size(BYTES_PER_20MS + 360))
        # Second write: 280 bytes, which combined with the 360 remainder
        # exactly forms another full frame.
        track._write_sync(_pcm_of_size(280))
        assert len(track._queue) == 2
        assert track._remainder == bytearray()

    def test_empty_input_is_a_noop(self) -> None:
        track = TencentAudioTrack()
        track._write_sync(_pcm_of_size(0))
        assert len(track._queue) == 0
        assert track._remainder == bytearray()

    def test_offsample_pcm_is_resampled_before_chunking(self) -> None:
        track = TencentAudioTrack()
        # 48 kHz input is 3x the target rate, so 3 * 640 input bytes of silence
        # resample down to 640 bytes at 16 kHz. flush=True flushes the stateful
        # resampler's filter tail so the full 20 ms frame lands (end of utterance).
        track._write_sync(
            _pcm_of_size(BYTES_PER_20MS * 3, sample_rate=48000), flush=True
        )
        # After resample we have exactly one 20 ms frame's worth of audio
        # at 16 kHz — no remainder, one chunk.
        assert len(track._queue) == 1
        assert len(track._queue[0]) == BYTES_PER_20MS
        assert track._remainder == bytearray()
