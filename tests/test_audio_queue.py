import asyncio
import logging

import numpy as np
import pytest
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core.utils.audio_queue import AudioQueue


class TestAudioQueue:
    def test_audio_queue_initialization(self):
        """Test that AudioQueue initializes correctly."""
        queue = AudioQueue(buffer_limit_ms=1000)
        assert queue.buffer_limit_ms == 1000
        assert queue._total_samples == 0
        assert queue._sample_rate is None
        assert queue.empty()

    async def test_audio_queue_put_and_get(self):
        """Test basic put and get operations."""
        queue = AudioQueue(buffer_limit_ms=1000)

        # Create test audio data
        samples = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        pcm = PcmData(
            samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
        )

        await queue.put(pcm)

        assert queue.qsize() == 1
        assert queue._total_samples == 5
        assert queue._sample_rate == 16000

        retrieved = await queue.get()

        assert np.array_equal(retrieved.samples, samples)
        assert retrieved.sample_rate == 16000
        assert queue.empty()
        assert queue._total_samples == 0

    async def test_audio_queue_put_nowait(self):
        """Test put_nowait method."""
        queue = AudioQueue(buffer_limit_ms=1000)

        samples = np.array([1, 2, 3], dtype=np.int16)
        pcm = PcmData(
            samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
        )

        queue.put_nowait(pcm)

        assert queue.qsize() == 1
        assert queue._total_samples == 3

    async def test_audio_queue_get_nowait(self):
        """Test get_nowait method."""
        queue = AudioQueue(buffer_limit_ms=1000)

        samples = np.array([1, 2, 3], dtype=np.int16)
        pcm = PcmData(
            samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
        )

        queue.put_nowait(pcm)
        retrieved = queue.get_nowait()

        assert np.array_equal(retrieved.samples, samples)
        assert queue.empty()

    async def test_audio_queue_get_nowait_empty(self):
        """Test get_nowait raises exception when queue is empty."""
        queue = AudioQueue(buffer_limit_ms=1000)

        with pytest.raises(asyncio.QueueEmpty):
            queue.get_nowait()

    async def test_audio_queue_buffer_limit_warning(self, caplog):
        """Test that a warning is logged when buffer limit is exceeded."""
        queue = AudioQueue(buffer_limit_ms=100)  # 100ms limit

        # At 16kHz, 100ms = 1600 samples
        # Create audio that's 150ms (2400 samples)
        samples = np.zeros(2400, dtype=np.int16)
        pcm = PcmData(
            samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
        )

        with caplog.at_level(logging.WARNING):
            await queue.put(pcm)

        assert "buffer limit exceeded" in caplog.text.lower()

    async def test_audio_queue_type_checking(self):
        """Test that AudioQueue only accepts PcmData."""
        queue = AudioQueue(buffer_limit_ms=1000)

        with pytest.raises(TypeError):
            await queue.put("not audio data")

        with pytest.raises(TypeError):
            queue.put_nowait(123)

    async def test_audio_queue_sample_rate_mismatch_warning(self, caplog):
        """Test warning when adding audio with different sample rates."""
        queue = AudioQueue(buffer_limit_ms=1000)

        # First add 16kHz audio
        samples1 = np.array([1, 2, 3], dtype=np.int16)
        pcm1 = PcmData(
            samples=samples1, sample_rate=16000, format=AudioFormat.S16, channels=1
        )
        await queue.put(pcm1)

        # Then add 48kHz audio (should warn)
        samples2 = np.array([4, 5, 6], dtype=np.int16)
        pcm2 = PcmData(
            samples=samples2, sample_rate=48000, format=AudioFormat.S16, channels=1
        )

        with caplog.at_level(logging.WARNING):
            await queue.put(pcm2)

        assert "sample rate mismatch" in caplog.text.lower()

    async def test_audio_queue_get_samples_exact(self):
        """Test getting an exact number of samples."""
        queue = AudioQueue(buffer_limit_ms=1000)

        # Add 10 samples
        samples = np.arange(10, dtype=np.int16)
        pcm = PcmData(
            samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
        )
        await queue.put(pcm)

        # Get 5 samples
        result = await queue.get_samples(5)

        assert len(result.samples) == 5
        assert np.array_equal(result.samples, np.arange(5, dtype=np.int16))
        assert result.sample_rate == 16000

        # Queue should still have 5 samples
        assert queue.qsize() == 1
        assert queue._total_samples == 5

    async def test_audio_queue_get_samples_multiple_chunks(self):
        """Test getting samples that span multiple audio chunks."""
        queue = AudioQueue(buffer_limit_ms=1000)

        # Add three chunks of 5 samples each
        for i in range(3):
            samples = np.arange(i * 5, (i + 1) * 5, dtype=np.int16)
            pcm = PcmData(
                samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
            )
            await queue.put(pcm)

        # Get 12 samples (should span 3 chunks)
        result = await queue.get_samples(12)

        assert len(result.samples) == 12
        assert np.array_equal(result.samples, np.arange(12, dtype=np.int16))

        # Should have 3 samples left
        assert queue._total_samples == 3

    async def test_audio_queue_get_samples_partial_chunk(self):
        """Test getting samples that require splitting a chunk."""
        queue = AudioQueue(buffer_limit_ms=1000)

        # Add 10 samples
        samples = np.arange(10, dtype=np.int16)
        pcm = PcmData(
            samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
        )
        await queue.put(pcm)

        # Get 7 samples (leaves 3)
        result1 = await queue.get_samples(7)
        assert len(result1.samples) == 7
        assert np.array_equal(result1.samples, np.arange(7, dtype=np.int16))

        # Get remaining 3 samples
        result2 = await queue.get_samples(3)
        assert len(result2.samples) == 3
        assert np.array_equal(result2.samples, np.arange(7, 10, dtype=np.int16))

        assert queue.empty()

    async def test_audio_queue_get_samples_more_than_available(self):
        """Test getting more samples than available returns what's available."""
        queue = AudioQueue(buffer_limit_ms=1000)

        # Add 5 samples
        samples = np.arange(5, dtype=np.int16)
        pcm = PcmData(
            samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
        )
        await queue.put(pcm)

        # Request 10 samples (only 5 available)
        result = await queue.get_samples(10)

        assert len(result.samples) == 5
        assert np.array_equal(result.samples, np.arange(5, dtype=np.int16))
        assert queue.empty()

    async def test_audio_queue_get_samples_empty_queue(self):
        """Test getting samples from empty queue raises exception."""
        queue = AudioQueue(buffer_limit_ms=1000)

        with pytest.raises(asyncio.QueueEmpty):
            await queue.get_samples(10)

    async def test_audio_queue_get_duration(self):
        """Test getting audio by duration."""
        queue = AudioQueue(buffer_limit_ms=1000)

        # At 16kHz, 100ms = 1600 samples
        samples = np.arange(3200, dtype=np.int16)  # 200ms of audio
        pcm = PcmData(
            samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
        )
        await queue.put(pcm)

        # Get 100ms of audio
        result = await queue.get_duration(100)

        # Should get 1600 samples
        assert len(result.samples) == 1600
        assert result.sample_rate == 16000

        # Should have 100ms (1600 samples) remaining
        assert queue._total_samples == 1600

    async def test_audio_queue_get_duration_multiple_chunks(self):
        """Test getting duration that spans multiple chunks."""
        queue = AudioQueue(buffer_limit_ms=1000)

        # Add four chunks of 50ms each (800 samples each at 16kHz)
        for i in range(4):
            samples = np.arange(i * 800, (i + 1) * 800, dtype=np.int16)
            pcm = PcmData(
                samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
            )
            await queue.put(pcm)

        # Get 150ms of audio (2400 samples)
        result = await queue.get_duration(150)

        assert len(result.samples) == 2400
        assert np.array_equal(result.samples, np.arange(2400, dtype=np.int16))

    async def test_audio_queue_get_duration_more_than_available(self):
        """Test getting more duration than available."""
        queue = AudioQueue(buffer_limit_ms=1000)

        # Add 50ms of audio at 16kHz (800 samples)
        samples = np.arange(800, dtype=np.int16)
        pcm = PcmData(
            samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
        )
        await queue.put(pcm)

        # Request 100ms (should get only 50ms)
        result = await queue.get_duration(100)

        assert len(result.samples) == 800  # Only 50ms worth
        assert queue.empty()

    async def test_audio_queue_get_buffer_info(self):
        """Test getting buffer information."""
        queue = AudioQueue(buffer_limit_ms=1000)

        # Initially empty
        info = queue.get_buffer_info()
        assert info["buffer_limit_ms"] == 1000
        assert info["current_duration_ms"] == 0
        assert info["total_samples"] == 0
        assert info["sample_rate"] is None
        assert info["num_chunks"] == 0

        # Add 100ms of audio (1600 samples at 16kHz)
        samples = np.arange(1600, dtype=np.int16)
        pcm = PcmData(
            samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
        )
        await queue.put(pcm)

        info = queue.get_buffer_info()
        assert info["current_duration_ms"] == 100.0
        assert info["total_samples"] == 1600
        assert info["sample_rate"] == 16000
        assert info["num_chunks"] == 1

    async def test_audio_queue_mixed_operations(self):
        """Test mixing different get operations."""
        queue = AudioQueue(buffer_limit_ms=1000)

        # Add 200ms of audio (3200 samples at 16kHz)
        samples = np.arange(3200, dtype=np.int16)
        pcm = PcmData(
            samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
        )
        await queue.put(pcm)

        # Get 50ms (800 samples)
        result1 = await queue.get_duration(50)
        assert len(result1.samples) == 800

        # Get 400 samples directly
        result2 = await queue.get_samples(400)
        assert len(result2.samples) == 400

        # Get 25ms (400 samples)
        result3 = await queue.get_duration(25)
        assert len(result3.samples) == 400

        # Should have 1600 samples remaining
        assert queue._total_samples == 1600

    async def test_audio_queue_concurrent_operations(self):
        """Test concurrent put and get operations."""
        queue = AudioQueue(buffer_limit_ms=5000)

        async def producer():
            for i in range(10):
                samples = np.full(160, i, dtype=np.int16)  # 10ms chunks at 16kHz
                pcm = PcmData(
                    samples=samples,
                    sample_rate=16000,
                    format=AudioFormat.S16,
                    channels=1,
                )
                await queue.put(pcm)
                await asyncio.sleep(0.001)

        async def consumer():
            results = []
            for _ in range(5):
                result = await queue.get_duration(10)  # Get 10ms chunks
                results.append(result)
            return results

        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())

        results = await consumer_task
        await producer_task

        assert len(results) == 5
        for result in results:
            assert len(result.samples) == 160  # 10ms at 16kHz

    async def test_audio_queue_empty_buffer_duration(self):
        """Test that empty queue reports 0 duration."""
        queue = AudioQueue(buffer_limit_ms=1000)
        assert queue._current_duration_ms() == 0.0

        info = queue.get_buffer_info()
        assert info["current_duration_ms"] == 0.0

    async def test_audio_queue_different_sample_rates(self):
        """Test queue behavior with different sample rates."""
        queue = AudioQueue(buffer_limit_ms=1000)

        # 100ms at 8kHz = 800 samples
        samples1 = np.arange(800, dtype=np.int16)
        pcm1 = PcmData(
            samples=samples1, sample_rate=8000, format=AudioFormat.S16, channels=1
        )
        await queue.put(pcm1)

        info = queue.get_buffer_info()
        assert info["sample_rate"] == 8000
        assert info["current_duration_ms"] == 100.0

        # Getting duration should work with the established sample rate
        result = await queue.get_duration(50)  # 50ms = 400 samples at 8kHz
        assert len(result.samples) == 400

    async def test_audio_queue_large_buffer(self):
        """Test queue with larger audio buffers."""
        queue = AudioQueue(buffer_limit_ms=10000)  # 10 second limit

        # Add 1 second of audio at 48kHz (48000 samples)
        samples = np.random.randint(-32768, 32767, size=48000, dtype=np.int16)
        pcm = PcmData(
            samples=samples, sample_rate=48000, format=AudioFormat.S16, channels=1
        )
        await queue.put(pcm)

        info = queue.get_buffer_info()
        assert info["current_duration_ms"] == pytest.approx(1000.0, rel=0.01)
        assert info["total_samples"] == 48000

        # Get 500ms
        result = await queue.get_duration(500)
        assert len(result.samples) == 24000  # 500ms at 48kHz

    async def test_audio_queue_preserve_data_integrity(self):
        """Test that audio data is preserved correctly through queue operations."""
        queue = AudioQueue(buffer_limit_ms=1000)

        # Create audio with specific pattern
        original_samples = np.array(
            [100, 200, 300, 400, 500, 600, 700, 800], dtype=np.int16
        )
        pcm = PcmData(
            samples=original_samples,
            sample_rate=16000,
            format=AudioFormat.S16,
            channels=1,
        )
        await queue.put(pcm)

        # Get partial samples
        result1 = await queue.get_samples(3)
        assert np.array_equal(result1.samples, [100, 200, 300])

        result2 = await queue.get_samples(3)
        assert np.array_equal(result2.samples, [400, 500, 600])

        result3 = await queue.get_samples(2)
        assert np.array_equal(result3.samples, [700, 800])

        assert queue.empty()

    async def test_put_drops_oldest_when_limit_exceeded(self):
        """Test that put() drops oldest chunks when the buffer limit would be exceeded."""
        queue = AudioQueue(buffer_limit_ms=100)  # 100ms = 1600 samples at 16kHz

        # Fill queue with 5 chunks of 20ms each (320 samples) = exactly 100ms
        chunks = []
        for i in range(5):
            samples = np.full(320, i, dtype=np.int16)
            pcm = PcmData(
                samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
            )
            chunks.append(pcm)
            await queue.put(pcm)

        assert queue._total_samples == 1600  # 100ms worth

        # Adding one more 20ms chunk should trigger dropping the oldest
        overflow = PcmData(
            samples=np.full(320, 99, dtype=np.int16),
            sample_rate=16000,
            format=AudioFormat.S16,
            channels=1,
        )
        await queue.put(overflow)

        # Buffer should still be at or under the limit
        duration_ms = (queue._total_samples / 16000) * 1000
        assert duration_ms <= 100.0

        # The oldest chunk (value=0) should have been dropped; newest (value=99) is present
        first = await queue.get()
        assert first.samples[0] != 0  # oldest was dropped

    async def test_put_nowait_drops_oldest_when_limit_exceeded(self):
        """Test that put_nowait() drops oldest chunks when the buffer limit would be exceeded."""
        queue = AudioQueue(buffer_limit_ms=100)

        for i in range(5):
            samples = np.full(320, i, dtype=np.int16)
            pcm = PcmData(
                samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
            )
            queue.put_nowait(pcm)

        assert queue._total_samples == 1600

        overflow = PcmData(
            samples=np.full(320, 99, dtype=np.int16),
            sample_rate=16000,
            format=AudioFormat.S16,
            channels=1,
        )
        queue.put_nowait(overflow)

        duration_ms = (queue._total_samples / 16000) * 1000
        assert duration_ms <= 100.0

        first = queue.get_nowait()
        assert first.samples[0] != 0

    async def test_drop_logs_dropped_count_and_duration(self, caplog):
        """Test that the drop warning includes chunk count and duration."""
        queue = AudioQueue(buffer_limit_ms=100)

        # Fill to capacity: 5 * 320 = 1600 samples = 100ms
        for _ in range(5):
            samples = np.zeros(320, dtype=np.int16)
            pcm = PcmData(
                samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
            )
            await queue.put(pcm)

        with caplog.at_level(logging.WARNING):
            overflow = PcmData(
                samples=np.zeros(320, dtype=np.int16),
                sample_rate=16000,
                format=AudioFormat.S16,
                channels=1,
            )
            await queue.put(overflow)

        assert "dropped 1 chunks" in caplog.text
        assert "20.0ms" in caplog.text

    async def test_drop_clears_not_empty_when_buffer_emptied(self):
        """Test that _not_empty is cleared if all chunks are dropped."""
        # Use a very tight limit: 10ms = 160 samples at 16kHz
        queue = AudioQueue(buffer_limit_ms=10)

        # Add a small chunk (10ms)
        small = PcmData(
            samples=np.zeros(160, dtype=np.int16),
            sample_rate=16000,
            format=AudioFormat.S16,
            channels=1,
        )
        await queue.put(small)
        assert queue._not_empty.is_set()

        # Add a chunk larger than the limit — existing chunk must be dropped
        big = PcmData(
            samples=np.zeros(160, dtype=np.int16),
            sample_rate=16000,
            format=AudioFormat.S16,
            channels=1,
        )
        await queue.put(big)

        # The new item was appended so _not_empty should be set again
        assert queue._not_empty.is_set()
        assert queue.qsize() == 1

    async def test_massive_overflow_drops_multiple_chunks(self):
        """Test that a large overflow drops enough chunks to fit within the limit."""
        queue = AudioQueue(buffer_limit_ms=100)  # 1600 samples at 16kHz

        # Fill with 10 chunks of 10ms each (160 samples) = 100ms total
        for i in range(10):
            samples = np.full(160, i, dtype=np.int16)
            pcm = PcmData(
                samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
            )
            await queue.put(pcm)

        assert queue._total_samples == 1600

        # Add a 50ms chunk (800 samples) — need to drop at least 50ms of old data
        big_chunk = PcmData(
            samples=np.full(800, 99, dtype=np.int16),
            sample_rate=16000,
            format=AudioFormat.S16,
            channels=1,
        )
        await queue.put(big_chunk)

        duration_ms = (queue._total_samples / 16000) * 1000
        assert duration_ms <= 100.0

        # Verify oldest chunks were the ones dropped (values 0-4 gone)
        remaining_values = []
        while not queue.empty():
            item = await queue.get()
            remaining_values.append(item.samples[0])

        assert 99 in remaining_values  # new chunk is present
        assert 0 not in remaining_values  # oldest was dropped

    async def test_audio_queue_clear_empty(self):
        """Test clear() on an empty AudioQueue."""
        queue = AudioQueue(buffer_limit_ms=1000)
        queue.clear()
        assert queue.qsize() == 0
        assert queue.empty()

    async def test_audio_queue_clear_non_empty(self):
        """Test clear() removes existing data from AudioQueue."""
        queue = AudioQueue(buffer_limit_ms=1000)

        # Create test audio data
        samples = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        pcm = PcmData(
            samples=samples, sample_rate=16000, format=AudioFormat.S16, channels=1
        )

        await queue.put(pcm)
        assert queue.qsize() > 0

        queue.clear()
        assert queue.qsize() == 0
        assert queue.empty()
