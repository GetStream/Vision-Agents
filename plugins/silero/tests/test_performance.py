"""
Performance tests for Silero VAD implementation.

This module contains performance tests for the Silero VAD implementation,
measuring CPU time to ensure that changes don't significantly impact performance.
"""

import time
import asyncio
import numpy as np

import pytest

from vision_agents.plugins import silero
from vision_agents.core.vad.events import VADAudioEvent
from getstream.video.rtc.track_util import PcmData


@pytest.mark.asyncio
async def test_performance():
    """Test that Silero VAD processes audio with acceptable CPU time."""
    # Initialize the VAD
    vad = silero.VAD(
        sample_rate=16000,
        frame_size=512,
        activation_th=0.5,
        deactivation_th=0.5,
        speech_pad_ms=30,
        min_speech_ms=250,
        model_rate=16000,
        window_samples=512,  # Silero requires exactly 512 samples at 16kHz
    )

    # Create test audio with alternating speech and silence patterns
    # 5 seconds of audio at 16kHz
    duration_sec = 5
    sample_rate = 16000
    total_samples = duration_sec * sample_rate

    # Create alternating pattern of "speech" (random noise) and silence
    audio_data = np.zeros(total_samples, dtype=np.int16)

    # Add 10 segments of "speech" (random noise with amplitude 1000-10000)
    segment_length = int(total_samples / 20)
    for i in range(10):
        start = i * 2 * segment_length
        end = start + segment_length
        if end <= total_samples:
            audio_data[start:end] = np.random.randint(
                -10000, 10000, size=(end - start), dtype=np.int16
            )

    # Create PCM data
    pcm_data = PcmData(samples=audio_data, sample_rate=sample_rate, format="s16")

    # Counter for audio events
    audio_events = []

    @vad.events.subscribe
    async def on_audio(event: VADAudioEvent):
        audio_events.append(event)

    # Allow event subscription to be processed
    await asyncio.sleep(0.01)

    # Measure CPU time
    start_time = time.time()

    # Process audio
    await vad.process_audio(pcm_data)
    await vad.flush()
    await asyncio.sleep(0.1)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    print(
        f"Processing time: {elapsed_time:.3f} seconds for {duration_sec} seconds of audio"
    )
    print(f"Detected {len(audio_events)} audio events")

    # Estimate real-time factor (should be << 1.0 for real-time processing)
    rtf = elapsed_time / duration_sec
    print(f"Real-time factor: {rtf:.3f}")

    # A real-time factor below 0.3 is typically very good for VAD
    assert rtf < 1.0, f"Processing is too slow: RTF = {rtf:.3f}"

    # Clean up
    await vad.close()
