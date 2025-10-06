"""
Benchmark tests for Silero VAD performance.

This module provides tests for benchmarking the Silero VAD implementation,
processing 10 seconds of audio and reporting RTF (Real-Time Factor) and other metrics.
"""

import asyncio
import time
import numpy as np
import logging
import pytest
import soundfile as sf
from vision_agents.plugins import silero
from vision_agents.core.vad.events import VADAudioEvent
from getstream.video.rtc.track_util import PcmData
from plugins.plugin_test_utils import get_audio_asset


async def benchmark_vad(use_onnx=False, device="cpu"):
    """
    Benchmark the Silero VAD implementation.

    Args:
        use_onnx: Whether to use ONNX runtime
        device: Device to run on ("cpu", "cuda", etc.)

    Returns:
        Dictionary with benchmark results
    """
    logging.basicConfig(level=logging.ERROR)  # Suppress logs during benchmark

    # Initialize the VAD
    vad = silero.VAD(
        sample_rate=16000,
        window_samples=512,
        activation_th=0.3,
        deactivation_th=0.2,
        speech_pad_ms=30,
        min_speech_ms=250,
        model_rate=16000,
        device=device,
        use_onnx=use_onnx,
    )

    # Create 10 seconds of audio (mixture of speech-like signals)
    try:
        # Try to use real speech data first
        audio_path = get_audio_asset("formant_speech_16k.wav")
        audio_data, sample_rate = sf.read(audio_path, dtype="int16")
    except Exception:
        # Fall back to synthetic data if real data not available
        sample_rate = 16000
        duration_sec = 10
        t = np.linspace(0, duration_sec, int(duration_sec * sample_rate))
        # Create speech-like signal with formants
        audio_data = np.zeros_like(t, dtype=np.int16)

        for formant, amplitude in [(600, 1.0), (1200, 0.5), (2400, 0.2)]:
            audio_data += (amplitude * 32767 * np.sin(2 * np.pi * formant * t)).astype(
                np.int16
            )

    # Ensure we have 10 seconds of audio
    if len(audio_data) < sample_rate * 10:
        # Repeat the audio to get at least 10 seconds
        repeats = int(np.ceil(sample_rate * 10 / len(audio_data)))
        audio_data = np.tile(audio_data, repeats)

    # Trim to exactly 10 seconds
    audio_data = audio_data[: sample_rate * 10]

    # Process audio in small chunks to simulate streaming (20ms chunks)
    chunk_size = int(sample_rate * 0.02)  # 20ms chunks
    chunks = [
        audio_data[i : i + chunk_size] for i in range(0, len(audio_data), chunk_size)
    ]

    # Capture detected speech segments
    detected_speech = []

    @vad.events.subscribe
    async def on_audio(event: VADAudioEvent, user=None):
        detected_speech.append(event)

    # Allow event subscription to be processed
    await asyncio.sleep(0.01)

    # Measure processing time
    start_time = time.time()

    # Process audio chunks
    for chunk in chunks:
        await vad.process_audio(
            PcmData(samples=chunk, sample_rate=sample_rate, format="s16")
        )

    # Flush any remaining speech
    await vad.flush()

    # Calculate total processing time
    total_time = time.time() - start_time

    # Calculate Real-Time Factor (RTF)
    audio_duration = len(audio_data) / sample_rate
    rtf = total_time / audio_duration

    # Clean up
    await vad.close()

    # Return benchmark results
    result = {
        "device": device,
        "use_onnx": use_onnx,
        "audio_duration_sec": audio_duration,
        "processing_time_sec": total_time,
        "rtf": rtf,
        "speech_segments": len(detected_speech),
    }

    return result


@pytest.mark.asyncio
async def test_vad_benchmark():
    """Test Silero VAD benchmark performance."""
    # Run benchmark
    results = await benchmark_vad(use_onnx=False, device="cpu")

    # Print results for visibility in test output
    print("\nSilero VAD Benchmark Results:")
    print(f"  Device: {results['device']}")
    print(f"  ONNX: {results['use_onnx']}")
    print(f"  Audio duration: {results['audio_duration_sec']:.2f} seconds")
    print(f"  Processing time: {results['processing_time_sec']:.2f} seconds")
    print(f"  Real-Time Factor (RTF): {results['rtf']:.3f}x")
    print(f"  Speech segments detected: {results['speech_segments']}")

    # Verify that the VAD is reasonably efficient (RTF < 0.5 means processing is at least 2x faster than real-time)
    # This threshold might need adjustment based on your specific hardware/environment
    assert results["rtf"] < 2.0, f"VAD performance too slow: RTF = {results['rtf']:.3f}"


@pytest.mark.asyncio
async def test_vad_benchmark_onnx():
    """Test Silero VAD benchmark performance with ONNX runtime."""
    # Skip if ONNX isn't available
    try:
        import importlib.util

        if importlib.util.find_spec("onnxruntime") is None:
            pytest.skip("ONNX runtime not available")
    except ImportError:
        pytest.skip("Error importing libraries needed to check for ONNX runtime")

    # Run benchmark with ONNX
    results = await benchmark_vad(use_onnx=True, device="cpu")

    # Print results for visibility in test output
    print("\nSilero VAD ONNX Benchmark Results:")
    print(f"  Device: {results['device']}")
    print(f"  ONNX: {results['use_onnx']}")
    print(f"  Audio duration: {results['audio_duration_sec']:.2f} seconds")
    print(f"  Processing time: {results['processing_time_sec']:.2f} seconds")
    print(f"  Real-Time Factor (RTF): {results['rtf']:.3f}x")
    print(f"  Speech segments detected: {results['speech_segments']}")

    # Verify that the ONNX implementation is reasonably efficient
    assert results["rtf"] < 2.0, (
        f"ONNX VAD performance too slow: RTF = {results['rtf']:.3f}"
    )
