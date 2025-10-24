"""Test to investigate resampling quality issues."""

import numpy as np
from vision_agents.core.edge.types import PcmData
import av


def test_compare_resampling_methods():
    """Compare PyAV resampling with scipy for quality."""
    # Create 1 second of clean sine wave at 16kHz
    sample_rate_in = 16000
    sample_rate_out = 48000
    duration = 1.0
    freq = 440  # A4 note

    num_samples_in = int(sample_rate_in * duration)
    t = np.linspace(0, duration, num_samples_in, dtype=np.float64)

    # Generate clean sine wave
    sine_wave = np.sin(2 * np.pi * freq * t)
    audio_int16 = (sine_wave * 32767).astype(np.int16)

    # Method 1: PyAV (current implementation)
    pcm_16k = PcmData(
        samples=audio_int16, sample_rate=sample_rate_in, format="s16", channels=1
    )
    pcm_48k_pyav = pcm_16k.resample(sample_rate_out, target_channels=1)

    print("\n=== PyAV Resampler ===")
    print(f"Input: {len(audio_int16)} samples @ {sample_rate_in}Hz")
    print(f"Output: {len(pcm_48k_pyav.samples)} samples @ {sample_rate_out}Hz")
    print(f"Output dtype: {pcm_48k_pyav.samples.dtype}")
    print(f"Output shape: {pcm_48k_pyav.samples.shape}")

    # Check for clipping or artifacts
    pyav_samples = (
        pcm_48k_pyav.samples.flatten()
        if pcm_48k_pyav.samples.ndim > 1
        else pcm_48k_pyav.samples
    )
    print(f"Output min: {pyav_samples.min()}, max: {pyav_samples.max()}")

    # Check for discontinuities (potential clicks)
    diffs = np.abs(np.diff(pyav_samples.astype(np.float32)))
    max_jump = np.max(diffs)
    mean_jump = np.mean(diffs)
    print(f"Max sample-to-sample jump: {max_jump:.1f}")
    print(f"Mean sample-to-sample jump: {mean_jump:.1f}")

    # Large jumps indicate clicks
    large_jumps = np.sum(diffs > 10000)
    print(f"Number of large jumps (>10000): {large_jumps}")

    # Method 2: Try scipy for comparison
    try:
        from scipy import signal

        # Resample using scipy's high-quality resampler
        num_samples_out = int(len(audio_int16) * sample_rate_out / sample_rate_in)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        resampled_scipy = signal.resample(audio_float, num_samples_out)
        resampled_scipy_int16 = (np.clip(resampled_scipy, -1.0, 1.0) * 32767).astype(
            np.int16
        )

        print("\n=== SciPy Resampler ===")
        print(f"Output: {len(resampled_scipy_int16)} samples @ {sample_rate_out}Hz")
        print(
            f"Output min: {resampled_scipy_int16.min()}, max: {resampled_scipy_int16.max()}"
        )

        diffs_scipy = np.abs(np.diff(resampled_scipy_int16.astype(np.float32)))
        max_jump_scipy = np.max(diffs_scipy)
        mean_jump_scipy = np.mean(diffs_scipy)
        print(f"Max sample-to-sample jump: {max_jump_scipy:.1f}")
        print(f"Mean sample-to-sample jump: {mean_jump_scipy:.1f}")

        large_jumps_scipy = np.sum(diffs_scipy > 10000)
        print(f"Number of large jumps (>10000): {large_jumps_scipy}")

        # Save both for manual inspection
        import wave
        import io

        def save_wav(samples, sr, filename):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(samples.tobytes())
            with open(filename, "wb") as f:
                f.write(buf.getvalue())

        save_wav(audio_int16, sample_rate_in, "/tmp/original_16k.wav")
        save_wav(pyav_samples.astype(np.int16), sample_rate_out, "/tmp/pyav_48k.wav")
        save_wav(resampled_scipy_int16, sample_rate_out, "/tmp/scipy_48k.wav")

        print("\nWAV files saved to /tmp/ for comparison")

    except ImportError:
        print("\nSciPy not available for comparison")


def test_pyav_resampler_settings():
    """Check if PyAV resampler has quality settings we're missing."""
    sample_rate_in = 16000
    sample_rate_out = 48000
    num_samples = 16000

    # Create test signal
    t = np.linspace(0, 1.0, num_samples, dtype=np.float64)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    # Create frame
    frame = av.AudioFrame.from_ndarray(
        audio.reshape(1, -1), format="s16p", layout="mono"
    )
    frame.sample_rate = sample_rate_in

    # Try different resampler configurations
    print("\n=== Testing PyAV Resampler Options ===")

    # Default resampler
    resampler_default = av.AudioResampler(
        format="s16", layout="mono", rate=sample_rate_out
    )

    print("Default resampler created")
    print(f"Resampler: {resampler_default}")

    # Check if there are any quality options available
    # Note: PyAV/FFmpeg's libswresample has quality options but might not be exposed

    frames = resampler_default.resample(frame)
    if frames:
        result = frames[0].to_ndarray().flatten()
        print(f"Default output: {len(result)} samples")

        diffs = np.abs(np.diff(result.astype(np.float32)))
        print(f"Max jump: {np.max(diffs):.1f}, Mean jump: {np.mean(diffs):.1f}")


if __name__ == "__main__":
    test_compare_resampling_methods()
    test_pyav_resampler_settings()
