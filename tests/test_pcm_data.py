import numpy as np

from vision_agents.core.edge.types import PcmData


def _i16_list_from_bytes(b: bytes):
    return list(np.frombuffer(b, dtype=np.int16))


def test_to_bytes_interleaves_from_channel_major():
    # Create (channels, samples) data: L=[1,2,3,4], R=[-1,-2,-3,-4]
    samples = np.array(
        [
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
        ],
        dtype=np.int16,
    )
    pcm = PcmData(samples=samples, sample_rate=16000, format="s16", channels=2)
    out = _i16_list_from_bytes(pcm.to_bytes())
    assert out == [1, -1, 2, -2, 3, -3, 4, -4]


def test_to_bytes_interleaves_from_time_major():
    # Create (samples, channels) data: time-major
    time_major = np.array(
        [
            [1, -1],
            [2, -2],
            [3, -3],
            [4, -4],
        ],
        dtype=np.int16,
    )
    pcm = PcmData(samples=time_major, sample_rate=16000, format="s16", channels=2)
    out = _i16_list_from_bytes(pcm.to_bytes())
    assert out == [1, -1, 2, -2, 3, -3, 4, -4]


def test_resample_upmix_produces_channel_major_and_interleaved_bytes():
    # Mono ramp 1..10
    mono = np.arange(1, 11, dtype=np.int16)
    pcm_mono = PcmData(samples=mono, sample_rate=16000, format="s16", channels=1)

    # Upmix to stereo (same sample rate)
    pcm_stereo = pcm_mono.resample(16000, target_channels=2)
    assert pcm_stereo.channels == 2
    assert hasattr(pcm_stereo, "samples")
    assert isinstance(pcm_stereo.samples, np.ndarray)
    assert pcm_stereo.samples.ndim == 2
    # Expect (channels, samples) shape
    assert pcm_stereo.samples.shape[0] == 2
    # Sample count may be >= input due to resampler buffering; check prefix
    assert pcm_stereo.samples.shape[1] >= mono.shape[0]
    # Both channels should be identical after upmix
    assert np.array_equal(pcm_stereo.samples[0], pcm_stereo.samples[1])

    # Bytes should be interleaved L0,R0,L1,R1,...
    out_bytes = pcm_stereo.to_bytes()
    # Verify interleaving pattern: L[i] == R[i] for a prefix
    out_i16 = _i16_list_from_bytes(out_bytes)
    # take first 2 * N pairs (N from input)
    pairs = min(len(mono), len(out_i16) // 2)
    left = out_i16[0 : 2 * pairs : 2]
    right = out_i16[1 : 2 * pairs : 2]
    assert left == right


def test_resample_rate_and_stereo_size_scaling():
    # 200 mono samples @16kHz -> expect ~3x samples at 48kHz and x2 for stereo
    mono = np.arange(200, dtype=np.int16)
    pcm_mono = PcmData(samples=mono, sample_rate=16000, format="s16", channels=1)

    pcm_48k_stereo = pcm_mono.resample(48000, target_channels=2)
    out = pcm_48k_stereo.to_bytes()

    # 16-bit stereo -> 4 bytes per sample frame
    # 20ms at 48k is 960 frames = 3840 bytes; our total depends on input size
    # Sanity: output length should be >= input_bytes * 6 - small tolerance
    input_bytes = mono.nbytes
    assert len(out) >= input_bytes * 5  # conservative lower bound


# ===== Bug reproduction tests =====


def test_bug_mono_to_stereo_duration_preserved():
    """
    BUG REPRODUCTION: Converting mono to stereo should preserve duration.
    If duration changes, playback will be slowed down or sped up.
    """
    # Create 1 second of mono audio at 16kHz
    sample_rate = 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)

    # Generate a simple sine wave
    t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    pcm_mono = PcmData(samples=audio, sample_rate=sample_rate, format="s16", channels=1)

    # Check initial duration
    mono_duration = pcm_mono.duration
    print(f"\nMono duration: {mono_duration}s (expected ~1.0s)")
    assert abs(mono_duration - duration_sec) < 0.01, (
        f"Mono duration should be ~{duration_sec}s, got {mono_duration}s"
    )

    # Convert to stereo (no resampling, just channel conversion)
    pcm_stereo = pcm_mono.resample(sample_rate, target_channels=2)

    # Duration should be EXACTLY the same
    stereo_duration = pcm_stereo.duration
    print(f"Stereo duration: {stereo_duration}s (expected ~1.0s)")
    print(f"Stereo shape: {pcm_stereo.samples.shape} (expected (2, {num_samples}))")

    assert abs(stereo_duration - duration_sec) < 0.01, (
        f"Stereo duration should be ~{duration_sec}s, got {stereo_duration}s (BUG: playback will be wrong!)"
    )

    # Verify shape is correct (channels, samples)
    assert pcm_stereo.samples.shape[0] == 2, (
        f"First dimension should be channels (2), got shape {pcm_stereo.samples.shape}"
    )
    assert pcm_stereo.samples.shape[1] >= num_samples - 10, (
        f"Second dimension should be ~samples ({num_samples}), got shape {pcm_stereo.samples.shape}"
    )


def test_bug_resample_16khz_to_48khz_quality():
    """
    BUG REPRODUCTION: Resampling 16kHz to 48kHz should produce correct sample count.
    If sample count is wrong, quality will be bad.
    """
    # Create 1 second of mono audio at 16kHz
    sample_rate_in = 16000
    sample_rate_out = 48000
    duration_sec = 1.0
    num_samples_in = int(sample_rate_in * duration_sec)

    # Generate a simple sine wave
    t = np.linspace(0, duration_sec, num_samples_in, dtype=np.float32)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    pcm_16k = PcmData(
        samples=audio, sample_rate=sample_rate_in, format="s16", channels=1
    )

    # Resample to 48kHz
    pcm_48k = pcm_16k.resample(sample_rate_out, target_channels=1)

    # Check that sample count increased by 3x (48k/16k = 3)
    expected_samples = num_samples_in * 3
    actual_samples = (
        len(pcm_48k.samples) if pcm_48k.samples.ndim == 1 else pcm_48k.samples.shape[-1]
    )

    print(f"\n16kHz samples: {num_samples_in}")
    print(f"48kHz samples: {actual_samples} (expected ~{expected_samples})")
    print(f"48kHz shape: {pcm_48k.samples.shape}")
    print(f"48kHz duration: {pcm_48k.duration}s (expected ~1.0s)")

    # Allow some tolerance for resampler edge effects
    assert abs(actual_samples - expected_samples) < 100, (
        f"Expected ~{expected_samples} samples at 48kHz, got {actual_samples} (BUG: quality will be bad!)"
    )

    # Duration should remain the same
    assert abs(pcm_48k.duration - duration_sec) < 0.01, (
        f"Duration should remain ~{duration_sec}s, got {pcm_48k.duration}s"
    )


def test_bug_resample_16khz_to_48khz_stereo_combined():
    """
    BUG REPRODUCTION: The worst case - 16kHz mono to 48kHz stereo.
    This combines both bugs: resampling quality AND duration preservation.
    """
    # Create 1 second of mono audio at 16kHz
    sample_rate_in = 16000
    sample_rate_out = 48000
    duration_sec = 1.0
    num_samples_in = int(sample_rate_in * duration_sec)

    # Generate a simple sine wave
    t = np.linspace(0, duration_sec, num_samples_in, dtype=np.float32)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    pcm_16k = PcmData(
        samples=audio, sample_rate=sample_rate_in, format="s16", channels=1
    )

    # Resample to 48kHz stereo (the problematic case!)
    pcm_48k_stereo = pcm_16k.resample(sample_rate_out, target_channels=2)

    print(f"\n16kHz mono shape: {pcm_16k.samples.shape}")
    print(f"48kHz stereo shape: {pcm_48k_stereo.samples.shape}")
    print(f"48kHz stereo duration: {pcm_48k_stereo.duration}s (expected ~1.0s)")

    # Check shape is correct (channels, samples)
    assert pcm_48k_stereo.samples.ndim == 2, "Should be 2D array"
    assert pcm_48k_stereo.samples.shape[0] == 2, (
        f"First dimension should be channels (2), got shape {pcm_48k_stereo.samples.shape}"
    )

    # Check that sample count increased by 3x (48k/16k = 3)
    expected_samples = num_samples_in * 3
    actual_samples = pcm_48k_stereo.samples.shape[1]

    print(f"Expected ~{expected_samples} samples, got {actual_samples}")

    # Allow some tolerance for resampler edge effects
    assert abs(actual_samples - expected_samples) < 100, (
        f"Expected ~{expected_samples} samples at 48kHz, got {actual_samples}"
    )

    # Duration should remain the same (THIS IS THE CRITICAL BUG)
    assert abs(pcm_48k_stereo.duration - duration_sec) < 0.01, (
        f"Duration should remain ~{duration_sec}s, got {pcm_48k_stereo.duration}s (BUG: causes slow playback!)"
    )


def test_bug_duration_with_different_array_shapes():
    """
    BUG REPRODUCTION: Duration calculation should work with any array shape.
    The bug is that shape[-1] is used, which gives wrong results for (samples, channels) arrays.
    """
    sample_rate = 16000
    num_samples = 16000  # 1 second
    expected_duration = 1.0

    # Test 1: 1D array (mono) - should work
    samples_1d = np.zeros(num_samples, dtype=np.int16)
    pcm_1d = PcmData(
        samples=samples_1d, sample_rate=sample_rate, format="s16", channels=1
    )
    print(f"\n1D mono: shape={pcm_1d.samples.shape}, duration={pcm_1d.duration}s")
    assert abs(pcm_1d.duration - expected_duration) < 0.01

    # Test 2: 2D array (channels, samples) - CORRECT format, should work
    samples_2d_correct = np.zeros((2, num_samples), dtype=np.int16)
    pcm_2d_correct = PcmData(
        samples=samples_2d_correct, sample_rate=sample_rate, format="s16", channels=2
    )
    print(
        f"2D (channels, samples): shape={pcm_2d_correct.samples.shape}, duration={pcm_2d_correct.duration}s"
    )
    assert abs(pcm_2d_correct.duration - expected_duration) < 0.01

    # Test 3: 2D array (samples, channels) - WRONG format but might happen from PyAV
    # This is where the bug manifests!
    samples_2d_wrong = np.zeros((num_samples, 2), dtype=np.int16)
    pcm_2d_wrong = PcmData(
        samples=samples_2d_wrong, sample_rate=sample_rate, format="s16", channels=2
    )
    wrong_duration = pcm_2d_wrong.duration
    print(
        f"2D (samples, channels): shape={pcm_2d_wrong.samples.shape}, duration={wrong_duration}s"
    )

    # With current buggy code using shape[-1], this will give duration = 2/16000 = 0.000125s
    # But we want it to be 1.0s
    # This assertion will FAIL with the bug
    assert abs(wrong_duration - expected_duration) < 0.01, (
        f"Duration with (samples, channels) shape is WRONG: {wrong_duration}s (expected {expected_duration}s)"
    )
