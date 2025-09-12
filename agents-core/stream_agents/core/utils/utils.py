import numpy as np
from getstream.video.rtc.track_util import PcmData


def to_mono(samples: np.ndarray, num_channels: int) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels."""
    if num_channels == 1:
        return samples
    if samples.size % num_channels != 0:
        raise ValueError(
            f"Invalid sample array size {samples.size} for {num_channels} channels: "
            "Sample array size not divisible by number of channels"
        )
    samples = samples.reshape(-1, num_channels)
    mono_samples = np.mean(samples, axis=1, dtype=np.int16)
    # Ensure we always return an array, not a scalar
    return np.asarray(mono_samples, dtype=np.int16)


def bytes_to_pcm_data(
        audio_bytes: bytes,
        sample_rate: int = 16000,
        format: str = "s16"
) -> PcmData:
    """Convert raw bytes to PcmData object."""
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    return PcmData(samples=audio_array, sample_rate=sample_rate, format=format)
