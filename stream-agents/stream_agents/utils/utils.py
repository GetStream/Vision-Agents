import numpy as np


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
