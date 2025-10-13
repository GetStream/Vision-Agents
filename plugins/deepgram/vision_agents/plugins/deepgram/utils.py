import numpy as np


def generate_silence(sample_rate: int, duration_ms: int) -> bytes:
    """
    Generate a silence of the given sample_rate and duration_ms.
    """
    # Audio parameters
    channels = 1
    sample_format = np.int16  # 16-bit signed PCM

    # Number of samples = sample_rate * duration_seconds
    num_samples = int(sample_rate * (duration_ms / 1000.0))

    # Create silence raw bytes (s16 mono PCM)
    pcm_bytes = np.zeros((num_samples, channels), dtype=sample_format).tobytes()
    return pcm_bytes

