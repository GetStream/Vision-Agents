"""Audio conversion utilities for Twilio mulaw/PCM."""

import warnings

import numpy as np
from getstream.video.rtc.track_util import PcmData, AudioFormat

TWILIO_SAMPLE_RATE = 8000  # Twilio streams mulaw at 8kHz

# Precompute mulaw decoding table (ITU-T G.711)
MULAW_DECODE_TABLE = np.array(
    [
        -32124,
        -31100,
        -30076,
        -29052,
        -28028,
        -27004,
        -25980,
        -24956,
        -23932,
        -22908,
        -21884,
        -20860,
        -19836,
        -18812,
        -17788,
        -16764,
        -15996,
        -15484,
        -14972,
        -14460,
        -13948,
        -13436,
        -12924,
        -12412,
        -11900,
        -11388,
        -10876,
        -10364,
        -9852,
        -9340,
        -8828,
        -8316,
        -7932,
        -7676,
        -7420,
        -7164,
        -6908,
        -6652,
        -6396,
        -6140,
        -5884,
        -5628,
        -5372,
        -5116,
        -4860,
        -4604,
        -4348,
        -4092,
        -3900,
        -3772,
        -3644,
        -3516,
        -3388,
        -3260,
        -3132,
        -3004,
        -2876,
        -2748,
        -2620,
        -2492,
        -2364,
        -2236,
        -2108,
        -1980,
        -1884,
        -1820,
        -1756,
        -1692,
        -1628,
        -1564,
        -1500,
        -1436,
        -1372,
        -1308,
        -1244,
        -1180,
        -1116,
        -1052,
        -988,
        -924,
        -876,
        -844,
        -812,
        -780,
        -748,
        -716,
        -684,
        -652,
        -620,
        -588,
        -556,
        -524,
        -492,
        -460,
        -428,
        -396,
        -372,
        -356,
        -340,
        -324,
        -308,
        -292,
        -276,
        -260,
        -244,
        -228,
        -212,
        -196,
        -180,
        -164,
        -148,
        -132,
        -120,
        -112,
        -104,
        -96,
        -88,
        -80,
        -72,
        -64,
        -56,
        -48,
        -40,
        -32,
        -24,
        -16,
        -8,
        0,
        32124,
        31100,
        30076,
        29052,
        28028,
        27004,
        25980,
        24956,
        23932,
        22908,
        21884,
        20860,
        19836,
        18812,
        17788,
        16764,
        15996,
        15484,
        14972,
        14460,
        13948,
        13436,
        12924,
        12412,
        11900,
        11388,
        10876,
        10364,
        9852,
        9340,
        8828,
        8316,
        7932,
        7676,
        7420,
        7164,
        6908,
        6652,
        6396,
        6140,
        5884,
        5628,
        5372,
        5116,
        4860,
        4604,
        4348,
        4092,
        3900,
        3772,
        3644,
        3516,
        3388,
        3260,
        3132,
        3004,
        2876,
        2748,
        2620,
        2492,
        2364,
        2236,
        2108,
        1980,
        1884,
        1820,
        1756,
        1692,
        1628,
        1564,
        1500,
        1436,
        1372,
        1308,
        1244,
        1180,
        1116,
        1052,
        988,
        924,
        876,
        844,
        812,
        780,
        748,
        716,
        684,
        652,
        620,
        588,
        556,
        524,
        492,
        460,
        428,
        396,
        372,
        356,
        340,
        324,
        308,
        292,
        276,
        260,
        244,
        228,
        212,
        196,
        180,
        164,
        148,
        132,
        120,
        112,
        104,
        96,
        88,
        80,
        72,
        64,
        56,
        48,
        40,
        32,
        24,
        16,
        8,
        0,
    ],
    dtype=np.int16,
)


def _build_mulaw_encode_table() -> np.ndarray:
    """Build a lookup table for PCM to mu-law encoding."""
    table = np.zeros(65536, dtype=np.uint8)

    # Try to use audioop (available in Python < 3.13) for reference-quality encoding
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import audioop

        for i in range(65536):
            pcm_bytes = i.to_bytes(2, byteorder="little", signed=False)
            table[i] = audioop.lin2ulaw(pcm_bytes, 2)[0]
        return table

    except ImportError:
        pass

    # Fallback: Try audioop-lts (available for Python >= 3.13)
    try:
        import audioop_lts

        for i in range(65536):
            pcm_bytes = i.to_bytes(2, byteorder="little", signed=False)
            table[i] = audioop_lts.lin2ulaw(pcm_bytes, 2)[0]
        return table

    except ImportError:
        pass

    # Final fallback: Pure Python implementation (ITU-T G.711)
    _MULAW_BIAS = 0x84  # 132
    _MULAW_CLIP = 0x1FDF  # 8159
    SEG_END = [0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF]

    for i in range(65536):
        # Convert unsigned index to signed 16-bit value
        pcm_val = i if i < 32768 else i - 65536

        # Convert 16-bit to 14-bit
        pcm_val = pcm_val >> 2

        # Get sign and magnitude
        if pcm_val < 0:
            pcm_val = -pcm_val
            mask = 0x7F
        else:
            mask = 0xFF

        # Clip and add bias
        if pcm_val > _MULAW_CLIP:
            pcm_val = _MULAW_CLIP
        pcm_val += _MULAW_BIAS

        # Find segment
        segment = 8
        for seg_idx in range(8):
            if pcm_val <= SEG_END[seg_idx]:
                segment = seg_idx
                break

        # Encode
        if segment >= 8:
            table[i] = 0x7F ^ mask
        else:
            mantissa = (pcm_val >> (segment + 1)) & 0x0F
            table[i] = ((segment << 4) | mantissa) ^ mask

    return table


# Precompute encoding table for fast vectorized encoding
MULAW_ENCODE_TABLE = _build_mulaw_encode_table()


def mulaw_to_pcm(mulaw_bytes: bytes) -> PcmData:
    """
    Convert mulaw audio bytes to PcmData using lookup table.

    Args:
        mulaw_bytes: Raw mulaw encoded audio bytes from Twilio.

    Returns:
        PcmData at 8kHz mono.
    """
    mulaw_samples = np.frombuffer(mulaw_bytes, dtype=np.uint8)
    samples = MULAW_DECODE_TABLE[mulaw_samples]

    return PcmData(
        samples=samples,
        sample_rate=TWILIO_SAMPLE_RATE,
        channels=1,
        format=AudioFormat.S16,
    )


def pcm_to_mulaw(pcm: PcmData) -> bytes:
    """
    Convert PcmData to mulaw bytes for Twilio using lookup table.

    Args:
        pcm: PCM audio data (will be resampled to 8kHz if needed).

    Returns:
        Mulaw encoded bytes suitable for Twilio.
    """
    # Resample to 8kHz mono if needed
    if pcm.sample_rate != TWILIO_SAMPLE_RATE or pcm.channels != 1:
        pcm = pcm.resample(target_sample_rate=TWILIO_SAMPLE_RATE, target_channels=1)

    # Convert signed int16 to unsigned index for table lookup
    # int16 range [-32768, 32767] maps to uint16 range [0, 65535]
    samples = pcm.samples.astype(np.int16).view(np.uint16)

    # Use precomputed lookup table for fast encoding
    mulaw = MULAW_ENCODE_TABLE[samples]

    return mulaw.tobytes()
