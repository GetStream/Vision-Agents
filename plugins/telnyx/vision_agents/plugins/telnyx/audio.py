"""Audio conversion utilities for Telnyx media streaming."""

import numpy as np
from getstream.video.rtc.track_util import AudioFormat, PcmData

TELNYX_DEFAULT_SAMPLE_RATE = 8000
TELNYX_L16_SAMPLE_RATE = 16000

_ULAW_BIAS = 0x84
_ULAW_CLIP = 32635
_SEG_END = (0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF)


def pcmu_to_pcm(payload: bytes) -> PcmData:
    """
    Convert Telnyx PCMU RTP payload bytes to PcmData.

    Args:
        payload: Base64-decoded PCMU RTP payload bytes from Telnyx.

    Returns:
        PcmData at 8 kHz mono.
    """
    samples = np.array([_decode_ulaw_byte(value) for value in payload], dtype=np.int16)
    return PcmData(
        samples=samples,
        sample_rate=TELNYX_DEFAULT_SAMPLE_RATE,
        channels=1,
        format=AudioFormat.S16,
    )


def pcma_to_pcm(payload: bytes) -> PcmData:
    """
    Convert Telnyx PCMA RTP payload bytes to PcmData.

    Args:
        payload: Base64-decoded PCMA RTP payload bytes from Telnyx.

    Returns:
        PcmData at 8 kHz mono.
    """
    samples = np.array([_decode_alaw_byte(value) for value in payload], dtype=np.int16)
    return PcmData(
        samples=samples,
        sample_rate=TELNYX_DEFAULT_SAMPLE_RATE,
        channels=1,
        format=AudioFormat.S16,
    )


def l16_to_pcm(payload: bytes, sample_rate: int = TELNYX_L16_SAMPLE_RATE) -> PcmData:
    """
    Convert Telnyx L16 RTP payload bytes to PcmData.

    RTP L16 is network byte order.
    """
    if len(payload) % 2:
        payload = payload[:-1]
    samples = np.frombuffer(payload, dtype=">i2").astype(np.int16)
    return PcmData(
        samples=samples,
        sample_rate=sample_rate,
        channels=1,
        format=AudioFormat.S16,
    )


def pcm_to_pcmu(pcm: PcmData) -> bytes:
    """
    Convert PcmData to PCMU bytes for Telnyx bidirectional RTP streaming.
    """
    pcm = _as_mono_s16(pcm, TELNYX_DEFAULT_SAMPLE_RATE)
    return bytes(_encode_ulaw_sample(sample) for sample in pcm.samples)


def pcm_to_pcma(pcm: PcmData) -> bytes:
    """
    Convert PcmData to PCMA bytes for Telnyx bidirectional RTP streaming.
    """
    pcm = _as_mono_s16(pcm, TELNYX_DEFAULT_SAMPLE_RATE)
    return bytes(_encode_alaw_sample(sample) for sample in pcm.samples)


def pcm_to_l16(pcm: PcmData, sample_rate: int = TELNYX_L16_SAMPLE_RATE) -> bytes:
    """
    Convert PcmData to big-endian L16 bytes for Telnyx bidirectional RTP.
    """
    pcm = _as_mono_s16(pcm, sample_rate)
    return pcm.samples.astype(">i2").tobytes()


def telnyx_payload_to_pcm(
    payload: bytes,
    encoding: str,
    sample_rate: int = TELNYX_DEFAULT_SAMPLE_RATE,
) -> PcmData:
    """
    Decode a Telnyx RTP payload using the stream media format.
    """
    encoding = encoding.upper()
    if encoding == "PCMU":
        return pcmu_to_pcm(payload)
    if encoding == "PCMA":
        return pcma_to_pcm(payload)
    if encoding == "L16":
        return l16_to_pcm(payload, sample_rate=sample_rate)
    raise ValueError(f"Unsupported Telnyx media encoding: {encoding}")


def pcm_to_telnyx_payload(
    pcm: PcmData,
    encoding: str,
    sample_rate: int = TELNYX_DEFAULT_SAMPLE_RATE,
) -> bytes:
    """
    Encode PcmData for a Telnyx bidirectional RTP media frame.
    """
    encoding = encoding.upper()
    if encoding == "PCMU":
        return pcm_to_pcmu(pcm)
    if encoding == "PCMA":
        return pcm_to_pcma(pcm)
    if encoding == "L16":
        return pcm_to_l16(pcm, sample_rate=sample_rate)
    raise ValueError(f"Unsupported Telnyx media encoding: {encoding}")


def _as_mono_s16(pcm: PcmData, sample_rate: int) -> PcmData:
    if pcm.sample_rate != sample_rate or pcm.channels != 1:
        pcm = pcm.resample(target_sample_rate=sample_rate, target_channels=1)
    if pcm.samples.dtype != np.int16:
        pcm = PcmData(
            samples=pcm.samples.astype(np.int16),
            sample_rate=pcm.sample_rate,
            channels=pcm.channels,
            format=AudioFormat.S16,
        )
    return pcm


def _decode_ulaw_byte(value: int) -> int:
    value = ~value & 0xFF
    sample = ((value & 0x0F) << 3) + _ULAW_BIAS
    sample <<= (value & 0x70) >> 4
    sample -= _ULAW_BIAS
    return -sample if value & 0x80 else sample


def _encode_ulaw_sample(sample: int) -> int:
    sample = int(sample)
    if sample < 0:
        sample = _ULAW_BIAS - sample
        mask = 0x7F
    else:
        sample += _ULAW_BIAS
        mask = 0xFF

    sample = min(sample, _ULAW_CLIP)
    segment = _search_segment(sample)
    if segment >= 8:
        return 0x7F ^ mask

    encoded = (segment << 4) | ((sample >> (segment + 3)) & 0x0F)
    return encoded ^ mask


def _decode_alaw_byte(value: int) -> int:
    value ^= 0x55
    sign = value & 0x80
    exponent = (value & 0x70) >> 4
    sample = (value & 0x0F) << 4
    if exponent == 0:
        sample += 8
    else:
        sample += 0x108
        sample <<= exponent - 1
    return sample if sign else -sample


def _encode_alaw_sample(sample: int) -> int:
    sample = int(sample)
    if sample >= 0:
        mask = 0xD5
    else:
        mask = 0x55
        sample = -sample - 8

    sample = max(0, min(sample, 32635))
    if sample < 256:
        encoded = sample >> 4
    else:
        segment = _search_segment(sample)
        encoded = (segment << 4) | ((sample >> (segment + 3)) & 0x0F)
    return encoded ^ mask


def _search_segment(sample: int) -> int:
    for index, end in enumerate(_SEG_END):
        if sample <= end:
            return index
    return 8
