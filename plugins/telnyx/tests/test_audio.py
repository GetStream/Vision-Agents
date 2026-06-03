"""Tests for Telnyx audio conversion."""

import numpy as np
from getstream.video.rtc.track_util import AudioFormat, PcmData

from vision_agents.plugins.telnyx.audio import (
    TELNYX_DEFAULT_SAMPLE_RATE,
    TELNYX_L16_SAMPLE_RATE,
    l16_to_pcm,
    pcma_to_pcm,
    pcm_to_l16,
    pcm_to_pcma,
    pcm_to_pcmu,
    pcm_to_telnyx_payload,
    pcmu_to_pcm,
    telnyx_payload_to_pcm,
)


def test_pcmu_to_pcm():
    pcm = pcmu_to_pcm(bytes([0xFF, 0x7F, 0x00, 0x80]))

    assert pcm.sample_rate == TELNYX_DEFAULT_SAMPLE_RATE
    assert pcm.channels == 1
    assert len(pcm.samples) == 4


def test_pcm_to_pcmu():
    pcm = PcmData(
        samples=np.array([0, 1000, -1000, 16000], dtype=np.int16),
        sample_rate=TELNYX_DEFAULT_SAMPLE_RATE,
        channels=1,
        format=AudioFormat.S16,
    )

    payload = pcm_to_pcmu(pcm)

    assert isinstance(payload, bytes)
    assert len(payload) == 4


def test_pcma_to_pcm():
    pcm = pcma_to_pcm(bytes([0xD5, 0x55, 0x00, 0x80]))

    assert pcm.sample_rate == TELNYX_DEFAULT_SAMPLE_RATE
    assert pcm.channels == 1
    assert len(pcm.samples) == 4


def test_pcm_to_pcma():
    pcm = PcmData(
        samples=np.array([0, 1000, -1000, 16000], dtype=np.int16),
        sample_rate=TELNYX_DEFAULT_SAMPLE_RATE,
        channels=1,
        format=AudioFormat.S16,
    )

    payload = pcm_to_pcma(pcm)

    assert isinstance(payload, bytes)
    assert len(payload) == 4


def test_l16_to_pcm():
    pcm = l16_to_pcm(b"\x00\x00\x7f\xff\x80\x00")

    assert pcm.sample_rate == TELNYX_L16_SAMPLE_RATE
    assert pcm.channels == 1
    assert pcm.samples.tolist() == [0, 32767, -32768]


def test_pcm_to_l16():
    pcm = PcmData(
        samples=np.array([0, 32767, -32768], dtype=np.int16),
        sample_rate=TELNYX_L16_SAMPLE_RATE,
        channels=1,
        format=AudioFormat.S16,
    )

    assert pcm_to_l16(pcm) == b"\x00\x00\x7f\xff\x80\x00"


def test_payload_helpers():
    pcm = telnyx_payload_to_pcm(bytes([0xFF]), "PCMU")
    payload = pcm_to_telnyx_payload(pcm, "PCMU")

    assert isinstance(payload, bytes)
    assert len(payload) == 1
