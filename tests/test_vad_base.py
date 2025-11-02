import asyncio
import numpy as np
import pytest

from vision_agents.core.vad.vad import VAD as BaseVAD
from vision_agents.core.vad.events import VADAudioEvent, VADPartialEvent
from getstream.video.rtc.track_util import PcmData


class EnergyVAD(BaseVAD):
    """Simple energy-based VAD for testing base class behavior.

    Returns high probability when RMS energy exceeds a threshold.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_samples: int = 512,
        energy_th: float = 0.02,
        activation_th: float = 0.5,
        deactivation_th: float = 0.3,
        speech_pad_ms: int = 300,
        min_speech_ms: int = 200,
        max_speech_ms: int = 30000,
        partial_frames: int = 10,
    ):
        super().__init__(
            sample_rate=sample_rate,
            window_samples=window_samples,
            channels=1,
            activation_th=activation_th,
            deactivation_th=deactivation_th,
            speech_pad_ms=speech_pad_ms,
            min_speech_ms=min_speech_ms,
            max_speech_ms=max_speech_ms,
            partial_frames=partial_frames,
            provider_name="EnergyVAD",
        )
        self.energy_th = float(energy_th)

    async def is_speech(self, window: PcmData) -> float:
        x = window.samples
        if isinstance(x, bytes):
            x = np.frombuffer(x, dtype=np.int16)
        if x.dtype != np.int16:
            x = x.astype(np.int16)
        # Normalize to [-1, 1]
        xf = x.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(xf)) + 1e-12))
        # Map to probability-like value
        return 1.0 if rms >= self.energy_th else 0.0


@pytest.mark.asyncio
async def test_base_vad_silence_no_turns():
    vad = EnergyVAD(
        sample_rate=16000,
        window_samples=512,
        energy_th=0.02,
        activation_th=0.5,
        deactivation_th=0.3,
        speech_pad_ms=300,
        min_speech_ms=200,
        max_speech_ms=30000,
        partial_frames=5,
    )

    # 5 seconds of silence @16k
    silence = np.zeros(16000 * 5, dtype=np.int16)
    pcm = PcmData(samples=silence, sample_rate=16000, format="s16")

    audio_events = []
    partial_events = []

    @vad.events.subscribe
    async def on_audio(ev: VADAudioEvent):
        audio_events.append(ev)

    @vad.events.subscribe
    async def on_partial(ev: VADPartialEvent):
        partial_events.append(ev)

    await vad.process_audio(pcm)
    await vad.flush()
    await asyncio.sleep(0.05)

    assert len(audio_events) == 0, "No audio events expected on silence"
    assert len(partial_events) == 0, "No partial events expected on silence"


def _decode_mia_to_pcm16k() -> PcmData:
    import av
    import os
    from conftest import get_assets_dir

    audio_file_path = os.path.join(get_assets_dir(), "mia.mp3")
    container = av.open(audio_file_path)
    audio_stream = container.streams.audio[0]
    target_rate = 16000
    resampler = None
    if audio_stream.sample_rate != target_rate:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)

    samples_list = []
    for frame in container.decode(audio_stream):
        if resampler:
            frame = resampler.resample(frame)[0]
        arr = frame.to_ndarray()
        if arr.ndim > 1:
            arr = arr.mean(axis=0).astype(np.int16)
        samples_list.append(arr)
    container.close()
    if not samples_list:
        return PcmData(samples=np.zeros(0, dtype=np.int16), sample_rate=target_rate, format="s16")
    samples = np.concatenate(samples_list).astype(np.int16)
    return PcmData(samples=samples, sample_rate=target_rate, format="s16")


@pytest.mark.asyncio
async def test_base_vad_mia_detects_speech_segments():
    vad = EnergyVAD(
        sample_rate=16000,
        window_samples=512,
        energy_th=0.01,
        activation_th=0.5,
        deactivation_th=0.3,
        speech_pad_ms=200,
        min_speech_ms=100,
        max_speech_ms=30000,
        partial_frames=5,
    )

    pcm = _decode_mia_to_pcm16k()

    audio_events = []
    partial_events = []

    @vad.events.subscribe
    async def on_audio(ev: VADAudioEvent):
        audio_events.append(ev)

    @vad.events.subscribe
    async def on_partial(ev: VADPartialEvent):
        partial_events.append(ev)

    await vad.process_audio(pcm)
    await vad.flush()
    await asyncio.sleep(0.05)

    assert len(audio_events) > 0, "Expected at least one speech segment from mia"
    # Sanity: total bytes > 0 and duration present
    total_bytes = sum(len(ev.audio_data or b"") for ev in audio_events)
    assert total_bytes > 0
    assert any((ev.duration_ms or 0) > 0 for ev in audio_events)


@pytest.mark.asyncio
async def test_base_vad_white_noise_triggers_speech():
    vad = EnergyVAD(
        sample_rate=16000,
        window_samples=512,
        energy_th=0.02,
        activation_th=0.5,
        deactivation_th=0.3,
        speech_pad_ms=100,
        min_speech_ms=50,
        max_speech_ms=30000,
        partial_frames=5,
    )

    # Generate white noise at modest amplitude
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(16000) * 0.1).astype(np.float32)
    noise_i16 = np.clip(noise * 32768.0, -32768, 32767).astype(np.int16)
    pcm = PcmData(samples=noise_i16, sample_rate=16000, format="s16")

    audio_events = []

    @vad.events.subscribe
    async def on_audio(ev: VADAudioEvent):
        audio_events.append(ev)

    await vad.process_audio(pcm)
    await vad.flush()
    await asyncio.sleep(0.05)

    assert len(audio_events) > 0, "Expected at least one speech event on white noise"

