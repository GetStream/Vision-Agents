"""Streaming Parakeet TDT ASR over a Baseten WebSocket.

Wire protocol:
    1. Client sends a JSON text frame: {"sample_rate": 16000, "encoding": "linear16"}.
       The server replies {"type": "ready"}.
    2. Client streams binary frames of little-endian mono PCM16.
    3. Server emits {"type": "start_of_turn"} when speech begins,
       {"type": "partial", "text": ...} as the utterance grows and
       {"type": "final", "text": ...} once trailing silence ends the utterance.
    4. Client may send {"type": "end_audio"} to flush; the server answers with a
       final (if any audio is pending) followed by {"status": "finished"}.

Parakeet TDT is a chunk model, not a streaming one, so each partial re-transcribes
the current utterance buffer. Partials are skipped while one is already running,
which keeps long utterances from queueing up behind their own inference.
"""

import asyncio
import json
import logging
import os
import threading
import time

import fastapi
import nemo.collections.asr as nemo_asr
import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = "/app/model_cache/parakeet-tdt-0.6b-v3"
MODEL_NEMO_FILE = "parakeet-tdt-0.6b-v3.nemo"
SAMPLE_RATE = 16_000
INT16_FULL_SCALE = 32768.0


class ProtocolError(Exception):
    """The client sent something the server cannot honour."""


def _detect_dtype() -> torch.dtype:
    """Returns bf16 on Ampere+, fp16 on Turing, fp32 without CUDA."""
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def _load_nemo_model(model_path: str, dtype: torch.dtype, enable_cuda_graphs: bool):
    """Restores the NeMo model, tunes it for short-window inference and warms it up.

    Args:
        model_path: Path to the cached .nemo file.
        dtype: Weight dtype to cast to.
        enable_cuda_graphs: Whether to keep CUDA graphs on for the TDT decoder.

    Returns:
        The loaded, warmed-up ASR model.
    """
    model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)

    if not enable_cuda_graphs:
        model.decoding.decoding.decoding_computer.disable_cuda_graphs()

    # Local windowed attention keeps memory flat as the utterance buffer grows.
    model.change_attention_model("rel_pos_local_attn", [256, 256])
    model.change_subsampling_conv_chunking_factor(1)
    model.to(dtype)
    model.eval()

    dummy = np.random.randn(SAMPLE_RATE * 5).astype(np.float32)
    with torch.inference_mode():
        model.transcribe([dummy], timestamps=False)

    return model


def _rms(samples: np.ndarray) -> float:
    """Returns the RMS of int16 samples on the int16 amplitude scale."""
    if samples.size == 0:
        return 0.0
    as_float = samples.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(as_float))))


class _Utterance:
    """Accumulates PCM16 for the current utterance and tracks trailing silence."""

    def __init__(self, silence_rms: float):
        self._silence_rms = silence_rms
        self._chunks: list[np.ndarray] = []
        self.total_samples = 0
        self.samples_since_partial = 0
        self.trailing_silence_samples = 0
        self.has_speech = False

    def append(self, samples: np.ndarray) -> None:
        self._chunks.append(samples)
        self.total_samples += samples.size
        self.samples_since_partial += samples.size

        if _rms(samples) < self._silence_rms:
            self.trailing_silence_samples += samples.size
        else:
            self.trailing_silence_samples = 0
            self.has_speech = True

    def audio(self) -> np.ndarray:
        """Returns the utterance as float32 in [-1, 1] for the model."""
        joined = np.concatenate(self._chunks) if self._chunks else np.empty(0, np.int16)
        return joined.astype(np.float32) / INT16_FULL_SCALE

    def duration_ms(self) -> float:
        return self.total_samples / SAMPLE_RATE * 1000.0

    def reset(self) -> None:
        self._chunks.clear()
        self.total_samples = 0
        self.samples_since_partial = 0
        self.trailing_silence_samples = 0
        self.has_speech = False


class Model:
    def __init__(self, lazy_data_resolver, **kwargs) -> None:
        self._lazy_data_resolver = lazy_data_resolver
        self._cuda_graphs = os.getenv("CUDA_GRAPHS", "true").lower() == "true"
        self._window_samples = int(
            float(os.getenv("WINDOW_MS", "480")) / 1000 * SAMPLE_RATE
        )
        self._silence_samples = int(
            float(os.getenv("SILENCE_MS", "700")) / 1000 * SAMPLE_RATE
        )
        self._max_samples = int(float(os.getenv("MAX_UTTERANCE_S", "30")) * SAMPLE_RATE)
        self._silence_rms = float(os.getenv("SILENCE_RMS", "300"))

        self.model = None
        # The model is shared by every connection on this replica.
        self._transcribe_lock = threading.Lock()

    def load(self) -> None:
        self._lazy_data_resolver.block_until_download_complete()
        model_path = os.path.join(MODEL_CACHE_DIR, MODEL_NEMO_FILE)

        started = time.time()
        self.model = _load_nemo_model(model_path, _detect_dtype(), self._cuda_graphs)
        logger.info("model loaded and warmed up in %.1fs", time.time() - started)

    async def websocket(self, websocket: fastapi.WebSocket) -> None:
        try:
            await self._negotiate(websocket)
        except ProtocolError as exc:
            await websocket.send_text(json.dumps({"type": "error", "error": str(exc)}))
            await websocket.close(code=1008)
            return

        utterance = _Utterance(self._silence_rms)
        partial_task: asyncio.Task | None = None

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return

                payload = message.get("bytes")
                if payload is None:
                    if await self._handle_control(
                        websocket, message.get("text"), utterance
                    ):
                        return
                    continue

                was_speaking = utterance.has_speech
                utterance.append(np.frombuffer(payload, dtype=np.int16))
                if utterance.has_speech and not was_speaking:
                    await websocket.send_text(json.dumps({"type": "start_of_turn"}))

                if utterance.total_samples >= self._max_samples:
                    await self._emit_final(websocket, utterance)
                    continue

                if (
                    utterance.has_speech
                    and utterance.trailing_silence_samples >= self._silence_samples
                ):
                    await self._emit_final(websocket, utterance)
                    continue

                if utterance.samples_since_partial >= self._window_samples:
                    utterance.samples_since_partial = 0
                    if utterance.has_speech and (
                        partial_task is None or partial_task.done()
                    ):
                        partial_task = asyncio.create_task(
                            self._emit_partial(websocket, utterance.audio())
                        )
        except fastapi.WebSocketDisconnect:
            pass
        finally:
            if partial_task is not None:
                partial_task.cancel()

    async def _negotiate(self, websocket: fastapi.WebSocket) -> None:
        """Reads and validates the opening metadata frame."""
        raw = await websocket.receive_text()
        try:
            config = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"first frame must be JSON metadata: {exc}") from exc

        sample_rate = config.get("sample_rate", SAMPLE_RATE)
        if sample_rate != SAMPLE_RATE:
            raise ProtocolError(f"sample_rate must be {SAMPLE_RATE}, got {sample_rate}")

        encoding = config.get("encoding", "linear16")
        if encoding != "linear16":
            raise ProtocolError(f"encoding must be linear16, got {encoding}")

        await websocket.send_text(json.dumps({"type": "ready"}))

    async def _handle_control(
        self, websocket: fastapi.WebSocket, text: str | None, utterance: _Utterance
    ) -> bool:
        """Handles a text control frame. Returns True when the session should end."""
        if text is None:
            return False

        try:
            control = json.loads(text)
        except json.JSONDecodeError:
            return False

        if control.get("type") != "end_audio":
            return False

        if utterance.has_speech:
            await self._emit_final(websocket, utterance)
        await websocket.send_text(json.dumps({"status": "finished"}))
        return True

    async def _emit_partial(
        self, websocket: fastapi.WebSocket, audio: np.ndarray
    ) -> None:
        started = time.monotonic()
        text = await asyncio.to_thread(self._transcribe, audio)
        if not text:
            return
        await websocket.send_text(
            json.dumps(
                {
                    "type": "partial",
                    "text": text,
                    "audio_duration_ms": audio.size / SAMPLE_RATE * 1000.0,
                    "processing_time_ms": (time.monotonic() - started) * 1000.0,
                }
            )
        )

    async def _emit_final(
        self, websocket: fastapi.WebSocket, utterance: _Utterance
    ) -> None:
        audio = utterance.audio()
        duration_ms = utterance.duration_ms()
        utterance.reset()

        started = time.monotonic()
        text = await asyncio.to_thread(self._transcribe, audio)
        await websocket.send_text(
            json.dumps(
                {
                    "type": "final",
                    "text": text,
                    "audio_duration_ms": duration_ms,
                    "processing_time_ms": (time.monotonic() - started) * 1000.0,
                }
            )
        )

    def _transcribe(self, audio: np.ndarray) -> str:
        with self._transcribe_lock, torch.inference_mode():
            hypotheses = self.model.transcribe([audio], timestamps=False, verbose=False)

        if not hypotheses:
            return ""
        first = hypotheses[0]
        return first if isinstance(first, str) else first.text
