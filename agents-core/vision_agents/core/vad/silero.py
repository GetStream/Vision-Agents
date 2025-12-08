import os
import time

import numpy as np
import asyncio

from getstream.video.rtc import PcmData
from vision_agents.core.utils.utils import ensure_model

SILERO_CHUNK = 512
SILERO_ONNX_FILENAME = "silero_vad.onnx"
SILERO_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


class SileroVAD:
    """
    Minimal Silero VAD ONNX wrapper
    """

    def __init__(self, model_path: str, reset_interval_seconds: float = 5.0):
        """
        Initialize Silero VAD.

        Args:
            model_path: Path to the ONNX model file
            reset_interval_seconds: Reset internal state every N seconds to prevent drift
        """
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self.context_size = 64  # Silero uses 64-sample context at 16 kHz
        self.reset_interval_seconds = reset_interval_seconds
        self._state: np.ndarray = np.zeros((2, 1, 128), dtype=np.float32)  # (2, B, 128)
        self._context: np.ndarray = np.zeros((1, 64), dtype=np.float32)
        self._init_states()

    def predict_speech(self, pcm: PcmData):
        # convert from pcm to the right format for silero

        chunks = pcm.resample(16000, 1).to_float32().chunks(SILERO_CHUNK, pad_last=True)
        scores = [self._predict_speech(c.samples) for c in chunks]
        return max(scores)

    def _init_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)  # (2, B, 128)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)
        self._last_reset_time = time.time()

    def _maybe_reset(self):
        if (time.time() - self._last_reset_time) >= self.reset_interval_seconds:
            self._init_states()

    def _predict_speech(self, chunk_f32: np.ndarray) -> float:
        """
        Compute speech probability for one chunk of length 512 (float32, mono).
        Returns a scalar float.
        """
        # Ensure shape (1, 512) and concat context
        x = np.reshape(chunk_f32, (1, -1))
        if x.shape[1] != SILERO_CHUNK:
            # Raise on incorrect usage
            raise ValueError(
                f"incorrect usage for predict speech. only send audio data in chunks of 512. got {x.shape[1]}"
            )
        x = np.concatenate((self._context, x), axis=1)

        # Run ONNX
        ort_inputs = {
            "input": x.astype(np.float32),
            "state": self._state,
            "sr": np.array(16000, dtype=np.int64),
        }
        outputs = self.session.run(None, ort_inputs)
        out, self._state = outputs

        # Update context (keep last 64 samples)
        self._context = x[:, -self.context_size :]
        self._maybe_reset()

        # out shape is (1, 1) -> return scalar
        prediction = float(out[0][0])
        return prediction


async def prepare_silero_vad(model_dir: str) -> SileroVAD:
    path = os.path.join(model_dir, SILERO_ONNX_FILENAME)
    await ensure_model(path, SILERO_ONNX_URL)
    # Initialize VAD in thread pool to avoid blocking event loop
    vad = await asyncio.to_thread(  # type: ignore[func-returns-value]
        lambda: SileroVAD(  # type: ignore[arg-type]
            path, reset_interval_seconds=5.0
        )
    )
    return vad
