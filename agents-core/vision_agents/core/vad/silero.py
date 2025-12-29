import asyncio
import os
import time
from typing_extensions import Self

import numpy as np
import onnxruntime as ort
from getstream.video.rtc import PcmData
from vision_agents.core.utils.utils import ensure_model

__all__ = ("SileroVADSession", "SileroVADSessionPool", "SILERO_CHUNK")

SILERO_CHUNK = 512
SILERO_ONNX_FILENAME = "silero_vad.onnx"
SILERO_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


class SileroVADSessionPool:
    """
    Initialize Silero VAD pool.
    It can be used to spawn stateful `SileroVAD` re-using the same inference session object.

    Args:
        model_path: Path to the ONNX model file
    """

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        self._model = ort.InferenceSession(model_path, sess_options=opts)

    @classmethod
    async def load(cls, model_dir: str) -> Self:
        """
        Initialize Silero VAD pool asynchronously without blocking event loop.

        Args:
            model_dir: path to the model directory.

        Returns:
            an instance of SileroVADSessionPool
        """
        path = os.path.join(model_dir, SILERO_ONNX_FILENAME)
        await ensure_model(path, SILERO_ONNX_URL)
        # Initialize VAD in thread pool to avoid blocking event loop
        pool = await asyncio.to_thread(  # type: ignore[func-returns-value]
            lambda: cls(  # type: ignore[arg-type]
                path
            )
        )
        return pool

    def session(self, reset_interval_seconds: float = 5.0) -> "SileroVADSession":
        """
        Create a new Silero VAD session.
        Each session has its own state but re-uses the same onnx inference session.

        Args:
            reset_interval_seconds: Reset internal state every N seconds to prevent drift

        Returns:
            an instace of `SileroVADSession`
        """
        return SileroVADSession(
            model=self._model, reset_interval_seconds=reset_interval_seconds
        )


class SileroVADSession:
    """
    Minimal Silero VAD ONNX wrapper
    """

    def __init__(self, model: ort.InferenceSession, reset_interval_seconds: float):
        """
        Initialize Silero VAD session.
        Each session has its own state but re-uses the same onnx inference session.

        Args:
            reset_interval_seconds: Reset internal state every N seconds to prevent drift
        """

        self._model = model
        self._context_size = 64  # Silero uses 64-sample context at 16 kHz
        self._reset_interval_seconds = reset_interval_seconds
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
        self._context = np.zeros((1, self._context_size), dtype=np.float32)
        self._last_reset_time = time.time()

    def _maybe_reset(self):
        if (time.time() - self._last_reset_time) >= self._reset_interval_seconds:
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
        outputs = self._model.run(None, ort_inputs)
        out, self._state = outputs

        # Update context (keep last 64 samples)
        self._context = x[:, -self._context_size :]
        self._maybe_reset()

        # out shape is (1, 1) -> return scalar
        prediction = float(out[0][0])
        return prediction
