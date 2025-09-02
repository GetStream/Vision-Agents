import logging
import torch
import numpy as np
import warnings
import time
from typing import Dict, Any, Optional
from getstream.plugins.common import VAD
from getstream.video.rtc.track_util import PcmData
from getstream.audio.utils import resample_audio
from getstream.plugins.common.events import VADAudioEvent
from getstream.plugins.common.event_utils import register_global_event
            

try:
    import onnxruntime as ort

    has_onnx = True
except ImportError:
    has_onnx = False


logger = logging.getLogger(__name__)


class SileroVAD(VAD):
    """
    Voice Activity Detection implementation using Silero VAD model.

    This class implements the VAD interface using the Silero VAD model,
    which is a high-performance speech detection model.

    Features:
    - Asymmetric thresholds for speech detection (activation_th and deactivation_th)
    - Automatic resampling to model's required rate (typically 16kHz)
    - GPU acceleration support with automatic fallback to CPU
    - Optional ONNX runtime support for potential performance improvements
    - Early partial events for real-time UI feedback during speech
    - Memory-efficient audio buffering using bytearrays
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        frame_size: int = None,
        activation_th: float = 0.4,
        deactivation_th: float = 0.2,
        speech_pad_ms: int = 300,
        min_speech_ms: int = 250,
        max_speech_ms: int = 30000,
        model_rate: int = 16000,
        window_samples: int = 512,
        device: str = "cpu",
        partial_frames: int = 10,
        use_onnx: bool = False,
    ):
        """
        Initialize the Silero VAD.

        Args:
            sample_rate: Audio sample rate in Hz expected for input
            frame_size: (Deprecated) Size of audio frames to process, use window_samples instead
            activation_th: Threshold for starting speech detection (0.0 to 1.0)
            deactivation_th: Threshold for ending speech detection (0.0 to 1.0) (defaults to 0.7*activation_th)
            speech_pad_ms: Number of milliseconds to pad before/after speech
            min_speech_ms: Minimum milliseconds of speech to emit
            max_speech_ms: Maximum milliseconds of speech before forced flush
            model_rate: Sample rate the model operates on (typically 16000 Hz)
            window_samples: Number of samples per window (must be 512 for 16kHz, 256 for 8kHz)
            device: Device to run the model on ("cpu", "cuda", "cuda:0", etc.)
            partial_frames: Number of frames to process before emitting a "partial" event
            use_onnx: Whether to use ONNX runtime for inference instead of PyTorch
        """
        # Issue deprecation warning for frame_size
        if frame_size is not None:
            warnings.warn(
                "The 'frame_size' parameter is deprecated and will be removed in a future version. "
                "Use 'window_samples' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            window_samples = frame_size

        super().__init__(
            sample_rate=sample_rate,
            frame_size=window_samples,  # Use window_samples for frame_size
            activation_th=activation_th,
            deactivation_th=deactivation_th,
            speech_pad_ms=speech_pad_ms,
            min_speech_ms=min_speech_ms,
            max_speech_ms=max_speech_ms,
            partial_frames=partial_frames,
        )

        # Model parameters
        self.model_rate = model_rate
        self.window_samples = window_samples
        self.device_name = device
        self.use_onnx = use_onnx and has_onnx

        # Verify window size is correct for the Silero model
        if self.model_rate == 16000 and self.window_samples != 512:
            logger.warning(
                f"Adjusting window_samples from {self.window_samples} to 512, "
                "which is required by Silero VAD at 16kHz"
            )
            self.window_samples = 512
        elif self.model_rate == 8000 and self.window_samples != 256:
            logger.warning(
                f"Adjusting window_samples from {self.window_samples} to 256, "
                "which is required by Silero VAD at 8kHz"
            )
            self.window_samples = 256

        # Buffer for raw input samples (before resampling)
        self._raw_buffer = np.array([], dtype=np.float32)

        # Buffer for resampled samples at model rate
        self._resampled = np.array([], dtype=np.float32)

        # ONNX session and model
        self.onnx_session = None
        self.model = None

        # Load the appropriate model
        self._load_model()

    def _load_model(self) -> None:
        """Load the Silero VAD model using torch hub or ONNX runtime."""
        try:
            if self.use_onnx:
                self._load_onnx_model()
            else:
                self._load_torch_model()
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise

    def _load_torch_model(self) -> None:
        """Load the PyTorch version of the Silero VAD model."""
        logger.info("Loading Silero VAD PyTorch model from torch hub")

        # Use torch.hub to load the model and utils
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )

        # Set model to evaluation mode
        self.model.eval()

        # Try to use the specified device, fall back to CPU if not available
        try:
            self.device = torch.device(self.device_name)
            # Test if CUDA is actually available when requested
            if self.device.type == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = torch.device("cpu")
                self.device_name = "cpu"
            self.model.to(self.device)
            logger.info(f"Using device: {self.device}")
        except Exception as e:
            logger.warning(
                f"Failed to use device {self.device_name}: {e}, falling back to CPU"
            )
            self.device = torch.device("cpu")
            self.device_name = "cpu"
            self.model.to(self.device)

        # Reset states
        self.reset_states()
        logger.info("Silero VAD PyTorch model loaded successfully")

    def _load_onnx_model(self) -> None:
        """Load the ONNX version of the Silero VAD model."""
        if not has_onnx:
            logger.warning("ONNX Runtime not available, falling back to PyTorch model")
            self.use_onnx = False
            self._load_torch_model()
            return

        logger.info("Loading Silero VAD ONNX model")

        try:
            # First load the model with PyTorch to get access to the ONNX export functionality
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )

            # Try to use the specified device for ONNX
            providers = []
            if (
                self.device_name.startswith("cuda")
                and "CUDAExecutionProvider" in ort.get_available_providers()
            ):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                logger.info("Using CUDA for ONNX inference")
            else:
                if self.device_name.startswith("cuda"):
                    logger.warning(
                        "CUDA requested but not available for ONNX, falling back to CPU"
                    )
                providers = ["CPUExecutionProvider"]
                self.device_name = "cpu"

            # Create a session options object and set graph optimization level
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # Export model to ONNX format in memory and load with ONNX Runtime
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
                # Create dummy input for model
                dummy_input = torch.randn(1, self.window_samples)

                # Export the model
                torch.onnx.export(
                    model,
                    dummy_input,
                    tmp.name,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size", 1: "sequence"},
                        "output": {0: "batch_size"},
                    },
                    opset_version=12,
                )

                # Create ONNX session
                self.onnx_session = ort.InferenceSession(
                    tmp.name, sess_options=session_options, providers=providers
                )

                # Get input name
                self.onnx_input_name = self.onnx_session.get_inputs()[0].name

            logger.info("Silero VAD ONNX model loaded successfully")

        except Exception as e:
            logger.warning(
                f"Failed to load ONNX model: {e}, falling back to PyTorch model"
            )
            self.use_onnx = False
            self._load_torch_model()

    def reset_states(self) -> None:
        """Reset the model states."""
        # Clear buffers
        self._raw_buffer = np.array([], dtype=np.float32)
        self._resampled = np.array([], dtype=np.float32)

    async def is_speech(self, frame: PcmData) -> float:
        """
        Detect speech in an audio frame using the Silero VAD model.

        Args:
            frame: PcmData object containing audio samples

        Returns:
            Probability (0.0 to 1.0) that the frame contains speech
        """
        try:
            # Convert PCM data to float32 in range [-1.0, 1.0]
            audio_array = frame.samples.astype(np.float32) / 32768.0

            # Add current frame to raw buffer (in original sample rate)
            self._raw_buffer = np.append(self._raw_buffer, audio_array)

            # Resample the accumulated raw buffer to model rate if needed
            if frame.sample_rate != self.model_rate:
                resampled_new = resample_audio(
                    self._raw_buffer, frame.sample_rate, self.model_rate
                )
                # Reset raw buffer after resampling
                self._raw_buffer = np.array([], dtype=np.float32)
            else:
                resampled_new = self._raw_buffer
                self._raw_buffer = np.array([], dtype=np.float32)

            # Add newly resampled data to existing resampled buffer
            self._resampled = np.append(self._resampled, resampled_new)

            # If we don't have enough samples for a full window, return 0
            if len(self._resampled) < self.window_samples:
                return 0.0

            # Process full windows of audio
            speech_probs = []

            # Process each complete window (512 samples @ 16kHz or 256 @ 8kHz)
            while len(self._resampled) >= self.window_samples:
                # Extract a window of samples
                window = self._resampled[: self.window_samples]
                self._resampled = self._resampled[self.window_samples :]

                # Measure inference time for RTF calculation
                start_time = time.time()

                try:
                    if self.use_onnx and self.onnx_session is not None:
                        # Convert to the format expected by ONNX (batch_size, sequence_length)
                        onnx_input = window.reshape(1, -1).astype(np.float32)

                        # Run ONNX inference
                        ort_inputs = {self.onnx_input_name: onnx_input}
                        ort_outputs = self.onnx_session.run(None, ort_inputs)

                        # Extract the speech probability
                        speech_prob = float(ort_outputs[0][0])
                    else:
                        # Convert numpy array to PyTorch tensor
                        tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)

                        # Get model predictions using PyTorch
                        with torch.no_grad():
                            speech_prob = self.model(tensor, self.model_rate)
                            speech_prob = float(speech_prob.item())

                    # Calculate real-time factor (RTF)
                    end_time = time.time()
                    inference_time = end_time - start_time
                    # Time taken to process window_samples at model_rate
                    audio_duration = self.window_samples / self.model_rate
                    rtf = inference_time / audio_duration

                    # Log speech probability and RTF at DEBUG level
                    logger.debug(
                        "Speech detection window processed",
                        extra={"p": speech_prob, "rtf": rtf},
                    )

                    speech_probs.append(speech_prob)

                except Exception as e:
                    logger.warning(f"Error during inference: {e}")
                    # If there was an error, continue with the next window
                    continue

            # Return highest probability if we have any valid predictions
            return max(speech_probs) if speech_probs else 0.0

        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            # On error, return low probability
            return 0.0

    async def _flush_speech_buffer(self, user: Optional[Dict[str, Any]] = None) -> None:
        """
        Flush the accumulated speech buffer if it meets minimum length requirements.

        Args:
            user: User metadata to include with emitted audio events
        """
        # Calculate min speech frames based on ms
        min_speech_frames = int(
            self.min_speech_ms * self.sample_rate / 1000 / self.frame_size
        )

        # Convert bytearray to numpy array
        speech_data = np.frombuffer(self.speech_buffer, dtype=np.int16).copy()

        if len(speech_data) >= min_speech_frames * self.frame_size:
            # Log turn emission at DEBUG level with duration and samples
            duration_ms = len(speech_data) / self.sample_rate * 1000
            logger.debug(
                "Turn emitted",
                extra={"duration_ms": duration_ms, "samples": len(speech_data)},
            )

            audio_event = VADAudioEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                audio_data=speech_data.tobytes(),
                sample_rate=self.sample_rate,
                audio_format="s16",
                channels=1,
                duration_ms=duration_ms,
                speech_probability=0.8,  # Default value, could be enhanced
                frame_count=len(speech_data) // self.frame_size,
                user_metadata=user
            )
            register_global_event(audio_event)
            self.emit("audio", audio_event)  # Structured event

        # Reset state variables
        self.speech_buffer = bytearray()
        self.silence_counter = 0
        self.is_speech_active = False
        self.total_speech_frames = 0
        self.partial_counter = 0

    async def reset(self) -> None:
        """Reset the VAD state."""
        await super().reset()
        self.reset_states()

    async def flush(self, user=None) -> None:
        """
        Flush accumulated speech buffer and emit any pending audio events.

        Args:
            user: User metadata to include with emitted audio events
        """
        await super().flush(user)
        # Reset buffer after flushing
        self.reset_states()

    async def close(self) -> None:
        """Release resources used by the model."""
        self.model = None
        if self.onnx_session is not None:
            self.onnx_session = None
        self.reset_states()
