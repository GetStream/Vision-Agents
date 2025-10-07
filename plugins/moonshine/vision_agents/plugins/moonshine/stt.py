import os
import logging
import time
from typing import Dict, Any, Optional, Tuple, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from vision_agents.core.edge.types import Participant
import numpy as np
import soundfile as sf

from vision_agents.core import stt
from getstream.video.rtc.track_util import PcmData
from getstream.audio.utils import resample_audio
from getstream.audio.pcm_utils import (
    pcm_to_numpy_array,
    validate_sample_rate_compatibility,
    log_audio_processing_info,
)

logger = logging.getLogger(__name__)

# Conditional import of moonshine_onnx
try:
    import moonshine_onnx as moonshine

    MOONSHINE_AVAILABLE = True
except ImportError:
    moonshine = None
    MOONSHINE_AVAILABLE = False

# Supported Moonshine models with canonical mapping
_SUPPORTED_MODELS = {
    "tiny": "moonshine/tiny",
    "moonshine/tiny": "moonshine/tiny",
    "base": "moonshine/base",
    "moonshine/base": "moonshine/base",
}


class STT(stt.STT):
    """
    Moonshine-based Speech-to-Text implementation.

    Moonshine is a family of speech-to-text models optimized for fast and accurate
    automatic speech recognition (ASR) on resource-constrained devices.

    This implementation operates in synchronous mode - it processes audio immediately
    and returns results to the base class, which then emits the appropriate events.

    Events:
        - transcript: Emitted when a complete transcript is available.
            Args: text (str), user_metadata (dict), metadata (dict)
        - partial_transcript: Emitted when a partial transcript is available.
            Args: text (str), user_metadata (dict), metadata (dict)
        - error: Emitted when an error occurs during transcription.
            Args: error (Exception)
    """

    def __init__(
        self,
        model_name: str = "moonshine/base",
        sample_rate: int = 16000,
        min_audio_length_ms: int = 100,
        target_dbfs: float = -26.0,
    ):
        """
        Initialize the Moonshine STT service.

        Args:
            model_name: Moonshine model to use ("moonshine/tiny" or "moonshine/base")
            sample_rate: Sample rate of the audio in Hz (default: 16000, Moonshine's native rate)
            min_audio_length_ms: Minimum audio length required for transcription
            target_dbfs: Target RMS level in dBFS for audio normalization (default: -26.0, Moonshine's optimal level)
        """
        super().__init__(sample_rate=sample_rate)

        # Check if moonshine_onnx is available
        if not MOONSHINE_AVAILABLE:
            raise ImportError(
                "moonshine_onnx is not installed. "
                "Please install it from GitHub: pip install 'useful-moonshine-onnx@git+https://github.com/usefulsensors/moonshine.git#subdirectory=moonshine-onnx'"
            )

        # Validate and normalize model name
        if model_name not in _SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown Moonshine model '{model_name}'. "
                f"Allowed values: {', '.join(_SUPPORTED_MODELS.keys())}"
            )
        self.model_name = _SUPPORTED_MODELS[model_name]  # Use canonical value
        self.min_audio_length_ms = min_audio_length_ms
        self.target_dbfs = target_dbfs

        # Track current user context
        self._current_user: Optional[Dict[str, Any]] = None
        # Local explicit state flags for mypy visibility
        self._is_closed: bool = False

        logger.info(
            "Initialized Moonshine STT",
            extra={
                "model_name": model_name,
                "canonical_model": self.model_name,
                "sample_rate": sample_rate,
                "min_audio_length_ms": min_audio_length_ms,
                "target_dbfs": target_dbfs,
            },
        )

        # Log the exact checkpoint that will be loaded
        logger.info("Moonshine model selected", extra={"checkpoint": self.model_name})

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio data to float32 in range [-1, 1].

        Args:
            audio_data: Input audio data as int16

        Returns:
            Normalized audio data as float32
        """
        # Convert int16 to float32 and normalize to [-1, 1]
        return audio_data.astype(np.float32) / 32768.0

    def _rms_normalise(self, audio: np.ndarray) -> np.ndarray:
        """
        Boost or attenuate audio to match target RMS in dBFS.

        Args:
            audio: Input audio data as int16

        Returns:
            RMS-normalized audio data as int16
        """
        # Calculate current RMS in dB
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        if rms == 0:
            logger.debug("Silent audio detected, skipping RMS normalization")
            return audio  # silence

        current_dbfs = 20 * np.log10(rms / 32768.0)
        gain_db = self.target_dbfs - current_dbfs
        gain = 10 ** (gain_db / 20)

        logger.debug(
            "RMS normalization applied",
            extra={
                "current_dbfs": current_dbfs,
                "target_dbfs": self.target_dbfs,
                "gain_db": gain_db,
                "gain_linear": gain,
            },
        )

        # Apply gain and clip to prevent overflow
        boosted = np.clip(audio.astype(np.float32) * gain, -32768, 32767)
        return boosted.astype(np.int16)

    async def _transcribe_audio(
        self, audio_data: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        """
        Transcribe audio data using Moonshine in-memory.

        Args:
            audio_data: Audio data as float32 array

        Returns:
            Transcribed text and processing time, or None if transcription failed
        """
        try:
            # Check minimum audio length
            audio_duration_ms = (len(audio_data) / self.sample_rate) * 1000
            if audio_duration_ms < self.min_audio_length_ms:
                logger.debug(
                    "Audio too short for transcription",
                    extra={
                        "duration_ms": audio_duration_ms,
                        "min_duration_ms": self.min_audio_length_ms,
                    },
                )
                return None

            # Apply RMS normalization to boost audio to optimal level for Moonshine
            # First convert to int16 for RMS calculation, then back to float32 for model
            audio_int16 = (audio_data * 32768.0).astype(np.int16)
            audio_int16_normalized = self._rms_normalise(audio_int16)

            # Convert back to float32 for the model (expected input format)
            audio_float32 = audio_int16_normalized.astype(np.float32) / 32768.0

            logger.debug(
                "Transcribing audio in-memory",
                extra={
                    "sample_rate": self.sample_rate,
                    "audio_shape": audio_float32.shape,
                    "audio_duration_ms": audio_duration_ms,
                    "format": "float32 normalized",
                    "rms_normalized": True,
                    "target_dbfs": self.target_dbfs,
                },
            )

            # Try direct array transcription first (newer moonshine_onnx versions)
            start_time = time.time()
            try:
                # Moonshine ONNX expects float32 input
                result = moonshine.transcribe(audio_float32, self.model_name)
                processing_time = time.time() - start_time
            except (TypeError, AttributeError, Exception):
                # Fall back to temporary file approach
                import tempfile

                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp_file:
                    temp_path = temp_file.name

                try:
                    # Save to temporary file as int16 PCM
                    sf.write(
                        temp_path,
                        audio_int16_normalized,
                        self.sample_rate,
                        subtype="PCM_16",
                    )

                    # Transcribe using file path
                    result = moonshine.transcribe(temp_path, self.model_name)
                    processing_time = time.time() - start_time
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

            # Moonshine returns a list of strings
            if result and len(result) > 0:
                text = " ".join(result).strip()
                if text:
                    logger.debug(
                        "Transcription completed",
                        extra={
                            "text_length": len(text),
                            "processing_time_ms": processing_time * 1000,
                            "audio_duration_ms": audio_duration_ms,
                            "real_time_factor": (processing_time * 1000)
                            / audio_duration_ms,
                        },
                    )
                    return text, processing_time * 1000
                else:
                    logger.debug("Empty transcription result")
                    return None
            else:
                logger.debug("No transcription result")
                return None

        except Exception as e:
            logger.error("Error during transcription", exc_info=e)
            return None

    async def _process_audio_impl(
        self, pcm_data: PcmData, user_metadata: Optional[Union[Dict[str, Any], "Participant"]] = None
    ) -> Optional[List[Tuple[bool, str, Dict[str, Any]]]]:
        """
        Process audio data through Moonshine for transcription.

        Moonshine operates in synchronous mode - it processes audio immediately and
        returns results to the base class for event emission.

        Args:
            pcm_data: The PCM audio data to process.
            user_metadata: Additional metadata about the user or session.

        Returns:
            List of tuples (is_final, text, metadata) representing transcription results,
            or None if no results are available. Moonshine returns final results only
            since it doesn't support streaming transcription.
        """
        if self._is_closed:
            logger.warning("Moonshine STT is closed, ignoring audio")
            return None

        # Store the current user context
        self._current_user = user_metadata  # type: ignore[assignment]

        try:
            # Log incoming audio details for debugging using shared utility
            log_audio_processing_info(pcm_data, self.sample_rate, "Moonshine")

            # Convert PCM data to numpy array using shared utility
            audio_array = pcm_to_numpy_array(pcm_data)

            # Validate and resample if needed using shared utility
            validate_sample_rate_compatibility(
                pcm_data.sample_rate, self.sample_rate, "Moonshine"
            )

            if pcm_data.sample_rate != self.sample_rate:
                audio_array = resample_audio(
                    audio_array, pcm_data.sample_rate, self.sample_rate
                ).astype(np.int16)

            # Normalize audio for Moonshine
            normalized_audio = self._normalize_audio(audio_array)

            # Transcribe the audio immediately
            result = await self._transcribe_audio(normalized_audio)

            if result:
                # Handle both tuple (text, processing_time) and string returns for backward compatibility
                if isinstance(result, tuple):
                    text, processing_time_ms = result
                else:
                    text = result
                    processing_time_ms = 0.0  # Default for mocked tests

                # Create metadata
                metadata = {
                    "model_name": self.model_name,
                    "audio_duration_ms": (len(audio_array) / self.sample_rate) * 1000,
                    "processing_time_ms": processing_time_ms,
                    # Moonshine doesn't provide confidence scores, so we don't set it
                    "original_sample_rate": pcm_data.sample_rate,
                    "target_sample_rate": self.sample_rate,
                    "resampled": pcm_data.sample_rate != self.sample_rate,
                }

                # Return as final transcript (Moonshine doesn't support streaming)
                return [(True, text, metadata)]

            return None

        except Exception as e:
            # Use the base class helper for consistent error handling
            self._emit_error_event(e, "Moonshine audio processing")
            return None

    async def flush(self):
        """
        Flush any remaining audio in the buffer and process it.

        Note: This is a no-op since we no longer buffer audio.
        """
        # No buffering, so nothing to flush
        pass

    async def close(self):
        """Close the Moonshine STT service and clean up resources."""
        if self._is_closed:
            logger.debug("Moonshine STT service already closed")
            return

        logger.info("Closing Moonshine STT service")
        self._is_closed = True

        # No buffers to clear since we don't buffer anymore
        logger.debug("Moonshine STT service closed successfully")
