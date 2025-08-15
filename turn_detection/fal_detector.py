"""Fal.ai Smart Turn detection implementation for Stream video calls.

This module provides turn detection using Fal.ai's Smart Turn API
for intelligent conversation turn-taking in Stream.io video calls.
"""

import logging
import time
import base64
import io
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from threading import Thread, Lock, Event
from queue import Queue, Empty
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from getstream.models import User
from aiortc import MediaStreamTrack

from .turn_detection import (
    BaseTurnDetector,
    TurnEvent,
    TurnEventData,
)

logger = logging.getLogger(__name__)


@dataclass
class FalConfig:
    """Configuration for Fal.ai Smart Turn integration."""

    # Fal.ai API settings
    api_key: Optional[str] = None  # If not provided, uses FAL_KEY env var
    api_url: str = "fal-ai/smart-turn"

    # Audio processing parameters
    sample_rate: int = 16000  # Sample rate for audio processing
    chunk_duration_ms: int = 500  # Duration of audio chunks to send (ms)
    overlap_ms: int = 100  # Overlap between chunks for continuity

    # Turn detection thresholds
    turn_confidence_threshold: float = 0.7  # Minimum confidence for turn detection
    interruption_threshold: float = 0.8  # Threshold for detecting interruptions

    # Buffering and batching
    audio_buffer_size: int = 32000  # Size of audio buffer (samples)
    max_batch_size: int = 5  # Maximum number of chunks to process in batch
    batch_timeout: float = 0.5  # Timeout for batch collection (seconds)

    # Performance tuning
    max_workers: int = 4  # Max concurrent API calls
    enable_caching: bool = True  # Cache recent predictions
    cache_duration: float = 2.0  # How long to cache predictions (seconds)

    # Request throttling and concatenation
    min_request_interval_ms: int = 1000  # Per-user min interval between API calls
    concat_max_ms: int = 1500  # Max audio length to concatenate per request

    # Smart Turn specific features
    enable_context_awareness: bool = True  # Use conversation context
    enable_prosody_analysis: bool = True  # Analyze speech patterns
    enable_semantic_analysis: bool = True  # Analyze content meaning


@dataclass
class AudioChunk:
    """Represents a chunk of audio data for processing."""

    user_id: str
    audio_data: np.ndarray
    timestamp: float
    sample_rate: int
    duration: float


@dataclass
class TurnPrediction:
    """Result from Fal.ai Smart Turn API."""

    user_id: str
    timestamp: float
    is_turn_end: bool
    is_turn_start: bool
    confidence: float
    next_speaker_id: Optional[str] = None
    interruption_detected: bool = False
    prosody_features: Optional[Dict[str, float]] = None
    semantic_features: Optional[Dict[str, Any]] = None


@dataclass
class ConversationContext:
    """Tracks conversation context for better turn prediction."""

    turn_history: List[tuple[str, float, float]] = field(
        default_factory=list
    )  # (user_id, start, end)
    current_topic_embedding: Optional[np.ndarray] = None
    interaction_patterns: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )  # who speaks after whom
    average_turn_duration: Dict[str, float] = field(default_factory=dict)
    last_update: float = 0.0


class FalSmartTurnProcessor:
    """Handles audio processing and Fal.ai API interactions."""

    def __init__(self, config: FalConfig):
        """Initialize the Fal.ai Smart Turn processor.

        Args:
            config: Configuration for Fal.ai integration
        """
        self.config = config
        self._is_initialized = False

        # Import fal_client
        try:
            import fal_client

            self.fal_client = fal_client
            logger.info("Fal client imported successfully")
        except ImportError:
            logger.warning(
                "fal-client not found. Install it with: pip install fal-client"
            )
            # Create a mock client for development/testing
            self.fal_client = self._create_mock_client()

        # Processing queue and thread pool
        self._processing_queue: Queue = Queue()
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)

        # Caching
        self._prediction_cache: Dict[str, TurnPrediction] = {}
        self._cache_lock = Lock()

        # Conversation context
        self._context = ConversationContext()

    def _create_mock_client(self):
        """Create a mock Fal client for development/testing."""

        class MockFalClient:
            """Mock implementation of fal_client for testing."""

            def run(self, model: str, arguments: Dict) -> Dict:
                """Mock run method that returns simulated results."""
                logger.debug(f"Mock Fal API call to {model}")
                # Produce alternating start/end with high confidence so examples work
                now = time.time()
                is_start = int(now * 2) % 2 == 0  # toggle ~every 0.5s
                return {
                    "is_turn_end": not is_start,
                    "is_turn_start": is_start,
                    "confidence": 0.9,
                    "next_speaker": None,
                    "interruption": False,
                    "prosody": {"pitch": 0.6, "energy": 0.6},
                    "semantics": {"topic": "general", "sentiment": "neutral"},
                }

            def __bool__(self):
                """Allow mock client to be used in boolean context."""
                return True

        logger.info("Using mock Fal client for development")
        return MockFalClient()

    def initialize(self) -> bool:
        """Initialize Fal.ai client and verify connection.

        Returns:
            True if initialization successful, False otherwise
        """
        if not self.fal_client:
            logger.error("Fal client not available")
            return False

        try:
            # Configure fal_client
            # Set API key via environment variable or config
            import os

            api_key = self.config.api_key or os.environ.get("FAL_KEY")
            if not api_key:
                logger.warning("FAL_KEY not set. Using mock Fal client for demo mode.")
                self.fal_client = self._create_mock_client()
                self._is_initialized = True
                return True

            if api_key:
                # Prefer client-level API configuration if exposed
                if hasattr(self.fal_client, "set_key"):
                    try:
                        self.fal_client.set_key(api_key)
                        logger.debug("Configured fal_client with API key via set_key")
                    except Exception as e:
                        logger.warning(
                            f"fal_client.set_key failed: {e}; falling back to env var"
                        )
                        os.environ["FAL_KEY"] = api_key
                else:
                    os.environ["FAL_KEY"] = api_key
                    logger.debug("Set FAL_KEY environment variable")

            # Check if we can import and use fal_client
            # The fal_client typically uses FAL_KEY env var automatically
            if hasattr(self.fal_client, "__version__"):
                logger.info(f"Using fal_client version: {self.fal_client.__version__}")

            logger.info("Fal.ai Smart Turn processor initialized")
            self._is_initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Fal.ai client: {e}")
            return False

    def process_audio_chunk(self, chunk: AudioChunk) -> Optional[TurnPrediction]:
        """Process an audio chunk through Fal.ai Smart Turn API.

        Args:
            chunk: Audio chunk to process

        Returns:
            Turn prediction or None if processing fails
        """
        if not self._is_initialized:
            return None

        try:
            # Check cache first
            if self.config.enable_caching:
                cached = self._get_cached_prediction(chunk.user_id, chunk.timestamp)
                if cached:
                    return cached

            # Prepare audio for API: upload WAV to obtain URL (preferred by API)
            audio_bytes = self._encode_wav_bytes(chunk.audio_data, chunk.sample_rate)
            audio_url = self._upload_audio_and_get_url(audio_bytes)
            if not audio_url:
                logger.error("Fal upload did not return a URL; skipping this chunk")
                return None

            # Prepare request with context if enabled
            if audio_url:
                request_data = {
                    "audio_url": audio_url,
                    "sample_rate": chunk.sample_rate,
                    "user_id": chunk.user_id,
                    "timestamp": chunk.timestamp,
                }

            if self.config.enable_context_awareness:
                request_data["context"] = self._get_context_data()

            if self.config.enable_prosody_analysis:
                request_data["analyze_prosody"] = True

            if self.config.enable_semantic_analysis:
                request_data["analyze_semantics"] = True

            # Call Fal.ai API
            # Note: The actual fal_client interface may vary
            # Common patterns include:
            # - fal_client.run(model, arguments)
            # - fal_client.submit(model, arguments)
            # - fal_client.subscribe(model, arguments)

            logger.info("Calling Fal Smart Turn with audio_url=%s", str(audio_url)[:60])
            result = self._invoke_model_with_fallbacks(request_data)
            logger.debug(
                f"[Fal] Raw response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}"
            )

            # Parse response
            prediction = self._parse_api_response(result, chunk)
            if not prediction and hasattr(self.fal_client, "subscribe"):
                # Try subscribe fallback to capture streamed events
                streamed = self._subscribe_for_prediction(request_data, timeout_s=1.5)
                if streamed:
                    prediction = self._parse_api_response(streamed, chunk)
            if not prediction:
                logger.info(f"[Fal] No turn parsed from response: {str(result)[:200]}")

            # Cache the prediction
            if self.config.enable_caching and prediction:
                self._cache_prediction(prediction)

            # Update context
            self._update_context(prediction)

            return prediction

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None

    def _invoke_model_with_fallbacks(
        self, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call the Fal model trying multiple client calling conventions."""
        model = self.config.api_url
        errors: List[str] = []
        # Try common patterns for run
        for fn_name, kwargs in [
            ("run", {"arguments": request_data}),
            ("run", {"input": request_data}),
            ("run", request_data),  # kwargs expanded
        ]:
            try:
                fn = getattr(self.fal_client, fn_name, None)
                if callable(fn):
                    logger.debug(
                        f"Fal call: {fn_name} with keys={list(kwargs if isinstance(kwargs, dict) else {}).copy()}"
                    )
                    if isinstance(kwargs, dict):
                        return fn(model, **kwargs)
                    else:
                        return fn(model, kwargs)
            except Exception as e:
                errors.append(
                    f"{fn_name}({list(kwargs.keys()) if isinstance(kwargs, dict) else 'positional'}): {e}"
                )

        # Try submit
        for submit_kwargs in (
            {"arguments": request_data},
            {"input": request_data},
            request_data,
        ):
            try:
                if hasattr(self.fal_client, "submit"):
                    handle = self.fal_client.submit(
                        model,
                        **(submit_kwargs if isinstance(submit_kwargs, dict) else {}),
                    )
                    if hasattr(handle, "get"):
                        return handle.get()
                    # Context manager style
                    try:
                        with self.fal_client.submit(
                            model,
                            **(
                                submit_kwargs if isinstance(submit_kwargs, dict) else {}
                            ),
                        ) as h:
                            return h.get()
                    except Exception:
                        pass
            except Exception as e:
                errors.append(
                    f"submit({list(submit_kwargs.keys()) if isinstance(submit_kwargs, dict) else 'positional'}): {e}"
                )

        # Try raw HTTP first with multiple body shapes for compatibility
        try:
            http_result = self._http_post_model(model, request_data)
            if http_result is not None:
                return http_result
            errors.append("raw_http: returned None")
        except Exception as e:
            errors.append(f"raw_http: {e}")

        # If everything failed, raise informative error
        raise RuntimeError(f"All Fal call patterns failed: {'; '.join(errors)}")

    def _subscribe_for_prediction(
        self, request_data: Dict[str, Any], timeout_s: float = 1.5
    ) -> Optional[Dict[str, Any]]:
        """Attempt to subscribe to the model stream and return the first useful event."""
        try:
            if not hasattr(self.fal_client, "subscribe"):
                return None
            start = time.time()
            # Some clients expose generator; some use context manager
            sub = self.fal_client.subscribe(self.config.api_url, arguments=request_data)
            iterator = None
            try:
                # Context manager style
                context = getattr(sub, "__enter__", None)
                if callable(context):
                    with sub as stream:
                        iterator = stream
                        for event in iterator:
                            if self._looks_like_prediction(event):
                                return event
                            if time.time() - start > timeout_s:
                                break
                else:
                    iterator = sub
                    for event in iterator:
                        if self._looks_like_prediction(event):
                            return event
                        if time.time() - start > timeout_s:
                            break
            except Exception as e:
                logger.debug(f"Fal subscribe failed: {e}")
                return None
        except Exception as e:
            logger.debug(f"Subscribe setup error: {e}")
            return None
        return None

    def _looks_like_prediction(self, event: Any) -> bool:
        if not isinstance(event, dict):
            return False
        payload = event
        # Some stream events wrap in 'data' or 'result'
        if "data" in payload and isinstance(payload["data"], dict):
            payload = payload["data"]
        keys = set(payload.keys())
        signal_keys = {
            "is_turn_start",
            "is_turn_end",
            "turn_start",
            "turn_end",
            "speech_started",
            "speech_ended",
        }
        return len(signal_keys.intersection(keys)) > 0

    def _http_post_model(
        self, model: str, body: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """HTTP fallback for posting directly to Fal endpoint.

        Requires FAL_KEY in environment. Sends JSON {audio_url: ...}.
        """
        try:
            import os
            import requests

            api_key = os.environ.get("FAL_KEY")
            if not api_key:
                logger.warning("HTTP fallback skipped: FAL_KEY not set")
                return None
            url = f"https://fal.run/{model}"
            headers = {
                "Authorization": f"Key {api_key}",
                "Content-Type": "application/json",
            }
            # Try top-level body
            for payload in (body, {"input": body}, {"arguments": body}):
                resp = requests.post(url, headers=headers, json=payload, timeout=10)
                logger.debug(
                    f"HTTP fallback status={resp.status_code} for payload keys={list(payload.keys())}"
                )
                if 200 <= resp.status_code < 300:
                    return resp.json()
                # If explicit 422 audio_url missing, try next payload shape
                if resp.status_code == 422:
                    continue
            # Exhausted payload variants
            logger.error(f"HTTP fallback error last={resp.status_code}: {resp.text}")
            return None
        except Exception as e:
            logger.error(f"HTTP fallback exception: {e}")
            return None

    def process_batch(self, chunks: List[AudioChunk]) -> List[Optional[TurnPrediction]]:
        """Process multiple audio chunks in batch.

        Args:
            chunks: List of audio chunks to process

        Returns:
            List of turn predictions
        """
        if not chunks:
            return []

        # Submit all chunks for parallel processing
        futures = []
        for chunk in chunks:
            future = self._executor.submit(self.process_audio_chunk, chunk)
            futures.append(future)

        # Collect results
        predictions = []
        for future in futures:
            try:
                prediction = future.result(timeout=2.0)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                predictions.append(None)

        return predictions

    def _encode_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Encode audio data to base64 for API transmission.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate

        Returns:
            Base64 encoded audio string
        """
        # Convert to 16-bit PCM
        if audio_data.dtype != np.int16:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data

        # Create WAV format in memory
        import wave

        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        # Encode to base64
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return audio_base64

    def _encode_wav_bytes(self, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Encode audio data to WAV bytes for upload.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate

        Returns:
            WAV file bytes
        """
        # Convert to 16-bit PCM
        if audio_data.dtype != np.int16:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data

        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        buffer.seek(0)
        return buffer.read()

    def _upload_audio_and_get_url(self, audio_bytes: bytes) -> Optional[str]:
        """Upload audio bytes to Fal storage and return a URL.

        Tries multiple strategies to maximize compatibility with fal_client versions.
        """
        try:
            if hasattr(self.fal_client, "upload"):
                # Try direct bytes upload
                try:
                    # Some clients require positional content_type
                    url = self.fal_client.upload(audio_bytes, "audio/wav")
                    if url:
                        return self._coerce_upload_result(url)
                except Exception as e:
                    logger.debug(f"fal_client.upload(bytes, 'audio/wav') failed: {e}")

                # Try with keyword content_type
                try:
                    url = self.fal_client.upload(audio_bytes, content_type="audio/wav")
                    if url:
                        return self._coerce_upload_result(url)
                except Exception as e:
                    logger.debug(
                        f"fal_client.upload(bytes, content_type='audio/wav') failed: {e}"
                    )

                # Try with bytes + content_type + filename (positional)
                try:
                    url = self.fal_client.upload(audio_bytes, "audio/wav", "audio.wav")
                    if url:
                        return self._coerce_upload_result(url)
                except Exception as e:
                    logger.debug(
                        f"fal_client.upload(bytes, 'audio/wav', 'audio.wav') failed: {e}"
                    )

                # Try kwargs: data + filename + content_type
                try:
                    url = self.fal_client.upload(
                        data=audio_bytes,
                        filename="audio.wav",
                        content_type="audio/wav",
                    )
                    if url:
                        return self._coerce_upload_result(url)
                except Exception as e:
                    logger.debug(
                        f"fal_client.upload(data=..., filename=..., content_type=...) failed: {e}"
                    )

                # Try temp file upload
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                try:
                    url = self.fal_client.upload(tmp_path)
                    return self._coerce_upload_result(url)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            # Some clients expose 'storage.upload'
            storage = getattr(self.fal_client, "storage", None)
            if storage and hasattr(storage, "upload"):
                try:
                    url = storage.upload(audio_bytes, "audio/wav")
                    return self._coerce_upload_result(url)
                except Exception as e:
                    logger.debug(
                        f"fal_client.storage.upload(bytes, 'audio/wav') failed: {e}"
                    )
                try:
                    url = storage.upload(
                        data=audio_bytes,
                        filename="audio.wav",
                        content_type="audio/wav",
                    )
                    return self._coerce_upload_result(url)
                except Exception as e:
                    logger.debug(
                        f"fal_client.storage.upload(data=..., filename=..., content_type=...) failed: {e}"
                    )

            # Fallback to upload_file API (used in some examples)
            if hasattr(self.fal_client, "upload_file"):
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                try:
                    upload_obj = self.fal_client.upload_file(tmp_path)
                    return self._coerce_upload_result(upload_obj)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"Audio upload error: {e}")
            return None

        # As a last resort, return a data URL so the API receives an audio_url.
        # Some servers may not accept data URLs; this at least avoids missing-field errors.
        try:
            b64 = base64.b64encode(audio_bytes).decode("utf-8")
            data_url = f"data:audio/wav;base64,{b64}"
            logger.debug("Using data URL fallback for audio upload")
            return data_url
        except Exception:
            return None

    def _coerce_upload_result(self, upload_result: Any) -> Optional[str]:
        """Normalize various fal_client upload return types into a URL string."""
        try:
            # If already a string
            if isinstance(upload_result, str):
                return upload_result
            # Dict-like with url field
            if isinstance(upload_result, dict):
                # direct
                if "url" in upload_result and isinstance(upload_result["url"], str):
                    return upload_result["url"]
                # nested patterns
                for key in ("file", "data", "result", "upload"):
                    nested = upload_result.get(key)
                    if (
                        isinstance(nested, dict)
                        and "url" in nested
                        and isinstance(nested["url"], str)
                    ):
                        return nested["url"]
            # Objects with .url attribute
            url_attr = getattr(upload_result, "url", None)
            if url_attr:
                return url_attr if isinstance(url_attr, str) else str(url_attr)
        except Exception:
            pass
        return None

    def _parse_api_response(
        self, response: Dict[str, Any], chunk: AudioChunk
    ) -> Optional[TurnPrediction]:
        """Parse Fal.ai API response into TurnPrediction with robust key handling."""
        try:
            if not isinstance(response, dict):
                logger.debug(f"Unexpected Fal response type: {type(response)}")
                return None

            # Some clients may wrap result
            resp = response
            for key in ("result", "output", "data"):
                if isinstance(resp.get(key), dict):
                    resp = resp[key]
                    break

            # Map booleans from various possible keys
            def any_bool(keys, default=False):
                for k in keys:
                    if k in resp:
                        return bool(resp[k])
                return default

            def any_val(keys):
                for k in keys:
                    if k in resp:
                        return resp[k]
                return None

            is_turn_start = any_bool(
                [
                    "is_turn_start",
                    "turn_start",
                    "start_of_turn",
                    "is_start_of_turn",
                    "speech_started",
                ],
                False,
            )
            is_turn_end = any_bool(
                [
                    "is_turn_end",
                    "turn_end",
                    "end_of_turn",
                    "is_end_of_turn",
                    "speech_ended",
                ],
                False,
            )
            confidence = any_val(["confidence", "score", "probability"]) or 0.0
            next_speaker_id = any_val(
                ["next_speaker_id", "next_speaker", "predicted_next_speaker"]
            )
            if next_speaker_id is None and isinstance(
                resp.get("next_speaker_prediction"), dict
            ):
                next_speaker_id = resp["next_speaker_prediction"].get("user_id")
            interruption_detected = any_bool(
                ["interruption", "interruption_detected", "is_interrupt"], False
            )
            prosody = any_val(["prosody", "prosody_features"]) or None
            semantics = any_val(["semantics", "semantic", "semantic_features"]) or None

            # If neither start nor end flags are present but a generic 'turn' exists
            if (
                not is_turn_start
                and not is_turn_end
                and isinstance(resp.get("turn"), str)
            ):
                if resp["turn"].lower() == "start":
                    is_turn_start = True
                elif resp["turn"].lower() == "end":
                    is_turn_end = True

            # Default confidence when a flag is asserted
            if confidence == 0.0 and (is_turn_start or is_turn_end):
                confidence = 0.8

            logger.debug(
                f"[Fal] Parsed response -> start={is_turn_start} end={is_turn_end} conf={confidence} next={next_speaker_id} interrupt={interruption_detected}"
            )

            return TurnPrediction(
                user_id=chunk.user_id,
                timestamp=chunk.timestamp,
                is_turn_end=is_turn_end,
                is_turn_start=is_turn_start,
                confidence=float(confidence),
                next_speaker_id=next_speaker_id,
                interruption_detected=interruption_detected,
                prosody_features=prosody,
                semantic_features=semantics,
            )

        except Exception as e:
            logger.error(f"Error parsing API response: {e}")
            return None

    def _get_context_data(self) -> Dict[str, Any]:
        """Get conversation context for API request.

        Returns:
            Context dictionary
        """
        with self._cache_lock:
            return {
                "turn_history": self._context.turn_history[-10:],  # Last 10 turns
                "interaction_patterns": self._context.interaction_patterns,
                "average_turn_duration": self._context.average_turn_duration,
            }

    def _update_context(self, prediction: Optional[TurnPrediction]):
        """Update conversation context with new prediction.

        Args:
            prediction: Turn prediction to incorporate
        """
        if not prediction:
            return

        with self._cache_lock:
            # Update turn history if turn ended
            if prediction.is_turn_end:
                # Find the turn start in history
                for i in range(len(self._context.turn_history) - 1, -1, -1):
                    if self._context.turn_history[i][0] == prediction.user_id:
                        # Update end time
                        start_time = self._context.turn_history[i][1]
                        self._context.turn_history[i] = (
                            prediction.user_id,
                            start_time,
                            prediction.timestamp,
                        )

                        # Update average duration
                        duration = prediction.timestamp - start_time
                        if prediction.user_id in self._context.average_turn_duration:
                            avg = self._context.average_turn_duration[
                                prediction.user_id
                            ]
                            self._context.average_turn_duration[prediction.user_id] = (
                                avg * 0.8 + duration * 0.2  # Exponential moving average
                            )
                        else:
                            self._context.average_turn_duration[prediction.user_id] = (
                                duration
                            )
                        break

            # Add new turn start
            if prediction.is_turn_start:
                self._context.turn_history.append(
                    (prediction.user_id, prediction.timestamp, 0.0)
                )

            # Update interaction patterns
            if prediction.next_speaker_id and len(self._context.turn_history) > 0:
                last_speaker = self._context.turn_history[-1][0]
                if last_speaker not in self._context.interaction_patterns:
                    self._context.interaction_patterns[last_speaker] = {}

                patterns = self._context.interaction_patterns[last_speaker]
                if prediction.next_speaker_id in patterns:
                    patterns[prediction.next_speaker_id] += 1
                else:
                    patterns[prediction.next_speaker_id] = 1

            self._context.last_update = time.time()

    def _get_cached_prediction(
        self, user_id: str, timestamp: float
    ) -> Optional[TurnPrediction]:
        """Get cached prediction if available and valid.

        Args:
            user_id: User ID
            timestamp: Timestamp to check

        Returns:
            Cached prediction or None
        """
        with self._cache_lock:
            cache_key = f"{user_id}:{int(timestamp * 10)}"  # 100ms resolution

            if cache_key in self._prediction_cache:
                cached = self._prediction_cache[cache_key]

                # Check if cache is still valid
                if timestamp - cached.timestamp < self.config.cache_duration:
                    return cached
                else:
                    # Remove expired cache
                    del self._prediction_cache[cache_key]

            return None

    def _cache_prediction(self, prediction: TurnPrediction):
        """Cache a prediction for future use.

        Args:
            prediction: Prediction to cache
        """
        with self._cache_lock:
            cache_key = f"{prediction.user_id}:{int(prediction.timestamp * 10)}"
            self._prediction_cache[cache_key] = prediction

            # Clean old cache entries
            current_time = time.time()
            expired_keys = [
                key
                for key, pred in self._prediction_cache.items()
                if current_time - pred.timestamp > self.config.cache_duration
            ]
            for key in expired_keys:
                del self._prediction_cache[key]

    def cleanup(self):
        """Clean up resources."""
        self._executor.shutdown(wait=False)
        self._prediction_cache.clear()
        self._is_initialized = False


class FalSmartTurnDetector(BaseTurnDetector):
    """Turn detection implementation using Fal.ai Smart Turn API.

    This detector uses Fal.ai's Smart Turn model for intelligent turn detection
    with context awareness, prosody analysis, and semantic understanding.
    """

    def __init__(
        self,
        mini_pause_duration: float = 0.5,
        max_pause_duration: float = 2.0,
        config: Optional[FalConfig] = None,
    ):
        """Initialize the Fal Smart Turn detector.

        Args:
            mini_pause_duration: Duration for mini pause detection
            max_pause_duration: Duration for max pause detection
            config: Fal configuration (uses defaults if not provided)
        """
        super().__init__(mini_pause_duration, max_pause_duration)

        self.config = config or FalConfig()
        self._processor = FalSmartTurnProcessor(self.config)

        # Speaker tracking
        self._speakers: Dict[str, User] = {}
        self._speaker_buffers: Dict[str, List[np.ndarray]] = {}
        self._lock = Lock()

        # Processing thread and queue
        self._processing_thread: Optional[Thread] = None
        self._chunk_queue: Queue = Queue()
        self._stop_event = Event()

        # Batch processing
        self._batch_buffer: List[AudioChunk] = []
        self._batch_lock = Lock()
        self._last_batch_time = time.time()

        # Turn state tracking
        self._current_speaker: Optional[str] = None
        self._pending_speaker: Optional[str] = None
        self._turn_transition_confidence: float = 0.0
        # Track last speech end to compute max pause reliably
        self._last_speech_end_time: float = 0.0
        # Throttling per user
        self._last_request_ts_by_user: Dict[str, float] = {}

        logger.info(
            f"Initialized FalSmartTurnDetector with mini_pause={mini_pause_duration}s, "
            f"max_pause={max_pause_duration}s"
        )

    def add_participant(self, user: User) -> None:
        """Add a participant for turn detection tracking.

        Args:
            user: The user to add as a participant
        """
        with self._lock:
            user_id = user.id
            self._speakers[user_id] = user
            self._speaker_buffers[user_id] = []

            user_name = user.custom.get("name", "Unknown") if user.custom else "Unknown"
            logger.info(f"Added participant: {user_name} ({user_id})")

    def remove_participant(self, user_id: str) -> None:
        """Remove a participant from turn detection.

        Args:
            user_id: The ID of the user to remove
        """
        with self._lock:
            if user_id in self._speakers:
                user = self._speakers[user_id]
                user_name = (
                    user.custom.get("name", "Unknown") if user.custom else "Unknown"
                )

                # End turn if this was the current speaker
                if self._current_speaker == user_id:
                    self._handle_turn_end(user_id, time.time())

                del self._speakers[user_id]
                del self._speaker_buffers[user_id]
                logger.info(f"Removed participant: {user_name} ({user_id})")

    def start_detection(self) -> None:
        """Start the turn detection process."""
        if self._is_detecting:
            logger.warning("Turn detection already started")
            return

        # Initialize Fal processor
        if not self._processor.initialize():
            logger.error("Failed to initialize Fal processor")
            return

        super().start_detection()

        # Start processing thread
        self._stop_event.clear()
        self._processing_thread = Thread(target=self._processing_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()

        logger.info(
            f"Started Fal Smart Turn detection with {len(self._speakers)} participants"
        )

    def stop_detection(self) -> None:
        """Stop the turn detection process."""
        if not self._is_detecting:
            logger.warning("Turn detection not running")
            return

        super().stop_detection()

        # Stop processing thread
        self._stop_event.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
            self._processing_thread = None

        # Process any remaining chunks
        self._process_batch(force=True)

        # Clean up processor
        self._processor.cleanup()

        # End any ongoing turn
        if self._current_speaker:
            self._handle_turn_end(self._current_speaker, time.time())

        logger.info("Stopped Fal Smart Turn detection")

    def add_audio_samples(
        self,
        user_id: str,
        audio_samples: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> None:
        """Ingest audio samples for a user and enqueue chunks for processing.

        This method enables adapters to feed decoded PCM samples directly
        without using a WebRTC track. It mirrors the logic in
        `process_audio_track`, handling chunking and overlap.

        Args:
            user_id: The speaker's user id
            audio_samples: Mono float32 samples in [-1, 1] or int16
            timestamp: Optional timestamp; defaults to current time
        """
        if not self._is_detecting:
            return

        # Normalize dtype to float32 in [-1, 1]
        if audio_samples.dtype == np.int16:
            samples = audio_samples.astype(np.float32) / 32768.0
        else:
            samples = audio_samples.astype(np.float32)

        chunk_samples = int(
            self.config.sample_rate * self.config.chunk_duration_ms / 1000
        )
        overlap_samples = int(self.config.sample_rate * self.config.overlap_ms / 1000)
        event_time = timestamp or time.time()

        with self._lock:
            if user_id not in self._speaker_buffers:
                # Unknown user; ignore safely
                return

            # Extend buffer
            buffer = self._speaker_buffers[user_id]
            buffer.extend(samples.tolist())
            logger.debug(
                f"[Fal] Buffered samples for {user_id}: +{len(samples)} (buffer={len(buffer)})"
            )

            # Limit buffer size
            max_size = self.config.audio_buffer_size
            if len(buffer) > max_size:
                buffer = buffer[-max_size:]
                self._speaker_buffers[user_id] = buffer

            # While we have enough for a chunk, enqueue and retain overlap
            while len(buffer) >= chunk_samples:
                chunk_data = np.array(buffer[:chunk_samples], dtype=np.float32)

                logger.debug(
                    f"[Fal] Enqueue chunk for {user_id}: {len(chunk_data)} samples"
                )
                chunk = AudioChunk(
                    user_id=user_id,
                    audio_data=chunk_data,
                    timestamp=event_time,
                    sample_rate=self.config.sample_rate,
                    duration=self.config.chunk_duration_ms / 1000.0,
                )
                self._chunk_queue.put(chunk)

                # Keep overlap
                buffer = buffer[chunk_samples - overlap_samples :]
                self._speaker_buffers[user_id] = buffer

    async def process_audio_track(self, track: MediaStreamTrack, user_id: str) -> None:
        """Process audio from a WebRTC media track.

        Args:
            track: The audio track to process
            user_id: The ID of the user associated with this track
        """
        if user_id not in self._speakers:
            logger.warning(f"User {user_id} not registered as participant")
            return

        logger.info(f"Starting audio processing for user {user_id}")

        chunk_samples = int(
            self.config.sample_rate * self.config.chunk_duration_ms / 1000
        )
        overlap_samples = int(self.config.sample_rate * self.config.overlap_ms / 1000)

        try:
            while self._is_detecting:
                # Receive audio frame from track
                frame = await track.recv()

                # Convert to numpy array
                audio_data = (
                    np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16).astype(
                        np.float32
                    )
                    / 32768.0
                )

                # Add to buffer
                self._add_to_buffer(user_id, audio_data)

                # Check if we have enough samples for a chunk
                buffer = self._get_buffer(user_id)
                if len(buffer) >= chunk_samples:
                    # Extract chunk with overlap
                    chunk_data = buffer[:chunk_samples]

                    # Create audio chunk
                    chunk = AudioChunk(
                        user_id=user_id,
                        audio_data=np.array(chunk_data),
                        timestamp=time.time(),
                        sample_rate=self.config.sample_rate,
                        duration=self.config.chunk_duration_ms / 1000.0,
                    )

                    # Queue for processing
                    self._chunk_queue.put(chunk)

                    # Update buffer (keep overlap)
                    self._update_buffer(
                        user_id, buffer[chunk_samples - overlap_samples :]
                    )

        except Exception as e:
            logger.error(f"Error processing audio track for {user_id}: {e}")

    def _add_to_buffer(self, user_id: str, audio_data: np.ndarray):
        """Add audio data to user's buffer.

        Args:
            user_id: User ID
            audio_data: Audio samples to add
        """
        with self._lock:
            if user_id in self._speaker_buffers:
                buffer = self._speaker_buffers[user_id]
                buffer.extend(audio_data.tolist())

                # Limit buffer size
                max_size = self.config.audio_buffer_size
                if len(buffer) > max_size:
                    self._speaker_buffers[user_id] = buffer[-max_size:]

    def _get_buffer(self, user_id: str) -> List[float]:
        """Get user's audio buffer.

        Args:
            user_id: User ID

        Returns:
            Audio buffer
        """
        with self._lock:
            return self._speaker_buffers.get(user_id, []).copy()

    def _update_buffer(self, user_id: str, new_buffer: List[float]):
        """Update user's audio buffer.

        Args:
            user_id: User ID
            new_buffer: New buffer content
        """
        with self._lock:
            if user_id in self._speaker_buffers:
                self._speaker_buffers[user_id] = new_buffer

    def _processing_loop(self):
        """Main processing loop for handling audio chunks."""
        logger.debug("Started Smart Turn processing loop")

        while not self._stop_event.is_set():
            try:
                # Collect chunks for batch processing
                timeout = self.config.batch_timeout

                try:
                    chunk = self._chunk_queue.get(timeout=timeout)

                    with self._batch_lock:
                        self._batch_buffer.append(chunk)

                        # Process batch if it's full
                        if len(self._batch_buffer) >= self.config.max_batch_size:
                            self._process_batch()

                except Empty:
                    # Timeout - process any pending chunks
                    self._process_batch(force=True)

                # Check if batch timeout exceeded
                current_time = time.time()
                if current_time - self._last_batch_time > self.config.batch_timeout:
                    self._process_batch(force=True)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")

        logger.debug("Stopped Smart Turn processing loop")

    def _process_batch(self, force: bool = False):
        """Process accumulated batch of audio chunks with per-user throttling and concatenation."""
        with self._batch_lock:
            if not self._batch_buffer:
                return
            # Always process when called due to timeout; otherwise wait for at least 2
            if not force and len(self._batch_buffer) < 2:
                return
            # Take and reset batch
            pending = self._batch_buffer.copy()
            self._batch_buffer.clear()
            self._last_batch_time = time.time()

        # Group by user and build concatenated chunks
        by_user: Dict[str, List[AudioChunk]] = {}
        for ch in pending:
            by_user.setdefault(ch.user_id, []).append(ch)

        to_submit: List[AudioChunk] = []
        now = time.time()
        min_interval = self.config.min_request_interval_ms / 1000.0
        max_samples = int(
            self.config.sample_rate * (self.config.concat_max_ms / 1000.0)
        )

        for user_id, arr in by_user.items():
            # Throttle
            last_ts = self._last_request_ts_by_user.get(user_id, 0.0)
            if not force and (now - last_ts) < min_interval:
                continue
            # Sort by timestamp and take most recent chunks; concatenate from the end backwards up to max_samples
            arr.sort(key=lambda c: c.timestamp)
            concat = []
            total = 0
            for ch in reversed(arr):
                s = ch.audio_data
                if total + len(s) > max_samples:
                    # Trim to fit
                    need = max(0, max_samples - total)
                    if need > 0:
                        concat.append(s[-need:])
                        total += need
                    break
                concat.append(s)
                total += len(s)
                if total >= max_samples:
                    break
            if not concat:
                continue
            concat.reverse()
            audio_combined = np.concatenate(concat).astype(np.float32)
            combined = AudioChunk(
                user_id=user_id,
                audio_data=audio_combined,
                timestamp=arr[-1].timestamp,
                sample_rate=self.config.sample_rate,
                duration=len(audio_combined) / self.config.sample_rate,
            )
            to_submit.append(combined)
            self._last_request_ts_by_user[user_id] = now

        if not to_submit:
            return

        logger.debug(f"[Fal] Processing batch for {len(to_submit)} user(s)")
        predictions = self._processor.process_batch(to_submit)

        for pred in predictions:
            if pred:
                logger.info(
                    f"[Fal] Prediction for {pred.user_id}: start={pred.is_turn_start} end={pred.is_turn_end} conf={pred.confidence:.2f}"
                )
                self._handle_prediction(pred)

    def _handle_prediction(self, prediction: TurnPrediction):
        """Handle a turn prediction from Fal.ai.

        Args:
            prediction: Turn prediction to handle
        """
        user_id = prediction.user_id

        if user_id not in self._speakers:
            return

        user = self._speakers[user_id]

        # Check confidence threshold
        if prediction.confidence < self.config.turn_confidence_threshold:
            return

        # Handle interruption
        if prediction.interruption_detected:
            self._handle_interruption(user_id, prediction)

        # Handle turn end
        if prediction.is_turn_end:
            self._handle_smart_turn_end(user_id, prediction)

        # Handle turn start
        if prediction.is_turn_start:
            self._handle_smart_turn_start(user_id, prediction)

        # Handle next speaker prediction
        if prediction.next_speaker_id and prediction.next_speaker_id in self._speakers:
            self._pending_speaker = prediction.next_speaker_id
            self._turn_transition_confidence = prediction.confidence

    def _handle_interruption(self, user_id: str, prediction: TurnPrediction):
        """Handle an interruption detection.

        Args:
            user_id: Interrupting user
            prediction: Turn prediction
        """
        if prediction.confidence < self.config.interruption_threshold:
            return

        user = self._speakers[user_id]

        # Emit custom interruption event
        event_data = TurnEventData(
            timestamp=prediction.timestamp,
            speaker=user,
            confidence=prediction.confidence,
            custom={
                "type": "interruption",
                "interrupted_speaker": self._current_speaker,
            },
        )

        # End current turn
        if self._current_speaker and self._current_speaker != user_id:
            self._handle_turn_end(self._current_speaker, prediction.timestamp)

        # Start new turn for interrupter
        self._handle_turn_start(user_id, prediction.timestamp, prediction.confidence)

        logger.info(
            f"Interruption detected: {user_id} interrupted {self._current_speaker}"
        )

    def _handle_smart_turn_end(self, user_id: str, prediction: TurnPrediction):
        """Handle smart turn end detection.

        Args:
            user_id: User ending turn
            prediction: Turn prediction
        """
        if self._current_speaker != user_id:
            return

        user = self._speakers[user_id]

        # Emit speech ended event
        event_data = TurnEventData(
            timestamp=prediction.timestamp,
            speaker=user,
            confidence=prediction.confidence,
            custom={
                "prosody": prediction.prosody_features,
                "semantics": prediction.semantic_features,
            },
        )
        self._emit_turn_event(TurnEvent.SPEECH_ENDED, event_data)

        # Handle turn end
        self._handle_turn_end(user_id, prediction.timestamp)

        # Check for predicted next speaker
        if self._pending_speaker:
            logger.debug(
                f"Next speaker predicted: {self._pending_speaker} "
                f"(confidence: {self._turn_transition_confidence:.2f})"
            )

    def _handle_smart_turn_start(self, user_id: str, prediction: TurnPrediction):
        """Handle smart turn start detection.

        Args:
            user_id: User starting turn
            prediction: Turn prediction
        """
        user = self._speakers[user_id]

        # Check if this was the predicted speaker
        was_predicted = user_id == self._pending_speaker

        # Emit speech started event
        event_data = TurnEventData(
            timestamp=prediction.timestamp,
            speaker=user,
            confidence=prediction.confidence,
            custom={
                "was_predicted": was_predicted,
                "prosody": prediction.prosody_features,
                "semantics": prediction.semantic_features,
            },
        )
        self._emit_turn_event(TurnEvent.SPEECH_STARTED, event_data)

        # Handle turn start
        self._handle_turn_start(user_id, prediction.timestamp, prediction.confidence)

        # Clear pending speaker
        self._pending_speaker = None
        self._turn_transition_confidence = 0.0

    def _handle_turn_start(self, user_id: str, timestamp: float, confidence: float):
        """Handle the start of a turn.

        Args:
            user_id: User starting turn
            timestamp: Event timestamp
            confidence: Detection confidence
        """
        if self._current_speaker == user_id:
            return  # Already speaking

        # End previous speaker's turn if any
        if self._current_speaker:
            self._handle_turn_end(self._current_speaker, timestamp)

        self._current_speaker = user_id
        user = self._speakers[user_id]

        # Emit turn started event
        event_data = TurnEventData(
            timestamp=timestamp,
            speaker=user,
            confidence=confidence,
        )
        self._emit_turn_event(TurnEvent.TURN_STARTED, event_data)

        user_name = user.custom.get("name", "Unknown") if user.custom else "Unknown"
        logger.debug(f"Turn started: {user_name} ({user_id})")

    def _handle_turn_end(self, user_id: str, timestamp: float):
        """Handle the end of a turn.

        Args:
            user_id: User ending turn
            timestamp: Event timestamp
        """
        if self._current_speaker != user_id:
            return

        user = self._speakers[user_id]
        self._current_speaker = None

        # Emit turn ended event
        event_data = TurnEventData(
            timestamp=timestamp,
            speaker=user,
        )
        self._emit_turn_event(TurnEvent.TURN_ENDED, event_data)

        # Check for max pause only if we have a previous end timestamp
        if self._last_speech_end_time > 0:
            if timestamp - self._last_speech_end_time >= self.max_pause_duration:
                self._emit_turn_event(TurnEvent.MAX_PAUSE_REACHED, event_data)

        self._last_speech_end_time = timestamp

        user_name = user.custom.get("name", "Unknown") if user.custom else "Unknown"
        logger.debug(f"Turn ended: {user_name} ({user_id})")

    def get_speaker_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific speaker.

        Args:
            user_id: The ID of the user

        Returns:
            Dictionary with speaker statistics or None if not found
        """
        with self._lock:
            if user_id not in self._speakers:
                return None

            # Get context data from processor
            context = self._processor._context

            # Calculate average turn duration for this speaker
            avg_duration = context.average_turn_duration.get(user_id, 0.0)

            # Count interactions
            interactions = context.interaction_patterns.get(user_id, {})

            return {
                "user_id": user_id,
                "is_current_speaker": self._current_speaker == user_id,
                "is_pending_speaker": self._pending_speaker == user_id,
                "average_turn_duration": avg_duration,
                "interaction_count": sum(interactions.values()),
                "frequent_successors": interactions,
            }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all speakers.

        Returns:
            Dictionary with overall statistics
        """
        with self._lock:
            context = self._processor._context

            stats = {
                "participants": len(self._speakers),
                "current_speaker": self._current_speaker,
                "pending_speaker": self._pending_speaker,
                "transition_confidence": self._turn_transition_confidence,
                "turn_count": len(context.turn_history),
                "speakers": {},
            }

            for user_id in self._speakers:
                stats["speakers"][user_id] = self.get_speaker_stats(user_id)

            return stats

    def get_conversation_insights(self) -> Dict[str, Any]:
        """Get advanced conversation insights from Smart Turn analysis.

        Returns:
            Dictionary with conversation insights
        """
        context = self._processor._context

        insights = {
            "turn_patterns": {},
            "dominant_speakers": [],
            "interaction_balance": 0.0,
            "average_turn_length": 0.0,
        }

        # Analyze turn patterns
        total_turns = len(context.turn_history)
        if total_turns > 0:
            speaker_turns = {}
            total_duration = 0.0

            for user_id, start, end in context.turn_history:
                if user_id not in speaker_turns:
                    speaker_turns[user_id] = {"count": 0, "duration": 0.0}

                speaker_turns[user_id]["count"] += 1
                if end > 0:
                    duration = end - start
                    speaker_turns[user_id]["duration"] += duration
                    total_duration += duration

            # Calculate dominance
            insights["turn_patterns"] = speaker_turns
            insights["dominant_speakers"] = sorted(
                speaker_turns.keys(),
                key=lambda x: speaker_turns[x]["duration"],
                reverse=True,
            )[:3]

            # Calculate balance (0 = unbalanced, 1 = perfectly balanced)
            if len(speaker_turns) > 1:
                durations = [s["duration"] for s in speaker_turns.values()]
                avg_duration = sum(durations) / len(durations)
                variance = sum((d - avg_duration) ** 2 for d in durations) / len(
                    durations
                )
                insights["interaction_balance"] = 1.0 / (
                    1.0 + variance / (avg_duration**2)
                )

            # Average turn length
            if total_duration > 0:
                insights["average_turn_length"] = total_duration / total_turns

        return insights

    def __del__(self):
        """Cleanup when the detector is destroyed."""
        if self._is_detecting:
            self.stop_detection()
