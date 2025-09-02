import asyncio
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import os
import time

# Conditional imports with error handling
try:
    from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

    _deepgram_available = True
except ImportError:
    DeepgramClient = None  # type: ignore
    LiveTranscriptionEvents = None  # type: ignore
    LiveOptions = None  # type: ignore
    _deepgram_available = False

from getstream.plugins.common import STT
from getstream.video.rtc.track_util import PcmData

logger = logging.getLogger(__name__)


class DeepgramSTT(STT):
    """
    Deepgram-based Speech-to-Text implementation.

    This implementation operates in asynchronous mode - it receives streaming transcripts
    from Deepgram's WebSocket connection and emits events immediately as they arrive,
    providing real-time responsiveness for live transcription scenarios.

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
        api_key: Optional[str] = None,
        options: Optional[LiveOptions] = None,  # type: ignore
        sample_rate: int = 48000,
        language: str = "en-US",
        keep_alive_interval: float = 3.0,
        interim_results: bool = False,
        client: Optional[DeepgramClient] = None,
    ):
        """
        Initialize the Deepgram STT service.

        Args:
            api_key: Deepgram API key. If not provided, the DEEPGRAM_API_KEY
                    environment variable will be used automatically.
            options: Deepgram live transcription options
            sample_rate: Sample rate of the audio in Hz (default: 48000)
            language: Language code for transcription
            keep_alive_interval: Interval in seconds to send keep-alive messages.
                                Default is 5.0 seconds (recommended value by Deepgram)
            interim_results: Whether to emit interim results (partial transcripts with the partial_transcript event).
        """
        super().__init__(sample_rate=sample_rate)

        # Check if deepgram is available
        if not _deepgram_available:
            raise ImportError("deepgram package not installed.")

        # If no API key was provided, check for DEEPGRAM_API_KEY in environment
        if api_key is None:
            api_key = os.environ.get("DEEPGRAM_API_KEY")
            if not api_key:
                logger.warning(
                    "No API key provided and DEEPGRAM_API_KEY environment variable not found."
                )

        # Initialize DeepgramClient with the API key
        logger.info("Initializing Deepgram client")
        self.deepgram = client if client is not None else DeepgramClient(api_key)
        self.dg_connection = None
        self.options = options or LiveOptions(
            model="nova-2",
            language=language,
            encoding="linear16",
            sample_rate=sample_rate,
            channels=1,
            interim_results=interim_results,
        )

        # Keep-alive mechanism
        self.keep_alive_interval = keep_alive_interval
        self.last_activity_time = time.time()
        self.keep_alive_task = None
        self._running = False
        self._setup_attempted = False
        self._is_closed = False

        # Track current user context for associating transcripts with users
        self._current_user = None

        self._setup_connection()

    def _handle_transcript_result(
        self, is_final: bool, text: str, metadata: Dict[str, Any]
    ):
        """
        Handle a transcript result by emitting it immediately.
        """
        # Emit immediately for real-time responsiveness
        if is_final:
            self._emit_transcript_event(text, self._current_user, metadata)
        else:
            self._emit_partial_transcript_event(text, self._current_user, metadata)

        logger.debug(
            "Handled transcript result",
            extra={
                "is_final": is_final,
                "text_length": len(text),
            },
        )

    def _setup_connection(self):
        """Set up the Deepgram connection with event handlers."""
        if self._is_closed:
            logger.warning("Cannot setup connection - Deepgram instance is closed")
            return

        if self.dg_connection is not None:
            logger.debug("Connection already set up, skipping initialization")
            return

        try:
            # Use the newer websocket interface instead of deprecated live
            logger.debug("Setting up Deepgram WebSocket connection")
            self.dg_connection = self.deepgram.listen.websocket.v("1")

            # Handler for transcript results
            def handle_transcript(conn, result=None):
                try:
                    # Update the last activity time
                    self.last_activity_time = time.time()

                    # Check if result is already a dict (from LiveResultResponse)
                    if hasattr(result, "to_dict"):
                        transcript = result.to_dict()
                    elif hasattr(result, "to_json"):
                        transcript = json.loads(result.to_json())
                    elif isinstance(result, (str, bytes, bytearray)):
                        transcript = json.loads(result)
                    else:
                        logger.warning(
                            "Unrecognized transcript format: %s", type(result)
                        )
                        return

                    # Get the transcript text from the response
                    alternatives = transcript.get("channel", {}).get("alternatives", [])
                    if not alternatives:
                        return

                    transcript_text = alternatives[0].get("transcript", "")
                    if not transcript_text:
                        return

                    # Check if this is a final result
                    is_final = transcript.get("is_final", False)

                    # Create metadata with useful information
                    metadata = {
                        "confidence": alternatives[0].get("confidence", 0),
                        "words": alternatives[0].get("words", []),
                        "is_final": is_final,
                        "channel_index": transcript.get("channel_index", 0),
                    }

                    # Handle the result (both collect and emit)
                    self._handle_transcript_result(is_final, transcript_text, metadata)

                    logger.debug(
                        "Received transcript",
                        extra={
                            "is_final": is_final,
                            "text_length": len(transcript_text),
                            "confidence": metadata["confidence"],
                        },
                    )
                except Exception as e:
                    logger.error("Error processing transcript", exc_info=e)
                    # Emit error immediately
                    self._emit_error_event(e, "Deepgram transcript processing")

            # Handler for errors
            def handle_error(conn, error=None):
                # Update the last activity time
                self.last_activity_time = time.time()

                error_text = str(error) if error is not None else "Unknown error"
                logger.error("Deepgram error received: %s", error_text)

                # Emit error immediately
                error_obj = Exception(f"Deepgram error: {error_text}")
                self._emit_error_event(error_obj, "Deepgram connection")

            # Register event handlers directly
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, handle_transcript)
            self.dg_connection.on(LiveTranscriptionEvents.Error, handle_error)

            # Start the connection
            logger.info("Starting Deepgram connection with options %s", self.options)
            self.dg_connection.start(self.options)

            # Start the keep-alive task
            self._running = True
            self._start_keep_alive_task()

        except Exception as e:
            # Log the error and set connection to None
            logger.error("Error setting up Deepgram connection", exc_info=e)
            self.dg_connection = None
            # Emit error immediately
            self._emit_error_event(e, "Deepgram connection setup")

    def _start_keep_alive_task(self):
        """Start the background task that sends keep-alive messages."""
        if self.keep_alive_task is None and self._running:
            logger.debug(
                "Starting keep-alive task with interval %ss", self.keep_alive_interval
            )
            self.keep_alive_task = asyncio.create_task(self._keep_alive_loop())

    async def _keep_alive_loop(self):
        """Background task that sends keep-alive messages to prevent connection timeout."""
        while self._running and self.dg_connection:
            try:
                # Check if we need to send a keep-alive message
                # Deepgram closes connection after 10s of inactivity, so we send
                # keep-alive messages if no activity for keep_alive_interval seconds
                time_since_last_activity = time.time() - self.last_activity_time

                if time_since_last_activity >= self.keep_alive_interval:
                    # Send a keep-alive message
                    logger.debug(
                        "Sending keep-alive message",
                        extra={"time_since_activity": time_since_last_activity},
                    )
                    await self.send_keep_alive()
                    # Update the last activity time
                    self.last_activity_time = time.time()

                # Sleep for a short period before checking again
                # We check more frequently than the interval to ensure timely keep-alive messages
                await asyncio.sleep(min(1.0, self.keep_alive_interval / 2))
            except asyncio.CancelledError:
                # Task was cancelled, exit the loop
                logger.info("Keep-alive task cancelled")
                break
            except Exception as e:
                logger.error("Error in keep-alive loop", exc_info=e)
                # Sleep briefly to avoid tight loop in case of persistent errors
                await asyncio.sleep(1.0)

    async def send_keep_alive(self):
        """Send a keep-alive message to maintain the connection."""
        if not self.dg_connection:
            logger.warning("Cannot send keep-alive: no connection available")
            return False

        try:
            # Create the keep-alive message according to Deepgram docs
            keep_alive_msg = json.dumps({"type": "KeepAlive"})

            # Send as a text message (not binary)
            # The Deepgram SDK doesn't have a send_text method, but the connection
            # does have a method to send non-binary data which we should use
            if hasattr(self.dg_connection, "send_text"):
                self.dg_connection.send_text(keep_alive_msg)
            elif hasattr(self.dg_connection, "keep_alive"):
                # Many SDKs have a dedicated keep_alive method
                self.dg_connection.keep_alive()
            else:
                # Fallback: try sending json as non-binary data
                self.dg_connection.send(keep_alive_msg)

            logger.info("Sent keep-alive message to Deepgram")
            return True
        except Exception as e:
            logger.error("Error sending keep-alive message", exc_info=e)
            return False

    async def _process_audio_impl(
        self, pcm_data: PcmData, user_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Tuple[bool, str, Dict[str, Any]]]]:
        """
        Process audio data through Deepgram for transcription.

        Args:
            pcm_data: The PCM audio data to process.
            user_metadata: Additional metadata about the user or session.

        Returns:
            None - Deepgram operates in asynchronous mode and emits events directly
            when transcripts arrive from the streaming service.
        """
        if self._is_closed:
            logger.warning("Deepgram connection is closed, ignoring audio")
            return None

        # Store the current user context for transcript events
        self._current_user = user_metadata

        # Check if the input sample rate matches the expected sample rate
        if pcm_data.sample_rate != self.sample_rate:
            logger.warning(
                "Input audio sample rate (%s Hz) does not match the expected sample rate (%s Hz). "
                "This may result in incorrect transcriptions. Consider resampling the audio.",
                pcm_data.sample_rate,
                self.sample_rate,
            )

        # Ensure connection is set up
        if not self.dg_connection and not self._setup_attempted:
            logger.warning("Deepgram connection not initialized, attempting setup")
            self._setup_connection()
            self._setup_attempted = True

        if not self.dg_connection:
            if not self._setup_attempted:
                logger.info("No Deepgram connection available, retrying setup")
                self._setup_connection()
                self._setup_attempted = True
            else:
                logger.error("No Deepgram connection available after retry")
                # Return an error result instead of emitting directly
                raise Exception("No Deepgram connection available")

        # Mark that we've attempted setup
        self._setup_attempted = True

        # Update the last activity time
        self.last_activity_time = time.time()

        # Convert PCM data to bytes if needed
        audio_data = pcm_data.samples
        if not isinstance(audio_data, bytes):
            # Convert numpy array to bytes
            audio_data = audio_data.astype(np.int16).tobytes()

        # Send the audio data to Deepgram
        try:
            logger.debug(
                "Sending audio data to Deepgram",
                extra={"audio_bytes": len(audio_data)},
            )
            self.dg_connection.send(audio_data)
        except Exception as e:
            # Raise exception to be handled by base class
            raise Exception(f"Deepgram audio transmission error: {e}")

        # Return None for asynchronous mode - events are emitted when they arrive
        return None

    async def close(self):
        """Close the Deepgram connection and clean up resources."""
        if self._is_closed:
            logger.debug("Deepgram STT service already closed")
            return

        logger.info("Closing Deepgram STT service")
        self._is_closed = True
        self._running = False

        # Cancel the keep-alive task if it exists
        if self.keep_alive_task:
            logger.debug("Cancelling keep-alive task")
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass
            self.keep_alive_task = None

        # Close the Deepgram connection if it exists
        if self.dg_connection:
            logger.debug("Closing Deepgram connection")
            try:
                self.dg_connection.finish()
                self.dg_connection = None
            except Exception as e:
                logger.error("Error closing Deepgram connection", exc_info=e)
