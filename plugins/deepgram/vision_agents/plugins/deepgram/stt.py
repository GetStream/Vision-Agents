import json
import logging
from typing import Dict, Any, Optional, Tuple, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from vision_agents.core.edge.types import Participant
import numpy as np
import os
import time

from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions, DeepgramClientOptions
from vision_agents.core import stt
from getstream.video.rtc.track_util import PcmData

logger = logging.getLogger(__name__)


class STT(stt.STT):
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
        interim_results: bool = True,
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
            interim_results: Whether to emit interim results (partial transcripts with the partial_transcript event).
        """
        super().__init__(sample_rate=sample_rate)

        # If no API key was provided, check for DEEPGRAM_API_KEY in environment
        if api_key is None:
            api_key = os.environ.get("DEEPGRAM_API_KEY")
            if not api_key:
                logger.warning(
                    "No API key provided and DEEPGRAM_API_KEY environment variable not found."
                )

        # Initialize DeepgramClient with the API key
        logger.info("Initializing Deepgram client")
        config = DeepgramClientOptions(
            options={"keepalive": "true"}  # Comment this out to see the effect of not using keepalive
        )
        self.deepgram = client if client is not None else DeepgramClient(api_key, config)
        self.dg_connection: Optional[Any] = None
        self.options = options or LiveOptions(
            model="nova-2",
            language=language,
            encoding="linear16",
            sample_rate=sample_rate,
            channels=1,
            interim_results=interim_results,
        )

        # Track current user context for associating transcripts with users
        self._current_user: Optional[Dict[str, Any]] = None

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
            assert self.dg_connection is not None

            # Handler for transcript results
            def handle_transcript(conn, result=None):
                try:
                    # Update the last activity time
                    self.last_activity_time = time.time()

                    # Check if result is already a dict (from LiveResultResponse or test mocks)
                    if isinstance(result, dict):
                        transcript = result
                    elif hasattr(result, "to_dict"):
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

        except Exception as e:
            # Log the error and set connection to None
            logger.error("Error setting up Deepgram connection", exc_info=e)
            self.dg_connection = None
            # Emit error immediately
            self._emit_error_event(e, "Deepgram connection setup")

    async def _process_audio_impl(
        self, pcm_data: PcmData, user_metadata: Optional[Union[Dict[str, Any], "Participant"]] = None
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
        self._current_user = user_metadata  # type: ignore[assignment]

        # Check if the input sample rate matches the expected sample rate
        if pcm_data.sample_rate != self.sample_rate:
            logger.warning(
                "Input audio sample rate (%s Hz) does not match the expected sample rate (%s Hz). "
                "This may result in incorrect transcriptions. Consider resampling the audio.",
                pcm_data.sample_rate,
                self.sample_rate,
            )

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
            assert self.dg_connection is not None
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

        # Close the Deepgram connection if it exists
        if self.dg_connection:
            logger.debug("Closing Deepgram connection")
            try:
                self.dg_connection.finish()
                self.dg_connection = None
            except Exception as e:
                logger.error("Error closing Deepgram connection", exc_info=e)
