import asyncio
import contextlib
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import websockets
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import (
    ListenV1ControlMessage,
    ListenV1MetadataEvent,
    ListenV1ResultsEvent,
    ListenV1SpeechStartedEvent,
    ListenV1UtteranceEndEvent,
)
from deepgram.listen.v1.socket_client import AsyncV1SocketClient
from getstream.video.rtc.track_util import PcmData

from vision_agents.core import stt

from .utils import generate_silence

if TYPE_CHECKING:
    from vision_agents.core.edge.types import Participant

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
        options: Optional[dict] = None,
        sample_rate: int = 48000,
        language: str = "en-US",
        interim_results: bool = True,
        client: Optional[AsyncDeepgramClient] = None,
        keep_alive_interval: float = 1.0,
        connection_timeout: float = 15.0,
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
            connection_timeout: Time to wait for the Deepgram connection to be established.

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
        self.deepgram = (
            client if client is not None else AsyncDeepgramClient(api_key=api_key)
        )
        self.dg_connection: Optional[AsyncV1SocketClient] = None

        self.options = options or {
            "model": "nova-2",
            "language": language,
            "encoding": "linear16",
            "sample_rate": sample_rate,
            "channels": 1,
            "interim_results": interim_results,
        }

        # Track current user context for associating transcripts with users
        self._current_user: Optional[Dict[str, Any]] = None

        # Generate a silence audio to use as keep-alive message
        self._keep_alive_data = generate_silence(
            sample_rate=sample_rate, duration_ms=10
        )
        self._keep_alive_interval = keep_alive_interval

        self._stack = contextlib.AsyncExitStack()
        # An event to detect that the connection was established once.
        self._connected_once = asyncio.Event()
        # Time to wait for connection to be established before sending the event
        self._connection_timeout = connection_timeout
        self._last_sent_at = float("-inf")
        # Lock to prevent concurrent connection opening
        self._connect_lock = asyncio.Lock()

        # Start the listener loop in the background
        asyncio.create_task(self.start())

    async def start(self):
        """
        Start the main task establishing the Deepgram connection and processing the events.
        """
        if self._is_closed:
            logger.warning("Cannot setup connection - Deepgram instance is closed")
            return None

        # Establish a Deepgram connection.
        # Use a lock to make sure it's established only once
        async with self._connect_lock:
            if self.dg_connection is not None:
                logger.debug("Connection already set up, skipping initialization")
                return None

            try:
                logger.info("Creating a Deepgram connection with options %s", self.options)
                dg_connection = await self._stack.enter_async_context(
                    self.deepgram.listen.v1.connect(**self.options)
                )
            except Exception as e:
                # Log the error and set connection to None
                logger.exception("Error setting up Deepgram connection")
                self.dg_connection = None
                # Emit error immediately
                self._emit_error_event(e, "Deepgram connection setup")
                raise
            finally:
                self._connected_once.set()

        self.dg_connection = dg_connection
        # Start the keep-alive loop to keep the connection open
        asyncio.create_task(self._keepalive_loop())

        # Register event handlers
        self.dg_connection.on(
            EventType.OPEN,
            lambda msg: logger.debug(f"Deepgram connection opened. message={msg}"),
        )
        self.dg_connection.on(EventType.CLOSE, self._on_connection_close)
        self.dg_connection.on(EventType.ERROR, self._on_connection_error)
        self.dg_connection.on(EventType.MESSAGE, self._on_message)

        # Start processing the events from Deepgram.
        # This is a blocking call.
        logger.debug("Listening to the events from a Deepgram connection")
        await self.dg_connection.start_listening()
        return None

    async def started(self):
        """
        Wait until the Deepgram connection is established.
        """
        if self._connected_once.is_set():
            return

        await asyncio.wait_for(
            self._connected_once.wait(), timeout=self._connection_timeout
        )

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
                await self.dg_connection.send_control(
                    ListenV1ControlMessage(type="CloseStream")
                )
                await self._stack.aclose()
                self.dg_connection = None
            except Exception:
                logger.exception("Error closing Deepgram connection")

    async def _on_message(
        self,
        message: ListenV1ResultsEvent
        | ListenV1MetadataEvent
        | ListenV1UtteranceEndEvent
        | ListenV1SpeechStartedEvent,
    ):
        if message.type != "Results":
            logger.debug(
                "Received non-transcript message, skip processing. message=%s", message
            )
            return

        transcript = message.dict()

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

        # Emit immediately for real-time responsiveness
        if is_final:
            self._emit_transcript_event(transcript_text, self._current_user, metadata)
        else:
            self._emit_partial_transcript_event(
                transcript_text, self._current_user, metadata
            )

        logger.debug(
            "Received transcript",
            extra={
                "is_final": is_final,
                "text_length": len(transcript_text),
                "confidence": metadata["confidence"],
            },
        )

    async def _on_connection_error(self, error: websockets.WebSocketException):
        error_text = str(error) if error is not None else "Unknown error"
        logger.error("Deepgram error received: %s", error_text)
        # Emit error immediately
        error_obj = Exception(f"Deepgram error: {error_text}")
        self._emit_error_event(error_obj, "Deepgram connection")

    async def _on_connection_close(self, message: Any):
        logger.warning(f"Deepgram connection closed. message={message}")
        await self.close()

    async def _process_audio_impl(
        self,
        pcm_data: PcmData,
        user_metadata: Optional[Union[Dict[str, Any], "Participant"]] = None,
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

        # Convert PCM data to bytes if needed
        audio_data = pcm_data.samples
        if not isinstance(audio_data, bytes):
            # Convert numpy array to bytes
            audio_data = audio_data.astype(np.int16).tobytes()

        # Wait for the attempt to establish the connection
        try:
            await self.started()
        except asyncio.TimeoutError:
            logger.error(
                f"Deepgram connection is not established within {self._connection_timeout} seconds. "
                f"Skipping the audio package."
            )
            return None

        # Send the audio data to Deepgram
        logger.debug(
            "Sending audio data to Deepgram",
            extra={"audio_bytes": len(audio_data)},
        )
        await self._send_audio(audio_data)
        return None

    async def _send_audio(self, data: bytes):
        if self.dg_connection is None:
            logger.warning("Deepgram connection is not established")
            return

        try:
            await self.dg_connection.send_media(data)
            self._last_sent_at = time.time()
        except Exception as e:
            # Raise exception to be handled by base class
            raise Exception(f"Deepgram audio transmission error: {e}") from e

    async def _keepalive_loop(self):
        """
        Send the silence audio every `interval` seconds
        to prevent Deepgram from closing the connection.
        """
        while not self._is_closed and self.dg_connection is not None:
            if self._last_sent_at + self._keep_alive_interval <= time.time():
                logger.debug("Sending keepalive packet to Deepgram...")
                # Send audio silence to keep the connection open
                await self._send_audio(self._keep_alive_data)
                # Send keep-alive message as well
                await self.dg_connection.send_control(
                    ListenV1ControlMessage(type="KeepAlive")
                )

            # Sleep max for 1s to avoid missing the keep-alive schedule
            timeout = min(self._keep_alive_interval, 1.0)
            await asyncio.sleep(timeout)
