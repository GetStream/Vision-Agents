import asyncio
import logging
import os
from typing import Any, Optional

from deepgram import AsyncDeepgramClient
from deepgram.core import EventType
from deepgram.listen.v2.socket_client import AsyncV2SocketClient
from getstream.video.rtc.track_util import PcmData
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import TranscriptResponse
from vision_agents.core.utils.utils import cancel_and_wait

# Handle API changes between deepgram-sdk 5.3.0 and 5.3.1
try:
    # deepgram-sdk >= 5.3.1
    from deepgram.listen.v2.types import ListenV2CloseStream

    _USE_NEW_API = True
except ImportError:
    # deepgram-sdk 5.3.0
    from deepgram.extensions.types.sockets import ListenV2ControlMessage

    _USE_NEW_API = False

logger = logging.getLogger(__name__)


class STT(stt.STT):
    """
    Deepgram Speech-to-Text implementation using Flux model.

    - https://developers.deepgram.com/docs/flux/quickstart
    - https://github.com/deepgram/deepgram-python-sdk/blob/main/examples/listen/v2/connect/async.py
    - https://github.com/deepgram/deepgram-python-sdk/tree/main
    - https://github.com/deepgram-devs/deepgram-demos-flux-streaming-transcription/blob/main/main.py

    Deepgram flux runs turn detection internally. So running turn detection in front of this is optional/not needed

    - eot_threshold controls turn end sensitivity
    - eager_eot_threshold controls eager turn ending (so you can already prepare the LLM response)
    """

    turn_detection: bool = True  # we support turn detection with deepgram

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "flux-general-en",
        language: Optional[str] = None,
        eager_turn_detection: bool = False,
        eot_threshold: Optional[float] = None,
        eager_eot_threshold: Optional[float] = None,
        client: Optional[AsyncDeepgramClient] = None,
    ):
        """
        Initialize Deepgram STT.

        Args:
            api_key: Deepgram API key. If not provided, will use DEEPGRAM_API_KEY env var.
            model: Model to use for transcription. Defaults to "flux-general-en".
            language: Language code (e.g., "en", "es"). If not provided, auto-detection is used.
            eot_threshold: End-of-turn threshold for determining when a turn is complete.
            eager_eot_threshold: Eager end-of-turn threshold for faster turn detection.
            client: Optional pre-configured AsyncDeepgramClient instance.
        """
        super().__init__(provider_name="deepgram")

        if not api_key:
            api_key = os.environ.get("DEEPGRAM_API_KEY")

        if client is not None:
            self.client = client
        else:
            # Initialize AsyncDeepgramClient with api_key as named parameter
            if api_key:
                self.client = AsyncDeepgramClient(api_key=api_key)
            else:
                self.client = AsyncDeepgramClient()

        self.model = model
        self.language = language
        self.eot_threshold = eot_threshold
        self.eager_turn_detection = eager_turn_detection
        if self.eager_turn_detection and eager_eot_threshold is None:
            eager_eot_threshold = 0.5
        self.eager_eot_threshold = eager_eot_threshold
        self._current_participant: Optional[Participant] = None
        self.connection: Optional[AsyncV2SocketClient] = None
        self._connection_ready = asyncio.Event()
        self._connection_context: Optional[Any] = None
        self._listen_task: Optional[asyncio.Task[Any]] = None

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        """
        Process audio data through Deepgram for transcription.

        This method sends audio to the existing WebSocket connection. The connection
        is started automatically on first use. Audio is automatically resampled to 16kHz.

        Args:
            pcm_data: The PCM audio data to process.
            participant: Optional participant metadata (currently not used in streaming mode).
        """
        if self.closed:
            logger.warning("Deepgram STT is closed, ignoring audio")
            return

        # Wait for connection to be ready
        await self._connection_ready.wait()

        # Double-check connection is still ready (could have closed while waiting)
        if not self._connection_ready.is_set():
            logger.warning("Deepgram connection closed while processing audio")
            return

        # Resample to 16kHz mono (recommended by Deepgram)
        resampled_pcm = pcm_data.resample(16_000, 1)

        # Convert int16 samples to bytes
        audio_bytes = resampled_pcm.samples.tobytes()

        self._current_participant = participant

        if self.connection is not None:
            await self.connection.send_media(audio_bytes)

    async def start(self):
        """
        Start the Deepgram WebSocket connection and begin listening for transcripts.
        """
        if self.connection is not None:
            logger.warning("Deepgram connection already started")
            return

        # Build connection parameters
        connect_params = {
            "model": self.model,
            "encoding": "linear16",
            "sample_rate": "16000",
        }

        # Add optional parameters if specified
        if self.eot_threshold is not None:
            connect_params["eot_threshold"] = str(self.eot_threshold)
        if self.eager_eot_threshold is not None:
            connect_params["eager_eot_threshold"] = str(self.eager_eot_threshold)

        # Connect to Deepgram v2 listen WebSocket with timeout
        self._connection_context = self.client.listen.v2.connect(**connect_params)

        # Add timeout for connection establishment
        self.connection = await asyncio.wait_for(
            self._connection_context.__aenter__(), timeout=10.0
        )

        # Register event handlers
        if self.connection is not None:
            self.connection.on(EventType.OPEN, self._on_open)
            self.connection.on(EventType.MESSAGE, self._on_message)
            self.connection.on(EventType.ERROR, self._on_error)
            self.connection.on(EventType.CLOSE, self._on_close)

            # Start listening for events
            self._listen_task = asyncio.create_task(self.connection.start_listening())

        # Mark connection as ready
        self._connection_ready.set()

    def _on_message(self, message):
        """
        Event handler for messages from Deepgram.

        Args:
            message: The message object from Deepgram

        TODO: errors in this function are hidden silently. Not sure why this happens.
        """
        # Extract message data
        if not hasattr(message, "type"):
            logger.warning(f"Received message without 'type' attribute: {message}")
            return

        # Handle TurnInfo messages (v2 API)
        if message.type == "TurnInfo":
            # Extract transcript text
            transcript_text = getattr(message, "transcript", "").strip()

            if not transcript_text:
                return

            # Get event type to determine if final or partial
            # "StartOfTurn" and "Update" = partial, "EndOfTurn" = final
            event = getattr(message, "event", "")

            is_final = event == "EndOfTurn"
            eager_end_of_turn = event == "EagerEndOfTurn"
            start_of_turn = event == "StartOfTurn"

            # Get end of turn confidence
            end_of_turn_confidence = getattr(message, "end_of_turn_confidence", 0.0)

            # Calculate average confidence from words
            words = getattr(message, "words", [])
            if words:
                confidences = [w.confidence for w in words if hasattr(w, "confidence")]
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0.0
                )
            else:
                avg_confidence = 0.0

            # Get audio duration
            audio_window_end = getattr(message, "audio_window_end", 0.0)
            duration_ms = int(audio_window_end * 1000)

            # Build response metadata
            response_metadata = TranscriptResponse(
                confidence=avg_confidence,
                language=self.language or "auto",
                audio_duration_ms=duration_ms,
                model_name=self.model,
            )

            # Use the participant from the most recent process_audio call
            participant = self._current_participant

            if participant is None:
                logger.warning("Received transcript but no participant set")
                return

            # broadcast STT event first
            if is_final:
                self._emit_transcript_event(
                    transcript_text, participant, response_metadata
                )
            else:
                self._emit_partial_transcript_event(
                    transcript_text, participant, response_metadata
                )

            # broadcast turn event
            if is_final or eager_end_of_turn:
                self._emit_turn_ended_event(
                    participant=participant,
                    eager_end_of_turn=eager_end_of_turn,
                    confidence=end_of_turn_confidence,
                )

            if start_of_turn:
                self._emit_turn_started_event(
                    participant=participant, confidence=end_of_turn_confidence
                )

    def _on_open(self, message):
        pass

    def _on_error(self, error):
        """
        Event handler for errors from Deepgram.

        Args:
            error: The error from Deepgram
        """
        logger.error(f"Deepgram WebSocket error: {error}")
        raise Exception(f"Deepgram WebSocket error {error}")

    def _on_close(self, error):
        """
        Event handler for connection close.
        """
        logger.debug(f"Deepgram WebSocket connection closed: {error}")
        self._connection_ready.clear()

    async def close(self):
        """
        Close the Deepgram connection and clean up resources.
        """
        # Mark as closed first
        await super().close()

        # Cancel the listen task and ensure it's finished
        if self._listen_task:
            await cancel_and_wait(self._listen_task)

        # Close connection
        if self.connection and self._connection_context:
            try:
                # Handle API differences between deepgram-sdk versions
                if _USE_NEW_API:
                    close_msg = ListenV2CloseStream(type="CloseStream")
                    await self.connection.send_close_stream(close_msg)
                else:
                    close_msg = ListenV2ControlMessage(type="CloseStream")
                    await self.connection.send_control(close_msg)
                await self._connection_context.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning(f"Error closing Deepgram websocket connection: {exc}")
            finally:
                self.connection = None
                self._connection_context = None
                self._connection_ready.clear()
