"""Twilio Media Stream WebSocket handler."""

import base64
import json
import logging
from typing import Any, Protocol

from getstream.video import rtc
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType
from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig

from .audio import mulaw_to_pcm, pcm_to_mulaw, TWILIO_SAMPLE_RATE

logger = logging.getLogger(__name__)


class WebSocketProtocol(Protocol):
    """Protocol for WebSocket connections (compatible with FastAPI, Starlette, etc.)."""

    async def accept(self) -> None: ...
    async def receive_text(self) -> str: ...
    async def send_json(self, data: Any) -> None: ...


class TwilioMediaStream:
    """
    Manages a Twilio Media Stream WebSocket connection.

    Handles:
    - Audio track for incoming audio from the caller
    - WebSocket connection to Twilio
    - Parsing Twilio WebSocket messages (connected, start, media, stop)
    - Sending audio back to Twilio (for agent responses)

    Usage:
        stream = TwilioMediaStream(websocket)
        await stream.accept()

        # Access audio track for publishing to a call
        connection.add_tracks(audio=stream.audio_track)

        # Send audio back to Twilio
        await stream.send_audio(pcm_data)

        # Run until stream ends
        await stream.run()
    """

    def __init__(self, websocket: WebSocketProtocol):
        """
        Initialize a TwilioMediaStream.

        Args:
            websocket: A WebSocket connection (FastAPI, Starlette, etc.)
        """
        self.websocket = websocket
        self.stream_sid: str | None = None
        self.audio_track = AudioStreamTrack(
            sample_rate=TWILIO_SAMPLE_RATE,
            channels=1,
            format="s16",
        )
        self._connected = False

    async def accept(self) -> None:
        """Accept the WebSocket connection."""
        await self.websocket.accept()
        self._connected = True
        logger.info("TwilioMediaStream: WebSocket connection accepted")

    async def run(self) -> None:
        """
        Process incoming Twilio WebSocket messages.

        Parses messages and writes audio to the audio track.
        Returns when the stream ends (stop event or connection closed).
        """
        has_seen_media = False
        message_count = 0

        try:
            while True:
                message = await self.websocket.receive_text()
                data = json.loads(message)

                match data["event"]:
                    case "connected":
                        logger.info(f"TwilioMediaStream: Connected: {data}")
                    case "start":
                        self.stream_sid = data["streamSid"]
                        logger.info(
                            f"TwilioMediaStream: Stream started, streamSid={self.stream_sid}"
                        )
                    case "media":
                        # Decode base64 mulaw audio
                        payload = data["media"]["payload"]
                        mulaw_bytes = base64.b64decode(payload)

                        # Convert to PCM and write to audio track
                        pcm = mulaw_to_pcm(mulaw_bytes)
                        await self.audio_track.write(pcm)

                        if not has_seen_media:
                            logger.info(
                                f"TwilioMediaStream: Receiving audio: {len(mulaw_bytes)} bytes/chunk"
                            )
                            has_seen_media = True
                    case "stop":
                        logger.info("TwilioMediaStream: Stream stopped")
                        break

                message_count += 1
        except Exception as e:
            logger.info(f"TwilioMediaStream: WebSocket closed: {e}")
        finally:
            self._connected = False

        logger.info(
            f"TwilioMediaStream: Connection closed. Received {message_count} messages"
        )

    async def send_audio(self, pcm: PcmData) -> None:
        """
        Send PCM audio back to Twilio as mulaw.

        Args:
            pcm: PCM audio data to send (will be converted to mulaw).
        """
        if not self._connected or self.stream_sid is None:
            return

        # Convert PCM to mulaw and base64 encode
        mulaw_bytes = pcm_to_mulaw(pcm)
        payload = base64.b64encode(mulaw_bytes).decode("ascii")

        # Send media message per Twilio docs
        message = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": payload},
        }

        await self.websocket.send_json(message)

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket is still connected."""
        return self._connected


async def attach_phone_to_call(
    call, twilio_stream: TwilioMediaStream, user_id: str
) -> None:
    """
    Attach a phone user to a Stream call, bridging audio between Twilio and Stream.

    Args:
        call: The Stream call to attach to.
        twilio_stream: The TwilioMediaStream handling the Twilio WebSocket.
        user_id: The user ID for the phone participant.
    """
    subscription_config = SubscriptionConfig(
        default=TrackSubscriptionConfig(track_types=[TrackType.TRACK_TYPE_AUDIO])
    )

    connection = await rtc.join(call, user_id, subscription_config=subscription_config)

    @connection.on("audio")
    async def on_audio_received(pcm: PcmData):
        await twilio_stream.send_audio(pcm)

    await connection.__aenter__()
    await connection.add_tracks(audio=twilio_stream.audio_track, video=None)

    logger.info(f"Phone user {user_id} attached to call")
