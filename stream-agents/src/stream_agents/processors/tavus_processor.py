import asyncio
import logging
import requests
from typing import Optional, Dict, Any
import numpy as np
from daily import CallClient, Daily, EventHandler
from aiortc import AudioStreamTrack, VideoStreamTrack
import av

from stream_agents.processors.base_processor import (
    AudioVideoProcessor,
    AudioPublisherMixin,
    VideoPublisherMixin,
)


class TavusDailyEventHandler(EventHandler):
    """Event handler for Daily call events."""

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.logger = logging.getLogger(__name__)

    def on_joined(self, data, error):
        """Handle call joined event."""
        self.logger.info(f"🔔 on_joined called - data: {data}, error: {error}")
        if error:
            self.logger.error(f"❌ Failed to join Daily call: {error}")
            return

        self.processor._call_joined = True
        self.logger.info(f"✅ Joined Daily call successfully: {data}")
        self.logger.info(f"📊 Call joined state updated: {self.processor._call_joined}")

    def on_left(self, data, error):
        """Handle call left event."""
        self.processor._call_joined = False
        self.logger.info(f"👋 Left Daily call: {data}")

    def on_participant_joined(self, data, error=None):
        """Handle participant joined event."""
        if not error:
            self.logger.info(f"👤 Participant joined: {data}")

    def on_participant_left(self, data, error=None):
        """Handle participant left event."""
        if not error:
            self.logger.info(f"👤 Participant left: {data}")

    def on_track_started(self, data, error):
        """Handle track started event."""
        self.logger.info(f"🔔 on_track_started called - data: {data}, error: {error}")
        if error:
            self.logger.error(f"❌ Track started error: {error}")
            return

        self.logger.info(f"🎬 Track started: {data}")

        # Handle audio tracks
        if data.get("track", {}).get("kind") == "audio":
            participant_id = data.get("participant", {}).get("id")
            self.logger.info(
                f"🎵 Audio track started from participant {participant_id}"
            )
            self.logger.info(f"🎵 Audio track details: {data.get('track')}")
            # Start forwarding audio frames
            asyncio.create_task(
                self.processor._forward_audio_from_daily(participant_id)
            )

        # Handle video tracks
        elif data.get("track", {}).get("kind") == "video":
            participant_id = data.get("participant", {}).get("id")
            self.logger.info(
                f"🎥 Video track started from participant {participant_id}"
            )
            self.logger.info(f"🎥 Video track details: {data.get('track')}")
            # Start forwarding video frames
            asyncio.create_task(
                self.processor._forward_video_from_daily(participant_id)
            )

    def on_track_stopped(self, data, error):
        """Handle track stopped event."""
        if not error:
            self.logger.info(f"🛑 Track stopped: {data}")

    def on_error(self, message: str) -> None:
        """Handle Daily call errors."""
        self.logger.error(f"❌ Daily call error: {message}")


class DailyAudioTrack(AudioStreamTrack):
    """Custom audio track that receives audio from Daily call."""

    def __init__(self):
        super().__init__()
        self.audio_queue: asyncio.Queue[av.AudioFrame] = asyncio.Queue(maxsize=50)
        self._stopped = False
        self.logger = logging.getLogger(__name__)

    async def add_audio_frame(self, frame):
        """Add an audio frame to the queue."""
        if self._stopped:
            return

        try:
            if not self.audio_queue.full():
                await self.audio_queue.put(frame)
        except Exception as e:
            self.logger.error(f"Error adding audio frame: {e}")

    async def recv(self):
        """Receive the next audio frame."""
        if self._stopped:
            raise Exception("Track stopped")

        try:
            frame = await asyncio.wait_for(self.audio_queue.get(), timeout=0.01)
            self.logger.debug(f"🎵 Retrieved audio frame from queue: {type(frame)}")
            return frame
        except asyncio.TimeoutError:
            # Return silence if no frame available - create proper silent audio frame
            import numpy as np

            # Create 20ms of silence at 48kHz (960 samples)
            samples = np.zeros(960, dtype=np.int16)
            frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
            frame.sample_rate = 48000
            self.logger.debug(
                f"🔇 Returning silent audio frame: {frame.sample_rate}Hz, {len(samples)} samples"
            )
            return frame
        except Exception as e:
            self.logger.debug(f"Audio frame error: {e}")
            # Return silence on error
            import numpy as np

            samples = np.zeros(960, dtype=np.int16)
            frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
            frame.sample_rate = 48000
            return frame

    def stop(self):
        """Stop the audio track."""
        self._stopped = True


class DailyVideoTrack(VideoStreamTrack):
    """Custom video track that receives video from Daily call."""

    def __init__(self):
        super().__init__()
        self.frame_queue: asyncio.Queue[av.VideoFrame] = asyncio.Queue(maxsize=10)
        self._stopped = False
        self.logger = logging.getLogger(__name__)
        # Default black frame
        self.last_frame = None

    async def add_video_frame(self, frame):
        """Add a video frame to the queue."""
        if self._stopped:
            return

        try:
            # Drop old frames if queue is full
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            await self.frame_queue.put(frame)
        except Exception as e:
            self.logger.error(f"Error adding video frame: {e}")

    async def recv(self):
        """Receive the next video frame."""
        if self._stopped:
            raise Exception("Track stopped")

        try:
            frame = await asyncio.wait_for(self.frame_queue.get(), timeout=0.1)
            self.last_frame = frame
            return frame
        except asyncio.TimeoutError:
            # Return last frame or black frame if no frame available
            if self.last_frame:
                return self.last_frame
            # Create a black frame as fallback
            pts, time_base = await self.next_timestamp()
            frame = av.VideoFrame.from_ndarray(
                np.zeros((480, 640, 3), dtype=np.uint8), format="bgr24"
            )
            frame.pts = pts
            frame.time_base = time_base
            return frame
        except Exception as e:
            self.logger.error(f"Error receiving video frame: {e}")
            pts, time_base = await self.next_timestamp()
            frame = av.VideoFrame.from_ndarray(
                np.zeros((480, 640, 3), dtype=np.uint8), format="bgr24"
            )
            frame.pts = pts
            frame.time_base = time_base
            return frame

    def stop(self):
        """Stop the video track."""
        self._stopped = True


class TavusClient:
    """
    Tavus API client wrapper for creating and managing conversations.

    This client provides methods to interact with the Tavus API for creating
    real-time video conversations with AI replicas.
    """

    def __init__(self, api_key: str, base_url: str = "https://tavusapi.com/v2"):
        """
        Initialize the Tavus API client.

        Args:
            api_key: Tavus API key for authentication
            base_url: Base URL for the Tavus API (defaults to v2 endpoint)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

        # Set up session with default headers
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "x-api-key": self.api_key}
        )

    def create_conversation(
        self,
        replica_id: str,
        persona_id: str,
        conversation_name: Optional[str] = None,
        callback_url: Optional[str] = None,
        audio_only: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a new conversation with the specified replica and persona.

        Args:
            replica_id: ID of the replica to use for the conversation
            persona_id: ID of the persona that defines behavior and capabilities
            conversation_name: Optional name for the conversation
            callback_url: Optional webhook URL for conversation updates
            audio_only: Whether to create an audio-only conversation
            **kwargs: Additional parameters to pass to the API

        Returns:
            Dict containing conversation details including conversation_id and conversation_url

        Raises:
            requests.RequestException: If the API request fails
            ValueError: If required parameters are missing
        """
        if not replica_id or not persona_id:
            raise ValueError("Both replica_id and persona_id are required")

        # Validate API key
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("API key is required and cannot be empty")

        # Log API key info (masked for security)
        masked_key = (
            f"{self.api_key[:8]}...{self.api_key[-4:]}"
            if len(self.api_key) > 12
            else "***"
        )
        self.logger.info(f"Using API key: {masked_key}")

        # Build request payload
        payload = {"replica_id": replica_id, "persona_id": persona_id}

        # Add optional parameters
        if conversation_name:
            payload["conversation_name"] = conversation_name
        if callback_url:
            payload["callback_url"] = callback_url
        if audio_only:
            payload["audio_only"] = str(audio_only)

        # Add any additional parameters
        payload.update(kwargs)

        try:
            self.logger.info(
                f"Creating Tavus conversation with replica_id={replica_id}, persona_id={persona_id}"
            )
            self.logger.info(f"Request URL: {self.base_url}/conversations")
            self.logger.info(f"Request payload: {payload}")
            self.logger.info(f"Request headers: {dict(self.session.headers)}")

            response = self.session.post(f"{self.base_url}/conversations", json=payload)

            self.logger.info(f"Response status code: {response.status_code}")
            self.logger.info(f"Response headers: {dict(response.headers)}")

            if response.status_code != 200:
                self.logger.error(
                    f"HTTP Error {response.status_code}: {response.reason}"
                )
                try:
                    error_body = response.json()
                    self.logger.error(f"Error response body: {error_body}")
                except Exception:
                    self.logger.error(f"Error response text: {response.text}")

            response.raise_for_status()

            conversation_data = response.json()
            self.logger.info(
                f"Successfully created conversation: {conversation_data.get('conversation_id')}"
            )
            self.logger.info(f"Full conversation response: {conversation_data}")

            return conversation_data

        except requests.RequestException as e:
            self.logger.error(f"Failed to create Tavus conversation: {e}")
            self.logger.error(f"Exception type: {type(e)}")

            if hasattr(e, "response") and e.response is not None:
                self.logger.error(f"Response status: {e.response.status_code}")
                self.logger.error(f"Response reason: {e.response.reason}")
                self.logger.error(f"Response URL: {e.response.url}")

                try:
                    error_detail = e.response.json()
                    self.logger.error(f"API error details (JSON): {error_detail}")
                except Exception as json_error:
                    self.logger.error(
                        f"Could not parse error response as JSON: {json_error}"
                    )
                    self.logger.error(f"Raw response text: {e.response.text}")
                    content_str = e.response.content.decode("utf-8", errors="replace")
                    self.logger.error(f"Response content: {content_str}")
            else:
                self.logger.error("No response object available in exception")

            raise

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get details of an existing conversation.

        Args:
            conversation_id: ID of the conversation to retrieve

        Returns:
            Dict containing conversation details
        """
        try:
            response = self.session.get(
                f"{self.base_url}/conversations/{conversation_id}"
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Failed to get conversation {conversation_id}: {e}")
            raise

    def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        End an active conversation.

        Args:
            conversation_id: ID of the conversation to end

        Returns:
            Dict containing the response from the API
        """
        try:
            response = self.session.post(
                f"{self.base_url}/conversations/{conversation_id}/end"
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Failed to end conversation {conversation_id}: {e}")
            raise


class TavusProcessor(AudioVideoProcessor, AudioPublisherMixin, VideoPublisherMixin):
    """
    Tavus processor that creates conversations on initialization.

    This processor integrates with the Tavus API to create real-time video
    conversations with AI replicas. It automatically creates a conversation
    when initialized and provides access to conversation details.
    """

    def __init__(
        self,
        api_key: str,
        replica_id: str,
        persona_id: str,
        conversation_name: Optional[str] = None,
        callback_url: Optional[str] = None,
        audio_only: bool = False,
        auto_create: bool = True,
        auto_join: bool = True,
        interval: int = 0,
        **kwargs,
    ):
        """
        Initialize the Tavus processor.

        Args:
            api_key: Tavus API key for authentication
            replica_id: ID of the replica to use for the conversation
            persona_id: ID of the persona that defines behavior and capabilities
            conversation_name: Optional name for the conversation
            callback_url: Optional webhook URL for conversation updates
            audio_only: Whether to create an audio-only conversation
            auto_create: Whether to automatically create the conversation on init
            auto_join: Whether to automatically join the Daily call after creating conversation
            interval: Processing interval in seconds
            **kwargs: Additional parameters to pass to the conversation creation
        """
        self.api_key = api_key
        self.replica_id = replica_id
        self.persona_id = persona_id
        self.conversation_name = conversation_name
        self.callback_url = callback_url
        self.audio_only = audio_only
        self.auto_join = auto_join
        self.logger = logging.getLogger(__name__)

        # Initialize the Tavus client
        self.client = TavusClient(api_key)

        # Conversation state
        self.conversation_data: Optional[Dict[str, Any]] = None
        self.conversation_id: Optional[str] = None
        self.conversation_url: Optional[str] = None

        # Initialize Daily call state before calling super().__init__
        self.daily_client: Optional[CallClient] = None
        self.daily_audio_track: Optional[DailyAudioTrack] = None
        self.daily_video_track: Optional[DailyVideoTrack] = None
        self._call_joined = False
        self.event_handler: Optional[TavusDailyEventHandler] = None

        # Initialize Daily
        Daily.init()

        # Call super().__init__ after all attributes are initialized
        super().__init__(
            interval=interval,
            receive_audio=True,
            receive_video=not audio_only,
            **kwargs,
        )

        # Automatically create conversation if requested
        if auto_create:
            try:
                self.create_conversation(**kwargs)
                # Auto-join Daily call if enabled
                if auto_join and self.conversation_url:
                    self.logger.info(
                        f"🔗 Auto-joining Daily call: {self.conversation_url}"
                    )
                    import pdb

                    pdb.set_trace()
                    asyncio.create_task(self._join_daily_call())
            except Exception as e:
                self.logger.error(f"Failed to auto-create conversation: {e}")
                raise

    def create_conversation(self, **kwargs) -> Dict[str, Any]:
        """
        Create a new conversation using the configured parameters.

        Args:
            **kwargs: Additional parameters to pass to the conversation creation

        Returns:
            Dict containing conversation details
        """
        try:
            self.conversation_data = self.client.create_conversation(
                replica_id=self.replica_id,
                persona_id=self.persona_id,
                conversation_name=self.conversation_name,
                callback_url=self.callback_url,
                audio_only=self.audio_only,
                **kwargs,
            )

            # Extract key information
            self.conversation_id = self.conversation_data.get("conversation_id")
            self.conversation_url = self.conversation_data.get("conversation_url")

            self.logger.info("Tavus conversation created successfully:")
            self.logger.info(f"  - ID: {self.conversation_id}")
            self.logger.info(f"  - URL: {self.conversation_url}")

            return self.conversation_data

        except Exception as e:
            self.logger.error(f"Failed to create Tavus conversation: {e}")
            raise

    def get_conversation_details(self) -> Optional[Dict[str, Any]]:
        """
        Get the current conversation details.

        Returns:
            Dict containing conversation details or None if no conversation exists
        """
        if self.conversation_id:
            try:
                return self.client.get_conversation(self.conversation_id)
            except Exception as e:
                self.logger.error(f"Failed to get conversation details: {e}")
                return None
        return None

    def end_conversation(self) -> Optional[Dict[str, Any]]:
        """
        End the current conversation.

        Returns:
            Dict containing the response from the API or None if no conversation exists
        """
        if self.conversation_id:
            try:
                result = self.client.end_conversation(self.conversation_id)
                self.logger.info(f"Ended Tavus conversation: {self.conversation_id}")
                return result
            except Exception as e:
                self.logger.error(f"Failed to end conversation: {e}")
                return None
        return None

    def create_audio_track(self) -> DailyAudioTrack:
        """Create an audio track for publishing Daily audio."""
        if not self.daily_audio_track:
            self.daily_audio_track = DailyAudioTrack()
            self.logger.info("🎵 Created Daily audio track for publishing")
        return self.daily_audio_track

    def create_video_track(self) -> DailyVideoTrack:
        """Create a video track for publishing Daily video."""
        if not self.daily_video_track:
            self.daily_video_track = DailyVideoTrack()
            self.logger.info("🎥 Created Daily video track for publishing")
            # Start generating test video immediately to verify the pipeline works
            asyncio.create_task(self._generate_test_video())
        return self.daily_video_track

    async def _join_daily_call(self):
        """Join the Daily call using the conversation URL."""
        if not self.conversation_url or self._call_joined:
            self.logger.info(
                f"⚠️  Skipping Daily join - URL: {self.conversation_url}, already joined: {self._call_joined}"
            )
            return

        try:
            self.logger.info("🔗 Starting Daily call join process...")
            self.logger.info(f"🔗 Conversation URL: {self.conversation_url}")

            # Create event handler
            self.logger.info("🔧 Creating Daily event handler...")
            self.event_handler = TavusDailyEventHandler(self)

            # Create Daily client with event handler
            self.logger.info("🔧 Creating Daily CallClient with event handler...")
            self.daily_client = CallClient(event_handler=self.event_handler)
            self.logger.info("✅ Daily CallClient created successfully")

            # Join the call with completion callback
            def join_completion(data, error):
                self.logger.info(
                    f"🔔 Join completion callback called - data: {data}, error: {error}"
                )
                if error:
                    self.logger.error(f"❌ Failed to join Daily call: {error}")
                else:
                    self.logger.info(f"✅ Join call completed successfully: {data}")
                    # Set joined state from completion callback since events might not fire
                    self._call_joined = True

            # Join the call (this is synchronous in the Daily Python API)
            self.logger.info("📞 Calling daily_client.join()...")
            self.logger.info(f"📞 Meeting URL: {self.conversation_url}")

            # The join method is synchronous but events are async
            self.daily_client.join(
                meeting_url=self.conversation_url, completion=join_completion
            )

            self.logger.info("📞 Daily call join method called successfully")

            # Wait longer for events to process and connection to establish
            for i in range(10):  # Wait up to 10 seconds
                await asyncio.sleep(1)
                self.logger.info(
                    f"📊 Waiting for connection... ({i + 1}/10) - _call_joined: {self._call_joined}"
                )
                if self._call_joined:
                    break

            if not self._call_joined:
                self.logger.warning(
                    "⚠️  Daily call join may not have completed successfully"
                )

            # Log current participants and start forwarding if we find the Tavus replica
            try:
                participants = (
                    self.daily_client.participants() if self.daily_client else {}
                )
                self.logger.info(f"📊 Current participants: {participants}")

                # Look for the Tavus replica participant
                tavus_participant_id = None
                for participant_id, participant_data in participants.items():
                    if participant_id != "local":  # Skip our local participant
                        user_info = participant_data.get("info", {})
                        if (
                            "tavus" in user_info.get("userId", "").lower()
                            or "replica" in user_info.get("userId", "").lower()
                        ):
                            tavus_participant_id = participant_id
                            self.logger.info(
                                f"🎭 Found Tavus participant: {participant_id}"
                            )
                            break

                # If we found a Tavus participant, start forwarding their media

                if tavus_participant_id and self._call_joined:
                    self.logger.info(
                        f"🚀 Starting media forwarding from Tavus participant: {tavus_participant_id}"
                    )
                    asyncio.create_task(
                        self._forward_video_from_daily(tavus_participant_id)
                    )
                    asyncio.create_task(
                        self._forward_audio_from_daily(tavus_participant_id)
                    )

            except Exception as e:
                self.logger.error(f"❌ Error getting participants: {e}")

        except Exception as e:
            self.logger.error(f"❌ Exception in _join_daily_call: {e}")
            import traceback

            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise

    async def _forward_audio_from_daily(self, participant_id: str):
        """Forward audio frames from Daily participant to our audio track."""
        self.logger.info(
            f"🎵 Starting audio forwarding from participant {participant_id}"
        )

        try:
            while self._call_joined and self.daily_audio_track:
                # Get participants and their tracks
                participants = (
                    self.daily_client.participants() if self.daily_client else {}
                )
                participant = participants.get(participant_id)

                if not participant:
                    self.logger.debug(f"Participant {participant_id} not found")
                    await asyncio.sleep(0.1)
                    continue

                # Get audio track from participant
                audio_track = participant.get("tracks", {}).get("audio")
                if audio_track:
                    # This is where we'd get frames - Daily Python API may need different approach
                    self.logger.debug(f"🎵 Found audio track for {participant_id}")
                    # For now, just log that we found it
                    await asyncio.sleep(0.1)
                else:
                    await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"❌ Error in audio forwarding: {e}")

    async def _forward_video_from_daily(self, participant_id: str):
        """Forward video frames from Daily participant to our video track."""
        self.logger.info(
            f"🎥 Starting video forwarding from participant {participant_id}"
        )

        try:
            while self._call_joined and self.daily_video_track:
                # Get participants and their tracks
                participants = (
                    self.daily_client.participants() if self.daily_client else {}
                )
                participant = participants.get(participant_id)

                if not participant:
                    self.logger.debug(f"Participant {participant_id} not found")
                    await asyncio.sleep(0.1)
                    continue

                # Get video track from participant
                video_track = participant.get("tracks", {}).get("video")
                if video_track:
                    self.logger.debug(f"🎥 Found video track for {participant_id}")
                    # This is where we'd get frames - Daily Python API may need different approach
                    # For now, create a test frame to see if video publishing works
                    test_frame = await self._create_test_video_frame()
                    if test_frame:
                        await self.daily_video_track.add_video_frame(test_frame)
                        self.logger.debug("🎥 Added test video frame to track")

                    await asyncio.sleep(1 / 30)  # 30 FPS
                else:
                    await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"❌ Error in video forwarding: {e}")

    async def _create_test_video_frame(self):
        """Create a test video frame to verify video publishing works."""
        try:
            import numpy as np

            # Create a simple test pattern (640x480 RGB)
            frame_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Add some pattern to make it visible
            frame_data[200:280, 280:360] = [255, 0, 0]  # Red square

            # Create av.VideoFrame
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            return frame
        except Exception as e:
            self.logger.error(f"❌ Error creating test frame: {e}")
            return None

    async def _generate_test_video(self):
        """Generate test video frames continuously to verify video publishing works."""
        self.logger.info("🎬 Starting test video generation...")
        frame_count = 0

        try:
            while self.daily_video_track and not self.daily_video_track._stopped:
                # Create test frame with frame counter
                test_frame = await self._create_test_video_frame_with_counter(
                    frame_count
                )
                if test_frame and self.daily_video_track:
                    await self.daily_video_track.add_video_frame(test_frame)
                    if frame_count % 30 == 0:  # Log every second
                        self.logger.info(
                            f"🎬 Generated test video frame #{frame_count}"
                        )

                frame_count += 1
                await asyncio.sleep(1 / 30)  # 30 FPS

        except Exception as e:
            self.logger.error(f"❌ Error in test video generation: {e}")

    async def _create_test_video_frame_with_counter(self, frame_count: int):
        """Create a test video frame with frame counter."""
        try:
            import numpy as np

            # Create a colorful test pattern (640x480 RGB)
            frame_data = np.zeros((480, 640, 3), dtype=np.uint8)

            # Create a gradient background
            for y in range(480):
                for x in range(640):
                    frame_data[y, x] = [
                        (x * 255) // 640,  # Red gradient
                        (y * 255) // 480,  # Green gradient
                        ((frame_count % 255) + 100) % 255,  # Blue animation
                    ]

            # Add a moving red square
            square_x = (frame_count * 2) % (640 - 100)
            square_y = 200
            frame_data[square_y : square_y + 80, square_x : square_x + 80] = [255, 0, 0]

            # Add frame counter text area
            frame_data[50:100, 50:200] = [255, 255, 255]  # White rectangle for text

            # Create av.VideoFrame
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            return frame
        except Exception as e:
            self.logger.error(f"❌ Error creating test frame with counter: {e}")
            return None

    async def leave_daily_call(self):
        """Leave the Daily call."""
        if self.daily_client and self._call_joined:
            try:
                # Leave the call with completion callback
                def leave_completion(error):
                    if error:
                        self.logger.error(f"❌ Error leaving Daily call: {error}")
                    else:
                        self.logger.info("👋 Left Daily call successfully")

                self.daily_client.leave(completion=leave_completion)
                self._call_joined = False

            except Exception as e:
                self.logger.error(f"❌ Error leaving Daily call: {e}")

        # Stop tracks
        if self.daily_audio_track:
            self.daily_audio_track.stop()
        if self.daily_video_track:
            self.daily_video_track.stop()

    def state(self) -> Dict[str, Any]:
        """
        Return the current state of the processor.

        Returns:
            Dict containing processor state information
        """
        return {
            "conversation_id": self.conversation_id,
            "conversation_url": self.conversation_url,
            "replica_id": self.replica_id,
            "persona_id": self.persona_id,
            "audio_only": self.audio_only,
            "daily_call_joined": self._call_joined,
            "has_audio_track": self.daily_audio_track is not None,
            "has_video_track": self.daily_video_track is not None,
            "status": "active" if self.conversation_id else "inactive",
        }

    def input(self) -> Dict[str, Any]:
        """
        Return input information for the processor.

        Returns:
            Dict containing input configuration
        """
        return {
            "replica_id": self.replica_id,
            "persona_id": self.persona_id,
            "conversation_name": self.conversation_name,
            "audio_only": self.audio_only,
        }

    async def cleanup(self):
        """Clean up resources including Daily call and tracks."""
        try:
            # Leave Daily call
            await self.leave_daily_call()

            # End Tavus conversation
            if self.conversation_id:
                self.end_conversation()

            self.logger.info("🧹 TavusProcessor cleaned up successfully")

        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, "_call_joined") and self._call_joined:
            try:
                # Note: This is synchronous cleanup for destructor
                # Proper async cleanup should be called explicitly
                if self.daily_audio_track:
                    self.daily_audio_track.stop()
                if self.daily_video_track:
                    self.daily_video_track.stop()
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.error(f"Error in destructor cleanup: {e}")
