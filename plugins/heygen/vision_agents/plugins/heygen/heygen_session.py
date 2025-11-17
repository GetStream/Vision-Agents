import logging
from typing import Optional, Dict, Any
from os import getenv
import aiohttp

from .heygen_types import VideoQuality

logger = logging.getLogger(__name__)


class HeyGenSession:
    """Manages HeyGen API session lifecycle and configuration.

    Handles authentication, session creation, and API communication
    with HeyGen's Streaming API.
    """

    def __init__(
        self,
        avatar_id: str = "default",
        quality: VideoQuality = VideoQuality.HIGH,
        api_key: Optional[str] = None,
    ):
        """Initialize HeyGen session manager.

        Args:
            avatar_id: HeyGen avatar ID to use for streaming.
            quality: Video quality setting (VideoQuality.LOW, VideoQuality.MEDIUM, or VideoQuality.HIGH).
            api_key: HeyGen API key. Uses HEYGEN_API_KEY env var if not provided.
        """
        self.avatar_id = avatar_id
        self.quality = quality
        self.api_key: str = api_key or getenv("HEYGEN_API_KEY") or ""

        if not self.api_key:
            raise ValueError(
                "HeyGen API key required. Set HEYGEN_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.base_url = "https://api.heygen.com/v1"
        self.session_id: Optional[str] = None
        self.session_info: Optional[Dict[str, Any]] = None
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def create_session(self) -> Dict[str, Any]:
        """Create a new HeyGen streaming session.

        Returns:
            Session information including session_id, ICE servers, and SDP offer.
        """
        if not self._http_session:
            self._http_session = aiohttp.ClientSession()

        headers: dict[str, str] = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "avatar_id": self.avatar_id,
            "quality": self.quality,
        }

        try:
            async with self._http_session.post(
                f"{self.base_url}/streaming.new",
                json=payload,
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Failed to create HeyGen session: {response.status} - {error_text}"
                    )

                data = await response.json()
                self.session_info = data.get("data", {})
                self.session_id = self.session_info.get("session_id")

                logger.info(f"HeyGen session created: {self.session_id}")
                return self.session_info

        except Exception as e:
            logger.error(f"Failed to create HeyGen session: {e}")
            raise

    async def start_session(self, sdp_answer: Optional[str] = None) -> Dict[str, Any]:
        """Start the HeyGen streaming session.

        Args:
            sdp_answer: Optional SDP answer to include in the start request.

        Returns:
            Start confirmation with session details.
        """
        if not self.session_id:
            raise RuntimeError("Session not created. Call create_session() first.")

        if not self._http_session:
            self._http_session = aiohttp.ClientSession()

        headers: dict[str, str] = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "session_id": self.session_id,
        }

        # Include SDP answer if provided
        if sdp_answer:
            payload["sdp"] = {"type": "answer", "sdp": sdp_answer}

        try:
            async with self._http_session.post(
                f"{self.base_url}/streaming.start",
                json=payload,
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Failed to start HeyGen session: {response.status} - {error_text}"
                    )

                data = await response.json()
                logger.info(f"HeyGen session started: {self.session_id}")
                return data

        except Exception as e:
            logger.error(f"Failed to start HeyGen session: {e}")
            raise

    async def send_task(self, text: str, task_type: str = "repeat") -> Dict[str, Any]:
        """Send a text task to HeyGen for the avatar to speak.

        This is the proper way to achieve lip-sync with HeyGen - send text,
        and HeyGen handles TTS and lip-sync server-side.

        Args:
            text: The text for the avatar to speak.
            task_type: Either "repeat" (avatar repeats text exactly) or
                      "talk" (processes through HeyGen's LLM first).

        Returns:
            Task response from HeyGen.
        """
        if not self.session_id:
            raise RuntimeError("Session not created. Call create_session() first.")

        if not self._http_session:
            self._http_session = aiohttp.ClientSession()

        headers: dict[str, str] = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "session_id": self.session_id,
            "text": text,
            "task_type": task_type,
        }

        try:
            async with self._http_session.post(
                f"{self.base_url}/streaming.task",
                json=payload,
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(
                        f"Failed to send task to HeyGen: {response.status} - {error_text}"
                    )
                    return {}

                data = await response.json()
                logger.debug(f"Sent text to HeyGen: '{text[:50]}...'")
                return data

        except Exception as e:
            logger.error(f"Error sending task to HeyGen: {e}")
            return {}

    async def stop_session(self) -> None:
        """Stop the HeyGen streaming session."""
        if not self.session_id:
            logger.warning("No active session to stop")
            return

        if not self._http_session:
            return

        headers: dict[str, str] = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "session_id": self.session_id,
        }

        try:
            async with self._http_session.post(
                f"{self.base_url}/streaming.stop",
                json=payload,
                headers=headers,
            ) as response:
                if response.status == 200:
                    logger.info(f"HeyGen session stopped: {self.session_id}")
                else:
                    logger.warning(f"Failed to stop HeyGen session: {response.status}")
        except Exception as e:
            logger.error(f"Error stopping HeyGen session: {e}")

    async def close(self) -> None:
        """Clean up session resources."""
        await self.stop_session()

        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        self.session_id = None
        self.session_info = None
        logger.info("HeyGen session cleaned up")
