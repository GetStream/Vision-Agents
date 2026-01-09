import json
import logging
import os
from collections import deque
from typing import Optional, cast

import aiohttp
import av
from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent, VideoLLM
from vision_agents.core.processors import Processor
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_utils import frame_to_jpeg_bytes

from . import events

logger = logging.getLogger(__name__)


PLUGIN_NAME = "nvidia_vlm"

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVCF_ASSET_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets"

SUPPORTED_FORMATS = {
    "png": {"mime": "image/png", "media": "img"},
    "jpg": {"mime": "image/jpg", "media": "img"},
    "jpeg": {"mime": "image/jpeg", "media": "img"},
}


class NvidiaVLM(VideoLLM):
    """
    NVIDIA VLM integration for vision language models.

    This plugin allows developers to interact with NVIDIA's vision models via
    the Chat Completions API. Supports models that accept both text and images.

    Features:
        - Video understanding: Automatically buffers and forwards video frames
        - Streaming responses with real-time chunk events
        - Configurable frame rate and buffer duration
        - Asset management: Automatically uploads and cleans up assets

    Examples:

        from vision_agents.plugins import nvidia
        vlm = nvidia.VLM(model="nvidia/cosmos-reason2-8b")

    """

    def __init__(
        self,
        model: str = "nvidia/cosmos-reason2-8b",
        api_key: Optional[str] = None,
        fps: int = 1,
        frame_buffer_seconds: int = 10,
        frame_width: int = 800,
        frame_height: int = 600,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.7,
        frames_per_second: int = 8,
        client: Optional[aiohttp.ClientSession] = None,
    ):
        """
        Initialize the NvidiaVLM class.

        Args:
            model: The NVIDIA model ID to use. Defaults to "nvidia/cosmos-reason2-8b".
            api_key: NVIDIA API token. Defaults to NVIDIA_API_KEY environment variable.
            fps: Number of video frames per second to handle.
            frame_buffer_seconds: Number of seconds to buffer for the model's input.
            frame_width: Width of the video frame to send. Default: 800.
            frame_height: Height of the video frame to send. Default: 600.
            max_tokens: Maximum response tokens. Default: 1024.
            temperature: Temperature for sampling. Default: 0.2.
            top_p: Top-p sampling parameter. Default: 0.7.
            frames_per_second: Frames per second for video models. Default: 8.
            client: Optional aiohttp.ClientSession for dependency injection.
        """
        super().__init__()
        self.model = model
        self.events.register_events_from_module(events)

        self._api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self._api_key:
            raise ValueError(
                "NVIDIA_API_KEY must be provided as argument or environment variable"
            )

        self._client: Optional[aiohttp.ClientSession] = None
        if client is not None:
            self._client = client
            self._own_client = False
        else:
            self._own_client = True

        self._fps = fps
        self._video_forwarder: Optional[VideoForwarder] = None
        self._frame_buffer: deque[av.VideoFrame] = deque(
            maxlen=fps * frame_buffer_seconds
        )
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._frames_per_second = frames_per_second
        self._current_asset_ids: list[str] = []

    async def _ensure_client(self) -> aiohttp.ClientSession:
        """Ensure the HTTP client is initialized."""
        if self._client is None:
            self._client = aiohttp.ClientSession()
        return self._client

    async def _upload_asset(self, frame_bytes: bytes) -> str:
        """
        Upload a frame as an asset to NVCF and return the asset ID.

        Args:
            frame_bytes: JPEG bytes of the frame to upload.

        Returns:
            Asset ID as a string.
        """
        client = await self._ensure_client()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        async with client.post(
            NVCF_ASSET_URL,
            headers=headers,
            json={"contentType": "image/jpeg", "description": "Video frame"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            authorize_res = await resp.json()

        upload_url = authorize_res["uploadUrl"]
        asset_id = authorize_res["assetId"]

        async with client.put(
            upload_url,
            data=frame_bytes,
            headers={
                "x-amz-meta-nvcf-asset-description": "Video frame",
                "content-type": "image/jpeg",
            },
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            resp.raise_for_status()
            if resp.status == 200:
                logger.debug(f"Uploaded asset {asset_id} successfully")
            else:
                logger.warning(f"Upload asset {asset_id} returned status {resp.status}")

        return asset_id

    async def _delete_asset(self, asset_id: str) -> None:
        """
        Delete an asset from NVCF.

        Args:
            asset_id: The asset ID to delete.
        """
        client = await self._ensure_client()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        url = f"{NVCF_ASSET_URL}/{asset_id}"

        async with client.delete(
            url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            resp.raise_for_status()
            logger.debug(f"Deleted asset {asset_id}")

    async def simple_response(
        self,
        text: str,
        processors: Optional[list[Processor]] = None,
        participant: Optional[Participant] = None,
    ) -> LLMResponseEvent:
        """
        Create an LLM response from text input with video context.

        This method is called when a new STT transcript is received.

        Args:
            text: The text to respond to.
            processors: List of processors with video/voice AI state.
            participant: The participant object. If not provided, uses "user" role.
        """
        if self._conversation is None:
            logger.warning(
                f'Cannot request a response from the LLM "{self.model}" - '
                "the conversation has not been initialized yet."
            )
            return LLMResponseEvent(original=None, text="")

        user_id = participant.user_id if participant else "user"
        await self._conversation.send_message(
            role="user", user_id=user_id, content=text
        )

        asset_ids: list[str] = []
        try:
            messages, asset_ids = await self._build_model_request()
            if asset_ids:
                self._current_asset_ids.clear()
            client = await self._ensure_client()
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }

            payload = {
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "top_p": self._top_p,
                "frames_per_second": self._frames_per_second,
                "messages": messages,
                "stream": True,
                "model": self.model,
            }

            if asset_ids:
                headers["NVCF-INPUT-ASSET-REFERENCES"] = ",".join(asset_ids)

            async with client.post(
                INVOKE_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                resp.raise_for_status()

                i = 0
                llm_response: LLMResponseEvent = LLMResponseEvent(
                    original=None, text=""
                )
                text_chunks: list[str] = []
                total_text = ""
                chunk_id = ""

                buffer = ""
                done = False
                async for chunk in resp.content.iter_chunked(1024):
                    if done:
                        break
                    buffer += chunk.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue

                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            done = True
                            break

                        chunk_data = json.loads(data_str)
                        chunk_id = chunk_data.get("id", chunk_id)

                        if not chunk_data.get("choices"):
                            continue

                        choice = chunk_data["choices"][0]
                        delta = choice.get("delta", {})
                        content = delta.get("content")
                        finish_reason = choice.get("finish_reason")

                        if content:
                            text_chunks.append(content)
                            self.events.send(
                                LLMResponseChunkEvent(
                                    plugin_name=PLUGIN_NAME,
                                    content_index=None,
                                    item_id=chunk_id,
                                    output_index=0,
                                    sequence_number=i,
                                    delta=content,
                                )
                            )

                        if finish_reason:
                            if finish_reason in ("length", "content_filter"):
                                logger.warning(
                                    f'The model finished the response due to reason "{finish_reason}"'
                                )
                            total_text = "".join(text_chunks)
                            self.events.send(
                                LLMResponseCompletedEvent(
                                    plugin_name=PLUGIN_NAME,
                                    original=chunk_data,
                                    text=total_text,
                                    item_id=chunk_id,
                                )
                            )

                        llm_response = LLMResponseEvent(
                            original=chunk_data, text=total_text
                        )
                        i += 1

                return llm_response

        except Exception as e:
            logger.exception(f'Failed to get a response from the model "{self.model}"')
            self.events.send(
                events.LLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    error_message=str(e),
                    event_data=e,
                )
            )
            return LLMResponseEvent(original=None, text="")
        finally:
            cleanup_ids = asset_ids if asset_ids else self._current_asset_ids
            for asset_id in cleanup_ids:
                await self._delete_asset(asset_id)
            self._current_asset_ids.clear()

    async def watch_video_track(
        self,
        track: MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """
        Setup video forwarding and start buffering video frames.

        Args:
            track: Instance of VideoStreamTrack.
            shared_forwarder: A shared VideoForwarder instance if present.
        """
        if self._video_forwarder is not None and shared_forwarder is None:
            logger.warning("Video forwarder already running, stopping the previous one")
            await self._video_forwarder.stop()
            self._video_forwarder = None
            logger.info("Stopped video forwarding")

        logger.info(f'🎥Subscribing plugin "{PLUGIN_NAME}" to VideoForwarder')
        if shared_forwarder:
            self._video_forwarder = shared_forwarder
        else:
            self._video_forwarder = VideoForwarder(
                cast(VideoStreamTrack, track),
                max_buffer=10,
                fps=self._fps,
                name=f"{PLUGIN_NAME}_forwarder",
            )
            self._video_forwarder.start()

        self._video_forwarder.add_frame_handler(
            self._frame_buffer.append, fps=self._fps
        )

    def _get_frames_bytes(self) -> list[bytes]:
        """Convert buffered video frames to JPEG bytes."""
        frames_bytes = []
        for frame in self._frame_buffer:
            frame_bytes = frame_to_jpeg_bytes(
                frame=frame,
                target_width=self._frame_width,
                target_height=self._frame_height,
                quality=85,
            )
            frames_bytes.append(frame_bytes)
        return frames_bytes

    async def _build_model_request(self) -> tuple[list[dict], list[str]]:
        """
        Build the model request with messages and uploaded frame assets.

        Returns:
            Tuple of (list of message dictionaries, list of asset IDs).
        """
        messages: list[dict] = []
        if self._instructions:
            messages.append(
                {
                    "role": "system",
                    "content": self._instructions,
                }
            )

        if self._conversation is not None:
            for message in self._conversation.messages:
                messages.append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )

        frames_bytes = self._get_frames_bytes()
        asset_ids: list[str] = []
        self._current_asset_ids = []
        media_content = ""

        if frames_bytes:
            logger.debug(f"Uploading {len(frames_bytes)} frames as assets")
            try:
                for frame_bytes in frames_bytes:
                    asset_id = await self._upload_asset(frame_bytes)
                    asset_ids.append(asset_id)
                    self._current_asset_ids.append(asset_id)
                    media_content += (
                        f'<img src="data:image/jpeg;asset_id,{asset_id}" />'
                    )
            except Exception:
                logger.warning(
                    f"Failed to upload all frames. {len(asset_ids)} assets were uploaded before failure."
                )
                raise

            if media_content:
                last_message = messages[-1] if messages else None
                if last_message and last_message.get("role") == "user":
                    last_message["content"] = (
                        f"{last_message['content']} {media_content}".strip()
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": media_content.strip(),
                        }
                    )

        return messages, asset_ids

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._own_client and self._client:
            await self._client.close()
            self._client = None
