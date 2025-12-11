import base64
import contextlib
import json
import time
from typing import Any, AsyncIterator, Optional, Self

import websockets
from getstream.video.rtc import PcmData


class Qwen3RealtimeClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
    ):
        self._base_url = f"{base_url}?model={model}"
        self._api_key = api_key
        self._real_ws: Optional[websockets.ClientConnection] = None
        self._exit_stack = contextlib.AsyncExitStack()

    async def connect(self, config: dict[str, Any]) -> Self:
        self._real_ws = await self._exit_stack.enter_async_context(
            websockets.connect(
                uri=self._base_url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
            )
        )
        # Initialize session with config params
        await self.update_session(config)
        return self

    async def close(self) -> None:
        await self._exit_stack.aclose()

    async def read(self) -> AsyncIterator[dict[str, Any]]:
        async for msg in self._ws:
            yield json.loads(msg)

    async def send_event(self, event: dict[str, Any]) -> None:
        event["event_id"] = f"event_{int(time.time() * 1000)}"
        await self._ws.send(json.dumps(event))

    async def update_session(self, config: dict[str, Any]) -> None:
        """Update the session configuration."""
        await self.send_event(event={"type": "session.update", "session": config})

    async def send_audio(self, pcm: PcmData) -> None:
        """Stream raw audio data to the API."""
        # Only 16-bit, 16 kHz, mono PCM is supported.
        audio_bytes = pcm.resample(
            target_sample_rate=16000, target_channels=1
        ).samples.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        append_event = {"type": "input_audio_buffer.append", "audio": audio_b64}
        await self.send_event(append_event)

    async def commit_audio(self) -> None:
        """Commit the audio buffer to trigger processing."""
        event = {"type": "input_audio_buffer.commit"}
        await self.send_event(event)

    async def send_frame(self, frame_bytes: bytes) -> None:
        """Append image data to the image buffer.
        Image data can come from local files or a real-time video stream.
        Note:
            - The image format must be JPG or JPEG. A resolution of 480p or 720p is recommended. The maximum supported resolution is 1080p.
            - A single image should not exceed 500 KB in size.
            - Encode the image data to Base64 before sending.
            - We recommend sending images to the server at a rate of no more than 2 frames per second.
            - You must send audio data at least once before sending image data.
        """
        image_b64 = base64.b64encode(frame_bytes).decode()
        event = {"type": "input_image_buffer.append", "image": image_b64}
        await self.send_event(event)

    async def cancel_response(self) -> None:
        """Cancel the current response."""
        event = {"type": "response.cancel"}
        await self.send_event(event)

    @property
    def _ws(self) -> websockets.ClientConnection:
        if self._real_ws is None:
            raise ValueError("The websocket connection is not established yet")
        return self._real_ws
