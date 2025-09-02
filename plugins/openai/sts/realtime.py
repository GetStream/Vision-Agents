"""OpenAI Realtime Speech-to-Speech implementation.

This module provides a wrapper around OpenAI's Realtime API that implements the STS
interface. It enables real-time, full-duplex speech-to-speech communication with
OpenAI's models through Stream's video call infrastructure.

Key Features:
- Real-time audio streaming and processing
- Full-duplex communication with OpenAI models
- Event-driven architecture for handling all OpenAI/Stream events
- Support for custom instructions and voice selection
- Session management and updates

Example
-------

```python
from getstream.stream import Stream
from getstream.plugins.sts.openai_realtime import OpenAIRealtime

# Initialize Stream client and create/join a call
client = Stream.from_env()
call = client.video.call("default", "demo-call")
call.get_or_create()

# Create and configure the STS bot
sts_bot = OpenAIRealtime(
    model="gpt-4o-realtime-preview",
    voice="alloy",  # Optional: specify voice
    instructions="You are a helpful assistant."  # Optional: system instructions
)

# Connect to the call and handle events
async with await sts_bot.connect(call, agent_user_id="assistant") as connection:
    # Optionally update session parameters
    await sts_bot.update_session(voice="nova")

    # Send a text message if needed
    await sts_bot.send_user_message("Hello, how are you?")

    # Listen for events
    async for event in connection:
        print(f"Event: {event.type}")
        print(f"Data: {event}")
```

The class handles all the complexity of:
1. Establishing and managing the WebSocket connection
2. Applying session configurations (instructions, voice, etc.)
3. Forwarding all events from OpenAI/Stream
4. Managing the connection lifecycle
5. Providing helper methods for common operations
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from getstream.plugins.common import STS
from getstream.video.call import Call
from getstream.video.openai import ConnectionManagerWrapper

logger = logging.getLogger(__name__)


class OpenAIRealtime(STS):
    """Speech-to-Speech wrapper for the OpenAI Realtime API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-realtime-preview",
        voice: Optional[str] = None,
        instructions: Optional[str] = None,
    ):
        """Create a new wrapper.

        Args:
            api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
            model: Model ID to use when connecting.
            voice: Optional voice selection passed to the session.
            instructions: Optional system instructions passed to the session.
        """

        super().__init__()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided and not found in env")

        self.model = model
        self.voice = voice
        self.instructions = instructions
        self._connection: Optional[ConnectionManagerWrapper] = None

    async def connect(
        self,
        call: Call,
        agent_user_id: str = "assistant",
        extra_session: Optional[Dict[str, Any]] = None,
    ) -> ConnectionManagerWrapper:
        """Connect an AI agent to *call*.

        Emits:
            • ``connected`` – once the websocket handshake succeeded
            • every event coming from OpenAI/Stream, verbatim
            • ``disconnected`` – after normal closure or error
        """

        if self._is_connected:
            raise RuntimeError("AI agent already connected")

        # Build session presets
        session_payload: Dict[str, Any] = {}
        if self.instructions:
            session_payload["instructions"] = self.instructions
        if self.voice:
            session_payload["voice"] = self.voice
        if extra_session:
            session_payload.update(extra_session)

        # Open connection manager through the SDK helper
        logger.info(
            "Connecting OpenAI agent to call %s/%s using model %s",
            call.call_type,
            call.id,
            self.model,
        )

        self._connection = call.connect_openai(
            self.api_key,
            agent_user_id,
            model=self.model,
        )
        self._is_connected = True
        # Use the parent class's structured event emission
        self._emit_connected_event()

        logger.info(
            f"Connected OpenAI agent to call {call.call_type}/{call.id} using model {self.model}"
        )
        return self._connection

    async def update_session(self, **session_fields):
        """Wrapper around ``connection.session.update()``."""
        if not self._is_connected or not self._connection:
            raise RuntimeError("Not connected")

        await self._connection.session.update(session=session_fields)

    async def send_function_call_output(self, tool_call_id: str, output: str):
        """Send a tool call output to the conversation."""
        if not self._is_connected or not self._connection:
            raise RuntimeError("Not connected")

        await self._connection.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": output,
            }
        )

    async def send_user_message(self, text: str):
        """Send a text message from the *human* side to the conversation."""
        if not self._is_connected or not self._connection:
            raise RuntimeError("Not connected")

        await self._connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text,
                    }
                ],
            }
        )
        await self.request_assistant_response()

    async def request_assistant_response(self):
        """Ask OpenAI to generate the next assistant turn."""
        if not self._is_connected or not self._connection:
            raise RuntimeError("Not connected")

        await self._connection.response.create()


# Re-export for ``from getstream.plugins.sts.openai_realtime import OpenAIRealtime``
__all__ = ["OpenAIRealtime"]
