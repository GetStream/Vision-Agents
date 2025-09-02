"""
OpenAI Realtime STS Model Implementation

This module provides an OpenAI Realtime API implementation that can be used
with the Agent class for Speech-to-Speech functionality.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from getstream.plugins.openai.sts import OpenAIRealtime


class OpenAIRealtimeModel:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-realtime-preview",
        voice: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        turn_detection: Optional[Dict[str, Any]] = None,
        input_audio_transcription: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI Realtime model.

        Args:
            api_key: OpenAI API key (if not using OPENAI_API_KEY env var)
            model: Model name (default: "gpt-4o-realtime-preview")
            voice: Voice to use (e.g., "alloy", "echo", "fable", "onyx", "nova", "shimmer")
            instructions: System instructions for the assistant
            tools: List of tools/functions the assistant can call
            turn_detection: Turn detection configuration
            input_audio_transcription: Input audio transcription configuration (e.g., {"model": "whisper-1", "language": "en"})
            **kwargs: Additional configuration options
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided and not found in environment")

        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.tools = tools or []
        self.turn_detection = turn_detection
        self.input_audio_transcription = input_audio_transcription
        self.kwargs = kwargs

        # Mark this as an STS model
        self.sts = True

        self.logger = logging.getLogger(f"OpenAIRealtimeModel[{model}]")

        # Create the underlying OpenAI Realtime instance
        self._realtime = OpenAIRealtime(
            api_key=self.api_key,
            model=self.model,
            voice=self.voice,
            instructions=self.instructions,
        )

        self.logger.info(f"Initialized OpenAI Realtime model: {model}")

    async def connect(self, call, agent_user_id: str = "assistant"):
        """
        Connect to a Stream video call.

        Args:
            call: Stream video call object
            agent_user_id: User ID for the agent

        Returns:
            Connection object that can be used to iterate over events
        """
        self.logger.info(f"Connecting OpenAI Realtime to call {call.id}")

        # Build extra session configuration
        extra_session = {}
        if self.tools:
            extra_session["tools"] = self.tools
        if self.turn_detection:
            extra_session["turn_detection"] = [self.turn_detection]
        if self.input_audio_transcription:
            extra_session["input_audio_transcription"] = [self.input_audio_transcription]
        if self.kwargs:
            extra_session.update(self.kwargs)

        # Connect using the underlying OpenAI Realtime instance
        connection = await self._realtime.connect(
            call,
            agent_user_id=agent_user_id,
            extra_session=extra_session if extra_session else None,
        )

        self.logger.info("✅ OpenAI Realtime connected successfully")
        return connection

    async def update_session(self, **session_fields):
        """Update session configuration."""
        return await self._realtime.update_session(**session_fields)

    async def send_function_call_output(self, tool_call_id: str, output: str):
        """Send a tool call output to the conversation."""
        return await self._realtime.send_function_call_output(tool_call_id, output)

    async def send_user_message(self, text: str):
        """Send a text message from the human side to the conversation."""
        return await self._realtime.send_user_message(text)

    async def request_assistant_response(self):
        """Ask OpenAI to generate the next assistant turn."""
        return await self._realtime.request_assistant_response()

    @property
    def is_connected(self) -> bool:
        """Check if the realtime session is currently active."""
        return self._realtime.is_connected

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"OpenAIRealtimeModel(model='{self.model}', voice='{self.voice}')"
