"""
Gemini Live STS Model Implementation

This module provides a Gemini Live API implementation that can be used
with the Agent class for Speech-to-Speech functionality with native audio mode.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore


class GeminiLiveModel:
    """
    Gemini Live implementation for use with Stream Agents.

    This class provides a wrapper around Google's Gemini Live API
    for real-time speech-to-speech with native audio capabilities.

    Example usage:
        # Basic usage with native audio
        sts_model = GeminiLiveModel(
            api_key="your-gemini-api-key",
            model="gemini-2.5-flash-preview-native-audio-dialog",
            instructions="You are a helpful assistant."
        )

        agent = Agent(
            llm=sts_model
        )

        await agent.join(call)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash-preview-native-audio-dialog",
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_modalities: Optional[List[str]] = None,
        voice_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the Gemini Live model.

        Args:
            api_key: Google AI API key (if not using GOOGLE_API_KEY env var)
            model: Model name (default: "gemini-2.5-flash-preview-native-audio-dialog" for native audio)
            instructions: System instructions for the assistant
            tools: List of tools/functions the assistant can call
            response_modalities: Response modalities (e.g., ["AUDIO"])
            voice_config: Voice configuration settings
            **kwargs: Additional configuration options
        """
        if genai is None:
            raise ImportError(
                "google-genai package is required for Gemini Live API. "
                "Install with: pip install google-genai"
            )

        self.api_key = (
            api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY not provided and not found in environment"
            )

        self.model = model
        self.instructions = instructions
        self.tools = tools or []
        self.response_modalities = response_modalities or ["AUDIO"]
        self.voice_config = voice_config or {}
        self.kwargs = kwargs

        # Mark this as an STS model
        self.sts = True

        self.logger = logging.getLogger(f"GeminiLiveModel[{model}]")

        # Initialize the Google GenAI client
        self._client = genai.Client(api_key=self.api_key)
        self._session_manager = None
        self._is_connected = False

        self.logger.info(f"Initialized Gemini Live model: {model}")

    async def connect(self, call, agent_user_id: str = "assistant"):
        """
        Connect to a Stream video call and establish Gemini Live session.

        Args:
            call: Stream video call object
            agent_user_id: User ID for the agent

        Returns:
            Connection object that can be used to iterate over events
        """
        self.logger.info(f"Connecting Gemini Live to call {call.id}")

        # Build configuration for Gemini Live
        config: Dict[str, Any] = {
            "response_modalities": self.response_modalities,
        }

        if self.instructions:
            config["system_instruction"] = [self.instructions]

        if self.tools:
            config["tools"] = [self.tools]

        # Add any additional configuration
        config.update(self.kwargs)

        try:
            # Get the async context manager for Gemini Live API
            self._session_manager = self._client.aio.live.connect(
                model=self.model, config=config  # type: ignore
            )

            self._is_connected = True
            connection = GeminiLiveConnection(
                self, call, agent_user_id, self._session_manager
            )

            self.logger.info("✅ Gemini Live connected successfully")
            return connection

        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Gemini Live: {e}")
            raise

    @property
    def is_connected(self) -> bool:
        """Check if the session is currently active."""
        return self._is_connected

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"GeminiLiveModel(model='{self.model}', connected={self._is_connected})"


class GeminiLiveConnection:
    """Connection wrapper for Gemini Live API."""

    def __init__(
        self, sts_model: GeminiLiveModel, call, agent_user_id: str, session_manager
    ):
        self.sts_model = sts_model
        self.call = call
        self.agent_user_id = agent_user_id
        self.session_manager = session_manager
        self.session = None
        self.logger = sts_model.logger
        self._audio_callbacks: List[Any] = []

    async def __aenter__(self):
        """Enter the async context and establish the Gemini Live session."""
        self.session = await self.session_manager.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the connection."""
        try:
            if self.session_manager:
                await self.session_manager.__aexit__(exc_type, exc_val, exc_tb)
            self.sts_model._is_connected = False
            self.sts_model._session_manager = None
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        return False

    def on_audio(self, callback):
        """Register a callback for audio data."""
        self._audio_callbacks.append(callback)

    async def send(self, event: Dict[str, Any]):
        """Send an event to Gemini Live API."""
        if not self.session:
            raise RuntimeError("Not connected to Gemini Live")

        # Handle different event types
        if event.get("type") == "input_audio_buffer.append":
            # Convert base64 audio to the format expected by Gemini
            audio_data = base64.b64decode(event["audio"])

            # Send audio to Gemini Live
            await self.session.send_realtime_input(
                audio=types.Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
            )

        else:
            self.logger.warning(f"Unhandled event type: {event.get('type')}")

    async def send_audio(self, audio_data: bytes):
        """Send raw audio data to Gemini Live."""
        if not self.session:
            raise RuntimeError("Not connected to Gemini Live")

        self.logger.debug(f"📤 Sending {len(audio_data)} bytes of audio to Gemini Live")

        # Send audio directly to Gemini Live
        await self.session.send_realtime_input(
            audio=types.Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
        )

    def __aiter__(self):
        return self

    async def __anext__(self):
        """Iterate over events from Gemini Live."""
        if not self.session:
            raise StopAsyncIteration

        try:
            # Receive response from Gemini Live
            async for response in self.session.receive():
                self.logger.debug(f"📥 Received Gemini Live response: {type(response)}")

                # Handle audio data
                if response.data is not None:
                    self.logger.info(
                        f"🎵 Received audio data from Gemini Live: {len(response.data)} bytes"
                    )
                    # Forward audio to registered callbacks
                    for callback in self._audio_callbacks:
                        try:
                            await callback(response.data)
                        except Exception as e:
                            self.logger.error(f"Error in audio callback: {e}")

                # Check for other response types
                if hasattr(response, "server_content") and response.server_content:
                    if (
                        hasattr(response.server_content, "model_turn")
                        and response.server_content.model_turn
                    ):
                        self.logger.debug("📝 Received model turn from Gemini Live")
                    elif (
                        hasattr(response.server_content, "turn_complete")
                        and response.server_content.turn_complete
                    ):
                        self.logger.debug("✅ Turn complete from Gemini Live")

                # Create a mock event structure for compatibility
                class GeminiEvent:
                    def __init__(self, event_type: str, data: Any = None):
                        self.type = event_type
                        self.data = data

                # Return event based on response type
                if response.data is not None:
                    return GeminiEvent("audio.data", response.data)
                elif hasattr(response, "server_content") and response.server_content:
                    if (
                        hasattr(response.server_content, "model_turn")
                        and response.server_content.model_turn
                    ):
                        return GeminiEvent("response.audio_transcript.done")
                    elif (
                        hasattr(response.server_content, "turn_complete")
                        and response.server_content.turn_complete
                    ):
                        return GeminiEvent("response.done")

                # Default event
                return GeminiEvent("session.updated")

        except Exception as e:
            self.logger.error(f"Error receiving from Gemini Live: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise StopAsyncIteration

    @property
    def connection(self):
        """Provide access to the connection for sending events."""
        return self


# Backward compatibility - keep the old class name as an alias
GeminiSTS = GeminiLiveModel
