"""Gemini Live Speech-to-Speech implementation.

This module provides a wrapper around Google's Gemini Live API.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
from typing import Any, Dict, List, Optional

from aiortc.mediastreams import MediaStreamTrack

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
import numpy as np
from google.genai import Client
from google.genai.live import AsyncSession
from google.genai.types import (
    AudioTranscriptionConfigDict,
    Blob,
    ContextWindowCompressionConfigDict,
    LiveConnectConfigDict,
    Modality,
    PrebuiltVoiceConfigDict,
    RealtimeInputConfigDict,
    SlidingWindowDict,
    SpeechConfigDict,
    TurnCoverage,
    VoiceConfigDict,
)

from getstream.audio.utils import resample_audio
from getstream.plugins.common import STS
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.track_util import PcmData

logger = logging.getLogger(__name__)


class GeminiLive(STS):
    """Speech-to-Speech wrapper for Google Gemini Live API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-live-2.5-flash-preview",
        *,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        response_modalities: Optional[List[Modality]] = None,
        voice: Optional[str] = None,
        barge_in: bool = True,
        activity_threshold: int = 1000,
        silence_timeout_ms: int = 1000,
        provider_config: Optional[LiveConnectConfigDict] = None,
    ):
        """Create a new Gemini Live wrapper.

        Args:
            api_key: Gemini API key. Falls back to ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY`` env var.
            model: Model ID to use when connecting.
            instructions: Optional system instructions passed to the session.
            temperature: Optional temperature passed to the session.
            response_modalities: Optional response modalities passed to the session.
            voice: Optional voice selection passed to the session.
            barge_in: Whether to interrupt current playback when the user starts speaking.
            activity_threshold: Minimum audio activity threshold for barge-in. Range is 0-32767.
            silence_timeout_ms: Time to wait for silence after user stops speaking.
            provider_config: Provider config to pass through unchanged. Pass a
                ``LiveConnectConfigDict`` from ``google.genai.types``.
        """

        modalities_str: Optional[List[str]] = (
            [str(m) for m in response_modalities] if response_modalities else None
        )
        super().__init__(
            model=model,
            instructions=instructions,
            temperature=temperature,
            voice=voice,
            provider_config=provider_config,
            response_modalities=modalities_str,
        )

        self.api_key = (
            api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY not provided and not found in env"
            )

        if Client is None:
            raise ImportError(
                "failed to import google genai SDK, install with: uv add google-genai"
            )

        # Use v1beta for Live APIs as required by current SDK
        self.client = Client(
            api_key=self.api_key, http_options={"api_version": "v1beta"}
        )
        self.model = model
        # Build a default config if none provided, and apply simple overrides
        self.config: Optional[LiveConnectConfigDict] = self._compose_config(
            base_config=provider_config,
            response_modalities=response_modalities,
            voice=voice,
        )
        self._session: Optional[AsyncSession] = None
        self._audio_receiver_task: Optional[asyncio.Task] = None
        self._connect_task: Optional[asyncio.Task] = None
        self._playback_enabled: bool = True
        # Barge-in control
        self._barge_in_enabled: bool = barge_in
        self._silence_timeout_ms: int = silence_timeout_ms
        self._user_speaking: bool = False
        self._eos_timer_task: Optional[asyncio.Task] = None
        # Simple energy-based activity detection (int16 amplitude threshold)
        self._activity_threshold: int = activity_threshold
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        # Video sender control
        self._video_sender_task: Optional[asyncio.Task] = None

        # Create an outbound audio track for assistant speech
        self.output_track = AudioStreamTrack(
            framerate=24000, stereo=False, format="s16"
        )
        # Kick off background connect if we are in an event loop
        loop = asyncio.get_running_loop()
        self._connect_task = loop.create_task(self._run_connection())

    def _default_config(self) -> LiveConnectConfigDict:
        return LiveConnectConfigDict(
            response_modalities=[Modality.AUDIO],
            input_audio_transcription=AudioTranscriptionConfigDict(),
            output_audio_transcription=AudioTranscriptionConfigDict(),
            speech_config=SpeechConfigDict(
                voice_config=VoiceConfigDict(
                    prebuilt_voice_config=PrebuiltVoiceConfigDict(voice_name="Puck")
                )
            ),
            realtime_input_config=RealtimeInputConfigDict(
                turn_coverage=TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY
            ),
            context_window_compression=ContextWindowCompressionConfigDict(
                trigger_tokens=25600,
                sliding_window=SlidingWindowDict(target_tokens=12800),
            ),
        )

    def _compose_config(
        self,
        *,
        base_config: Optional[LiveConnectConfigDict],
        response_modalities: Optional[List[Modality]],
        voice: Optional[str],
    ) -> LiveConnectConfigDict:
        # Start from provided provider_config (preferred) or defaults
        cfg: Dict[str, Any] = dict(base_config or self._default_config())

        # Apply response modality override if given
        if response_modalities is not None:
            cfg["response_modalities"] = response_modalities

        # Voice mapping: allow simple voice string
        speech_cfg: Dict[str, Any] = dict(cfg.get("speech_config") or {})
        if voice is not None:
            speech_cfg["voice_config"] = VoiceConfigDict(
                prebuilt_voice_config=PrebuiltVoiceConfigDict(voice_name=voice)
            )
        if speech_cfg:
            cfg["speech_config"] = speech_cfg

        return LiveConnectConfigDict(**cfg)

    async def _require_session(self) -> AsyncSession:
        """Ensure the live session is connected and return it.

        Raises RuntimeError if not connected within the wait period.
        """
        if not self._is_connected or not self._session:
            await self.wait_until_ready()
        if not self._is_connected or not self._session:
            raise RuntimeError("Not connected")
        return self._session

    async def _run_connection(self) -> None:
        """Background task that connects and holds the session open until stopped."""
        if self._is_connected:
            return
        logger.info("Connecting Gemini agent using model %s", self.model)
        connect_cm = self.client.aio.live.connect(model=self.model, config=self.config)
        async with connect_cm as session:
            self._session = session
            self._is_connected = True
            self._ready_event.set()
            self.emit("connected")
            logger.info("Connected Gemini agent using model %s", self.model)
            # Start listener automatically
            await self.start_response_listener()
            # Wait until asked to stop
            await self._stop_event.wait()
        # After context exits
        self._session = None
        self._is_connected = False
        self._ready_event.clear()
        self._audio_receiver_task = None

    async def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Wait until the live session is ready. Returns True if ready."""
        if self._ready_event.is_set():
            return True
        try:
            return await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False

    async def send_text(self, text: str):
        """Send a text message from the human side to the conversation."""
        session = await self._require_session()
        await session.send_realtime_input(text=text)

    async def send_audio_pcm(self, pcm: PcmData, target_rate: int = 48000):
        """Send a `PcmData` frame to Gemini, resampling if needed.

        Args:
            pcm: PcmData from RTC pipeline (int16 numpy array or bytes).
            target_rate: Target sample rate for Gemini input.
        """
        session = await self._require_session()

        # Convert to numpy int16 array
        if isinstance(pcm.samples, (bytes, bytearray)):
            # Interpret as int16 little-endian
            audio_array = np.frombuffer(pcm.samples, dtype=np.int16)
        else:
            audio_array = np.asarray(pcm.samples)
            if audio_array.dtype != np.int16:
                audio_array = audio_array.astype(np.int16)

        # Resample if needed
        if pcm.sample_rate != target_rate:
            audio_array = resample_audio(
                audio_array, pcm.sample_rate, target_rate
            ).astype(np.int16)

        # Activity detection to avoid constant interruption when streaming continuous audio
        is_active = bool(np.mean(np.abs(audio_array)) > self._activity_threshold)

        # On first frame of a speaking burst, interrupt current playback (barge-in)
        if self._barge_in_enabled and is_active and not self._user_speaking:
            logger.info("Barge-in enabled, interrupting playback")
            await self.interrupt_playback()
            self._user_speaking = True

        # (Re)start silence timer only on active chunks; silence will let timer elapse
        if self._barge_in_enabled and is_active:
            logger.info("Barge-in enabled, starting silence timer")
            if self._eos_timer_task and not self._eos_timer_task.done():
                self._eos_timer_task.cancel()
            self._eos_timer_task = asyncio.create_task(self._silence_timeout_task())

        # Build blob and send directly
        audio_bytes = audio_array.tobytes()
        mime = f"audio/pcm;rate={target_rate}"
        blob = Blob(data=audio_bytes, mime_type=mime)
        logger.info(
            "Sending %d bytes audio to Gemini Live (%s)", len(audio_bytes), mime
        )
        await session.send_realtime_input(audio=blob)

    async def start_video_sender(self, track: MediaStreamTrack, fps: int = 1) -> None:
        """Start a background task that forwards video frames to Gemini Live.

        Args:
            track: Remote video track to read frames from.
            fps: Frames per second throttle for sending.
        """
        await self._require_session()

        if self._video_sender_task and not self._video_sender_task.done():
            return

        interval = max(0.01, 1.0 / max(1, fps))

        def _frame_to_png_bytes(frame: Any) -> bytes:
            if Image is None:
                return b""
            if hasattr(frame, "to_image"):
                img = frame.to_image()  # type: ignore[attr-defined]
            else:
                arr = frame.to_ndarray(format="rgb24")  # type: ignore[attr-defined]
                img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        async def _loop():
            try:
                while self._is_connected and self._session is not None:
                    frame = await track.recv()  # type: ignore[reportUnknownReturnType]
                    png_bytes = _frame_to_png_bytes(frame)
                    if png_bytes:
                        blob = Blob(data=png_bytes, mime_type="image/png")
                        await self._session.send_realtime_input(media=blob)
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.emit("error", e)

        self._video_sender_task = asyncio.create_task(_loop())

    async def stop_video_sender(self) -> None:
        """Stop the background video sender task, if running."""
        if self._video_sender_task and not self._video_sender_task.done():
            self._video_sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._video_sender_task
        self._video_sender_task = None

    async def _silence_timeout_task(self) -> None:
        try:
            await asyncio.sleep(self._silence_timeout_ms / 1000)
            # Consider utterance ended; resume playback for next assistant turn
            self._user_speaking = False
            self.resume_playback()
        except asyncio.CancelledError:
            return

    async def start_response_listener(
        self,
        emit_events: bool = True,
    ) -> None:
        """Start a background task to read responses and emit/forward audio chunks.

        Args:
            emit_events: If True, emit an "audio" event with chunk bytes.
        """
        await self._require_session()

        if self._audio_receiver_task and not self._audio_receiver_task.done():
            return

        async def _audio_receive_loop():
            try:
                assert self._session is not None
                # Continuously read turns; receive() yields one complete model turn
                while self._is_connected and self._session is not None:
                    async for resp in self._session.receive():
                        if data := resp.data:
                            if emit_events:
                                self.emit("audio", data)
                            # Write directly to the outbound track when playback is enabled
                            if self._playback_enabled:
                                await self.output_track.write(data)
                        text = getattr(resp, "text", None)
                        if text and emit_events:
                            self.emit("text", text)

                    # Small pause between turns to avoid tight loop
                    await asyncio.sleep(0)
            except asyncio.CancelledError:  # graceful stop
                return
            except Exception as e:
                self.emit("error", e)

        logger.info("Response listener started")
        self._audio_receiver_task = asyncio.create_task(_audio_receive_loop())
        return None

    async def interrupt_playback(self) -> None:
        """Stop current playback immediately and clear queued audio chunks."""
        # Disable playback so writer loop stops sending frames
        self._playback_enabled = False
        # Flush any pending bytes in the output track
        await self.output_track.flush()

    def resume_playback(self) -> None:
        """Re-enable playback after an interruption."""
        self._playback_enabled = True

    async def stop_response_listener(self) -> None:
        """Stop the background response listener, if running."""
        if self._audio_receiver_task and not self._audio_receiver_task.done():
            self._audio_receiver_task.cancel()
            try:
                await self._audio_receiver_task
            except asyncio.CancelledError:
                pass
        self._audio_receiver_task = None
        # Cancel end-of-speech timer
        if self._eos_timer_task and not self._eos_timer_task.done():
            self._eos_timer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._eos_timer_task
        self._eos_timer_task = None

    async def close(self) -> None:
        """Stop the session and background tasks."""
        self._stop_event.set()
        await self.stop_video_sender()
        await self.stop_response_listener()
        if self._connect_task and not self._connect_task.done():
            self._connect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._connect_task


__all__ = ["GeminiLive"]
