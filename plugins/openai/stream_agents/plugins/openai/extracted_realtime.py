"""OpenAI Realtime (audio/text only) over WebRTC.

This module provides a minimal, cleaned-up implementation of OpenAI's
Realtime API using WebRTC for audio send/receive and a data channel for
events. All video-related functionality has been intentionally removed.

Key features:
- Establish WebRTC with OpenAI using REST-based SDP exchange
- Push microphone PCM S16 mono frames into a WebRTC audio track
- Receive assistant audio via remote WebRTC audio track and forward to
  the framework's output track
- Send text turns via data channel and request an audio response
- Emit standardized events via the shared Realtime base
"""

from __future__ import annotations

import asyncio
from fractions import Fraction
import json
import logging
import os
from typing import Any, Optional

import httpx
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamError, MediaStreamTrack
from av import AudioFrame

from getstream.audio.utils import resample_audio
from stream_agents.core.llm import realtime
from stream_agents.core.llm.llm import LLMResponse


logger = logging.getLogger(__name__)


class RealtimeConnection:
    """Internal WebRTC connection handler for OpenAI Realtime API (audio only)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-realtime",
        voice: str = "alloy",
        turn_detection: bool = True,
        system_instructions: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.turn_detection = turn_detection
        self.system_instructions = system_instructions

        self.pc = RTCPeerConnection()
        self.dc = None
        self._http: Optional[httpx.AsyncClient] = None

        self._mic_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        self._mic_track: Optional[MediaStreamTrack] = None
        self._running = False

        self._audio_callbacks: list = []
        self._event_callbacks: list = []

        self._frames_dropped: int = 0
        self._max_queue_frames: int = 3  # keep latency low

    def on_audio(self, callback) -> None:
        self._audio_callbacks.append(callback)

    def on_event(self, callback) -> None:
        self._event_callbacks.append(callback)

    async def send_audio(self, audio_data: bytes, sample_rate: int = 48000) -> None:
        if not audio_data:
            return
        while self._mic_queue.qsize() >= self._max_queue_frames:
            try:
                _ = self._mic_queue.get_nowait()
                self._frames_dropped += 1
            except asyncio.QueueEmpty:
                break
        self._mic_queue.put_nowait((audio_data, int(sample_rate)))

    async def send_text(self, text: str) -> None:
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }
        await self._send_event(event)
        await self._send_event({"type": "response.create"})

    async def _send_event(self, event: dict) -> None:
        if not self.dc:
            return
        try:
            self.dc.send(json.dumps(event))
        except Exception as e:
            logger.error(f"Failed to send event: {e}")

    async def _get_session_token(self) -> Optional[str]:
        url = "https://api.openai.com/v1/realtime/sessions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "voice": self.voice}
        client = self._http or httpx.AsyncClient()
        created_here = self._http is None
        try:
            for attempt in range(2):
                try:
                    resp = await client.post(url, headers=headers, json=payload, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()
                    secret = data.get("client_secret")
                    return secret.get("value") if isinstance(secret, dict) else secret
                except Exception as e:
                    if attempt == 0:
                        await asyncio.sleep(1.0)
                        continue
                    logger.error(f"Failed to get OpenAI Realtime session token: {e}")
                    return None
        finally:
            if created_here:
                await client.aclose()

    async def _exchange_sdp(self, sdp: str) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/sdp",
        }
        url = f"https://api.openai.com/v1/realtime?model={self.model}"
        client = self._http or httpx.AsyncClient()
        created_here = self._http is None
        try:
            for attempt in range(2):
                try:
                    resp = await client.post(url, headers=headers, content=sdp, timeout=20)
                    resp.raise_for_status()
                    return resp.text or None
                except Exception as e:
                    if attempt == 0:
                        await asyncio.sleep(1.0)
                        continue
                    logger.error(f"SDP exchange failed: {e}")
                    return None
        finally:
            if created_here:
                await client.aclose()

    async def _start_session(self) -> None:
        # HTTP client for token+SDP requests
        if self._http is None:
            self._http = httpx.AsyncClient()

        self.token = await self._get_session_token()
        if not self.token:
            raise RuntimeError("Failed to obtain OpenAI Realtime session token")

        # Setup WebRTC handlers
        @self.pc.on("track")
        async def on_track(track):
            if getattr(track, "kind", None) == "audio":
                asyncio.create_task(self._process_remote_audio(track))

        # Data channel for events
        self.dc = self.pc.createDataChannel("oai-events")

        @self.dc.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
            except Exception:
                return
            asyncio.create_task(self._handle_event(data))

        # Microphone track: pull PCM16 mono frames from queue
        from aiortc.mediastreams import AudioStreamTrack as _AudioStreamTrack

        class MicAudioTrack(_AudioStreamTrack):
            kind = "audio"

            def __init__(self, queue: asyncio.Queue, sample_rate: int = 48000):
                super().__init__()
                self._queue = queue
                self._sample_rate = int(sample_rate)
                self._ts = 0
                self._silence_cache = {}

            async def recv(self):
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=0.02)
                    if isinstance(item, tuple):
                        data, sr = item
                    else:
                        data, sr = item, self._sample_rate
                    self._sample_rate = int(sr) if sr else self._sample_rate
                    arr = np.frombuffer(data, dtype=np.int16)
                    samples = arr.reshape(1, -1) if arr.ndim == 1 else arr[:1, :]
                except asyncio.TimeoutError:
                    sr = int(self._sample_rate) if self._sample_rate else 48000
                    cached = self._silence_cache.get(sr)
                    if cached is None:
                        num = int(0.02 * sr)
                        cached = np.zeros((1, num), dtype=np.int16)
                        self._silence_cache[sr] = cached
                    samples = cached

                frame = AudioFrame.from_ndarray(samples, format="s16", layout="mono")
                sr = int(self._sample_rate)
                frame.sample_rate = sr
                frame.pts = self._ts
                frame.time_base = Fraction(1, sr)
                self._ts += samples.shape[1]
                return frame

        self._mic_track = MicAudioTrack(self._mic_queue, 48000)
        self.pc.addTrack(self._mic_track)

        # Offer/Answer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        answer_sdp = await self._exchange_sdp(offer.sdp)
        if not answer_sdp:
            raise RuntimeError("Failed to exchange SDP with OpenAI")
        answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
        await self.pc.setRemoteDescription(answer)

        # Minimal session preferences
        await self._update_session()
        self._running = True
        logger.info("OpenAI Realtime session established (audio-only)")

    async def _stop_session(self) -> None:
        self._running = False
        try:
            if self._mic_track is not None:
                try:
                    self._mic_track.stop()
                except Exception:
                    pass
                self._mic_track = None
            if self.dc is not None:
                try:
                    self.dc.close()
                except Exception:
                    pass
                self.dc = None
            await self.pc.close()
        finally:
            if self._http is not None:
                try:
                    await self._http.aclose()
                except Exception:
                    pass
                self._http = None

    async def _update_session(self) -> None:
        event = {
            "type": "session.update",
            "session": {
                "instructions": self.system_instructions or "You are a helpful assistant.",
                "voice": self.voice,
                "input_audio_transcription": {"model": "gpt-4o-transcribe"},
                "turn_detection": {
                    "type": "semantic_vad" if self.turn_detection else None,
                    "eagerness": "auto",
                    "create_response": True,
                    "interrupt_response": True,
                }
                if self.turn_detection
                else None,
                "tools": [],
                "tool_choice": "auto",
                "temperature": 0.8,
                "max_response_output_tokens": 4096,
            },
        }
        await self._send_event(event)

    async def _handle_event(self, event: dict) -> None:
        et = event.get("type")
        if et == "error":
            logger.error(f"OpenAI error event: {event}")
        for cb in self._event_callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(event)
            else:
                cb(event)

    async def _process_remote_audio(self, track: MediaStreamTrack) -> None:
        while self._running:
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                samples = frame.to_ndarray()
                if samples.ndim == 2 and samples.shape[0] > 1:
                    samples = samples.mean(axis=0)
                if samples.dtype != np.int16:
                    samples = (samples * 32767).astype(np.int16)
                in_rate = getattr(track, "sample_rate", None) or getattr(frame, "sample_rate", 48000) or 48000
                if in_rate != 48000:
                    samples = resample_audio(samples, int(in_rate), 48000).astype(np.int16)
                audio_bytes = samples.tobytes()
                for cb in self._audio_callbacks:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(audio_bytes)
                    else:
                        cb(audio_bytes)
            except asyncio.TimeoutError:
                continue
            except MediaStreamError:
                break
            except Exception as e:
                logger.error(f"Error processing remote audio: {e}")
                break


class Realtime(realtime.Realtime):
    """OpenAI Realtime provider (audio/text only)."""

    def __init__(
        self,
        model: str = "gpt-realtime",
        api_key: Optional[str] = None,
        voice: str = "alloy",
        turn_detection: bool = True,
        instructions: Optional[str] = None,
        client: Optional[Any] = None,
        *,
        # Video-related args accepted for compatibility but ignored
        enable_video_input: bool = False,
        video_fps: int = 1,
        video_width: int = 1280,
        video_height: int = 720,
        video_debug_enabled: bool = False,
        save_snapshots_enabled: bool = False,
        snapshot_interval_sec: float = 3.0,
        delay_until_video: bool = False,
    ) -> None:
        super().__init__(
            provider_name="openai-realtime",
            model=model,
            instructions=instructions,
            voice=voice,
            provider_config={"turn_detection": turn_detection},
        )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.turn_detection = turn_detection
        self.system_prompt = instructions
        self.realtime = True

        self._connection: Optional[RealtimeConnection] = None
        self._connection_lock = asyncio.Lock()

        # Local playback gating
        self._playback_enabled: bool = True

        # Accept and ignore video flags for compatibility with callers
        self._ignore_video_args = {
            "enable_video_input": enable_video_input,
            "video_fps": video_fps,
            "video_width": video_width,
            "video_height": video_height,
            "video_debug_enabled": video_debug_enabled,
            "save_snapshots_enabled": save_snapshots_enabled,
            "snapshot_interval_sec": snapshot_interval_sec,
            "delay_until_video": delay_until_video,
        }

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._ensure_connection())
        except RuntimeError:
            pass

    async def _ensure_connection(self) -> None:
        async with self._connection_lock:
            if not self._connection or not self._is_connected:
                await self._create_connection()

    async def _create_connection(self) -> None:
        try:
            conn = RealtimeConnection(
                api_key=self.api_key,
                model=self.model,
                voice=self.voice,
                turn_detection=self.turn_detection,
                system_instructions=self.system_prompt or self.instructions,
            )
            conn.on_event(self._handle_openai_event)
            conn.on_audio(self._handle_audio_output)
            await conn._start_session()

            self._connection = conn
            self._emit_connected_event(
                session_config={"model": self.model, "voice": self.voice, "turn_detection": self.turn_detection},
                capabilities=["text", "audio"],
            )

            # Seed publisher pipeline with brief silence
            if getattr(self, "output_track", None) is not None:
                try:
                    await self.output_track.write(b"\x00" * 9600)
                except Exception:
                    pass
        except Exception as e:
            self._emit_error_event(e, "connection")
            raise

    async def _handle_openai_event(self, event: dict) -> None:
        et = event.get("type")
        if et == "response.audio_transcript.done":
            transcript = event.get("transcript", "")
            if transcript:
                self._emit_transcript_event(text=transcript, user_metadata={"role": "assistant", "source": "openai"})
                self._emit_response_event(
                    text=transcript,
                    response_id=event.get("response_id"),
                    is_complete=True,
                    conversation_item_id=event.get("item_id"),
                )
        elif et == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                self._emit_transcript_event(text=transcript, user_metadata={"role": "user", "source": "openai"})
        elif et == "error":
            error = event.get("error", {})
            self._emit_error_event(Exception(error.get("message", "Unknown error")), context=f"openai_event: {error.get('code', 'unknown')}")

    async def _handle_audio_output(self, audio_bytes: bytes) -> None:
        listeners_fn = getattr(self, "listeners", None)
        has_listeners = bool(listeners_fn("audio_output")) if callable(listeners_fn) else False
        if has_listeners:
            self._emit_audio_output_event(audio_data=audio_bytes, sample_rate=48000)
        if self._playback_enabled and getattr(self, "output_track", None) is not None:
            try:
                await self.output_track.write(audio_bytes)
            except Exception:
                pass

    async def send_audio_pcm(self, pcm_data, target_rate: int = 48000):
        await self._send_audio_pcm_async(pcm_data, target_rate)

    async def _send_audio_pcm_async(self, pcm_data, target_rate: int) -> None:
        await self._ensure_connection()
        if not self._connection:
            return

        audio_bytes = None
        samples = getattr(pcm_data, "samples", None)
        if samples is not None:
            if isinstance(samples, (bytes, bytearray)):
                if len(samples) >= 1920:
                    audio_bytes = bytes(samples)
            else:
                arr = np.asarray(samples)
                if arr.size:
                    if arr.dtype != np.int16:
                        arr = arr.astype(np.int16)
                    audio_bytes = arr.tobytes()
        elif isinstance(pcm_data, bytes) and len(pcm_data) >= 1920:
            audio_bytes = pcm_data

        if not audio_bytes:
            return

        src_rate = int(getattr(pcm_data, "sample_rate", target_rate) or 48000)
        self._emit_audio_input_event(audio_data=audio_bytes, sample_rate=src_rate)

        samples_per_frame = int(0.02 * src_rate)
        frame_bytes = max(1920, samples_per_frame * 2)
        total_frames = max(1, (len(audio_bytes) + frame_bytes - 1) // frame_bytes)
        start_frame = max(0, total_frames - 3)
        start_offset = start_frame * frame_bytes
        for i in range(start_offset, len(audio_bytes), frame_bytes):
            chunk = audio_bytes[i : i + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
            await self._connection.send_audio(chunk, sample_rate=src_rate)

    async def send_text(self, text: str) -> None:
        await self._ensure_connection()
        self._emit_transcript_event(text=text, user_metadata={"role": "user"})
        if not self._connection:
            raise RuntimeError("Failed to establish OpenAI Realtime connection")
        self._playback_enabled = True
        await self._connection.send_text(text)

    async def simple_response(self, *, text: str, timeout: Optional[float] = 30.0):
        return await super().simple_response(text=text, timeout=timeout)

    async def create_response(self, *args, **kwargs):
        text = kwargs.get("input", args[0] if args else "")
        rt_resp = await self.simple_response(text=text)
        return LLMResponse(original=rt_resp.original, text=rt_resp.text)

    async def _close_impl(self) -> None:
        if self._connection:
            try:
                await self._connection._stop_session()
            except Exception as e:
                logger.error(f"Error closing OpenAI Realtime connection: {e}")
            finally:
                self._connection = None

    async def interrupt_playback(self) -> None:
        self._playback_enabled = False
        if getattr(self, "output_track", None) is not None:
            flush_fn = getattr(self.output_track, "flush", None)
            if callable(flush_fn):
                try:
                    await flush_fn()
                except Exception:
                    pass

    def resume_playback(self) -> None:
        self._playback_enabled = True


__all__ = ["Realtime"]


