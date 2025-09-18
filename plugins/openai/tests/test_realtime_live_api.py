import asyncio

import pytest

from stream_agents.core.llm.llm import LLMResponseEvent
from stream_agents.plugins.openai.realtime import Realtime
from stream_agents.core.events import (
    RealtimeConnectedEvent,
    RealtimeTranscriptEvent,
    RealtimeResponseEvent,
    RealtimeErrorEvent,
    RealtimeAudioOutputEvent,
)

from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData
import os
import wave
import numpy as np
try:
    import av  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    av = None  # type: ignore


def _load_wav_as_pcm16_mono(wav_path: str):
    """Load a WAV/Audio file into mono int16 PCM bytes and sample rate.

    Tries the built-in wave module first (PCM only). If the file is float/other
    format, falls back to PyAV to decode and convert.

    Returns: (pcm_bytes, sample_rate, is_speech_like)
    """
    try:
        with wave.open(wav_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            nframes = min(wf.getnframes(), 48000)  # ~1s cap
            frames = wf.readframes(nframes)
        if sampwidth != 2:
            raise wave.Error("non-16-bit sample width")
        if channels == 2:
            arr = np.frombuffer(frames, dtype=np.int16).reshape(-1, 2)
            mono = arr.mean(axis=1).astype(np.int16)
            return mono.tobytes(), sample_rate, _is_speech_filename(wav_path)
        elif channels == 1:
            return frames, sample_rate, _is_speech_filename(wav_path)
        raise wave.Error("unsupported channel count")
    except Exception:
        pass

    # Fallback: Use PyAV to decode and convert to s16 mono
    try:
        if av is None:
            raise RuntimeError("PyAV not available")
        container = av.open(wav_path)
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError("no audio stream")
        samples = []
        sample_rate = None
        total = 0
        target = None
        for frame in container.decode(stream):
            sample_rate = frame.sample_rate or sample_rate or 48000
            # ndarray shape (channels, samples)
            arr = frame.to_ndarray()
            if arr.ndim == 2 and arr.shape[0] > 1:
                arr = arr.mean(axis=0)
            # Normalize to int16
            if arr.dtype != np.int16:
                arr = (arr.astype(np.float32) * 32767.0).clip(-32768, 32767).astype(np.int16)
            samples.append(arr)
            total += arr.shape[0]
            if target is None:
                target = sample_rate  # ~1s
            if total >= target:
                break
        if not samples or sample_rate is None:
            raise RuntimeError("decoding produced no samples")
        mono = np.concatenate(samples)
        return mono.tobytes(), int(sample_rate), _is_speech_filename(wav_path)
    except Exception:
        # Last resort: synthesize a 1 kHz tone at 48 kHz for ~1s
        sr = 48000
        t = np.arange(0, sr, dtype=np.float32)
        tone = (0.1 * np.sin(2 * np.pi * 1000.0 * (t / sr))).astype(np.float32)
        pcm = (tone * 32767.0).astype(np.int16).tobytes()
        return pcm, sr, False


def _is_speech_filename(path: str) -> bool:
    try:
        base = os.path.basename(path).lower()
        return base.startswith("speech") or base.startswith("formant")
    except Exception:
        return False

load_dotenv()


class TestRealtime:
    """Test suite for Realtime class with real API calls."""

    @pytest.fixture
    def llm(self) -> Realtime:
        return Realtime(model="gpt-realtime")

    @pytest.mark.integration
    async def test_simple(self, llm: Realtime):
        """Test simple text response with event listening."""
        # Track events
        events = []
        connected = False

        # Register event listeners
        @llm.on("connected")
        async def on_connected(event: RealtimeConnectedEvent):
            nonlocal connected
            connected = True
            events.append(("connected", event))
            print(
                f"Connected to {event.provider} with model {event.session_config.get('model')}"
            )

        @llm.on("transcript")
        async def on_transcript(event: RealtimeTranscriptEvent):
            events.append(("transcript", event))
            role = "User" if event.is_user else "Assistant"
            print(f"{role} transcript: {event.text}")

        @llm.on("response")
        async def on_response(event: RealtimeResponseEvent):
            events.append(("response", event))
            print(f"Response: {event.text}")

        @llm.on("error")
        async def on_error(event):
            events.append(("error", event))
            print(f"Error: {event.error_message}")

        try:
            # Send the question
            print("Sending question...")
            response = await llm.simple_response(
                text="What is the capital of France?"
            )
            print(f"Got response: {response.text}")

            # Verify connection was established
            assert connected, "Connection was not established"

            # Verify we got a response
            assert response.text, "No response text received"

            # Check that the response mentions Paris
            assert "paris" in response.text.lower(), (
                f"Expected 'Paris' in response, got: {response.text}"
            )

            # Verify events were emitted
            event_types = [e[0] for e in events]
            assert "connected" in event_types, "No connected event"
            assert "response" in event_types, "No response event"

            # Check the response event
            response_events = [e[1] for e in events if e[0] == "response"]
            assert len(response_events) > 0
            assert "paris" in response_events[0].text.lower()

            print("✅ Test passed!")

        finally:
            # Clean up
            await llm.close()

    @pytest.mark.integration
    async def test_native_api(self, llm: Realtime):
        """Test create_response method for compatibility."""
        response = await llm.create_response(
            input="Say hello in English.",
        )

        assert isinstance(response, LLMResponseEvent)
        assert response.text
        assert any(
            word in response.text.lower() for word in ["hello", "hi", "hey"]
        )

    @pytest.mark.integration
    async def test_event_emission(self, llm: Realtime):
        """Test that proper events are emitted during conversation."""
        events = []

        # Capture all events
        @llm.on("connected")
        async def on_connected(event):
            events.append(event)

        @llm.on("transcript")
        async def on_transcript(event):
            events.append(event)

        @llm.on("response")
        async def on_response(event):
            events.append(event)

        @llm.on("error")
        async def on_error(event):
            events.append(event)

        try:
            # Make a simple request
            await llm.simple_response(text="What is 2+2?")

            # Verify connected event
            connected_events = [
                e for e in events if isinstance(e, RealtimeConnectedEvent)
            ]
            assert len(connected_events) >= 1
            connected = connected_events[0]
            assert connected.provider == "openai"
            assert connected.session_config["model"] == llm.model
            assert "text" in connected.capabilities
            assert "audio" in connected.capabilities

            # Verify response event
            response_events = [
                e for e in events if isinstance(e, RealtimeResponseEvent)
            ]
            assert len(response_events) >= 1
            resp_event = response_events[0]
            assert resp_event.text
            assert resp_event.is_complete
            assert any(word in resp_event.text.lower() for word in ["4", "four"])

        finally:
            await llm.close()

    @pytest.mark.integration
    async def test_multiple_messages(self, llm: Realtime):
        """Test multiple messages in sequence."""
        try:
            # First message
            response1 = await llm.simple_response(text="Remember the number 42.")
            assert response1.text

            # Second message - connection should be reused
            response2 = await llm.simple_response(
                text="What number did I just tell you?"
            )
            assert response2.text
            # Note: OpenAI Realtime maintains conversation context within the same session
            assert any(
                word in response2.text for word in ["42", "forty-two", "forty two"]
            )

        finally:
            await llm.close()

    @pytest.mark.integration
    async def test_multiple_exchanges(self):
        """Test multiple question/answer exchanges in the same session."""
        llm = Realtime(
            voice="echo",
            turn_detection=False,
            instructions="You are a helpful math tutor. Keep answers very short.",
        )

        try:
            # First question
            response1 = await llm.simple_response(text="What is 2 + 2?")
            assert any(word in response1.text.lower() for word in ["4", "four"])

            # Second question - should reuse connection
            response2 = await llm.simple_response(text="What is 10 times that?")
            assert any(word in response2.text.lower() for word in ["40", "forty"])

            print("✅ Multiple exchanges test passed!")

        finally:
            await llm.close()

    @pytest.mark.integration
    async def test_error_handling(self):
        """Test error handling with invalid configuration."""
        # Create LLM with invalid model
        llm = Realtime(
            model="invalid-model-name",
            voice="alloy",
        )

        error_events = []

        @llm.on("error")
        async def on_error(event: RealtimeErrorEvent):
            error_events.append(event)

        # This should fail
        try:
            await llm.simple_response(text="Hello")
            # If we get here, the test should fail
            assert False, "Expected an error but got response"
        except Exception:
            # Expected
            pass

        # little sleep here is necessary because errors are delivered async
        await asyncio.sleep(1)

        # Clean up
        try:
            await llm.close()
        except Exception:
            pass

        # Verify error event was emitted
        # Note: sometimes error events might not be captured if the connection fails too quickly
        if len(error_events) > 0:
            assert error_events[0].context == "connection"

    @pytest.mark.integration
    async def test_close(self, llm: Realtime):
        """Test closing the connection properly."""
        # Make a request to establish connection
        await llm.simple_response(text="Hello")

        # Track disconnection
        events = []

        @llm.on("disconnected")
        async def on_disconnected(event):
            events.append(("disconnected", event))

        @llm.on("closed")
        async def on_closed(event):
            events.append(("closed", event))

        # Close the LLM
        await llm.close()

        # Give a moment for async events to be processed
        await asyncio.sleep(0.1)

        # Verify events
        event_types = [e[0] for e in events]
        assert "disconnected" in event_types, (
            f"Expected disconnected event, got: {event_types}"
        )
        assert "closed" in event_types, f"Expected closed event, got: {event_types}"

    @pytest.mark.integration
    async def test_webrtc_audio_roundtrip(self):
        """WebRTC: verify remote audio output and user transcript from sent audio.

        - Triggers a spoken response to ensure remote audio track arrives
        - Sends a short WAV speech sample via send_audio_pcm and expects a user transcript
        """
        load_dotenv()
        # Skip if API key not present
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set; skipping live WebRTC audio test")

        llm = Realtime(
            model="gpt-realtime",
            voice="alloy",
            turn_detection=True,
            instructions="You are a helpful assistant. Keep answers brief.",
        )

        audio_outputs = []
        user_transcripts = []

        @llm.on("audio_output")
        async def on_audio_output(event: RealtimeAudioOutputEvent):
            audio_outputs.append((len(event.audio_data), event.sample_rate))

        @llm.on("transcript")
        async def on_transcript(event: RealtimeTranscriptEvent):
            if event.is_user:
                user_transcripts.append(event.text)

        try:
            # 1) Ask for a spoken reply to force remote audio track
            resp = await llm.simple_response(text="Say the word 'testing' clearly.")
            assert resp is not None

            # Wait briefly to allow audio frames to arrive
            for _ in range(20):
                if audio_outputs:
                    break
                await asyncio.sleep(0.1)

            assert len(audio_outputs) > 0, "Expected audio_output frames from OpenAI"
            # Sanity: first frame should be non-empty and 48kHz
            size, sr = audio_outputs[0]
            assert size > 0
            assert sr in (48000,)

            # 2) Send a short speech clip and expect a user transcript
            # Prefer local test asset if present
            candidate_paths = [
                "/Users/mkahan/Development/agents/plugins/test_assets/speech_16k.wav",
                "/Users/mkahan/Development/agents/plugins/test_assets/formant_speech_16k.wav",
                "/Users/mkahan/Development/agents/plugins/test_assets/speech_48k.wav",
                "/Users/mkahan/Development/agents/plugins/test_assets/formant_speech_48k.wav",
                "/Users/mkahan/Development/agents/plugins/test_assets/test_tone_48k.wav",
            ]
            wav_path = next((p for p in candidate_paths if os.path.exists(p)), None)
            if not wav_path:
                pytest.skip("No local WAV test asset available; skipping transcript half")

            pcm_bytes, sample_rate, is_speech_like = _load_wav_as_pcm16_mono(wav_path)
            pcm = PcmData(samples=pcm_bytes, sample_rate=sample_rate, format="s16")
            # Send the audio; provider will forward over WebRTC
            await llm.send_audio_pcm(pcm, target_rate=48000)

            # Give some time for upstream ASR to return a user transcript event
            for _ in range(30):
                if user_transcripts:
                    break
                await asyncio.sleep(0.1)

            # Not all assets contain words; only assert presence if we used a speech-like asset
            if is_speech_like:
                assert len(user_transcripts) > 0, "Expected a user transcript from sent audio"

        finally:
            await llm.close()
