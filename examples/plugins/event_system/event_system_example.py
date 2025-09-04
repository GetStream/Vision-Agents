#!/usr/bin/env python3
"""
Example demonstrating the GetStream AI Plugins Event System.

This example shows how to:
1. Use structured events with different plugin types
2. Set up event filtering and analysis
3. Monitor plugin performance
4. Handle errors gracefully
5. Serialize events for storage

Run this example to see the event system in action.
"""

import asyncio
import json
import logging

# Import the common plugin system
from getstream.plugins.common import (
    # Base classes
    STT,
    TTS,
    VAD,
    EventType,
    STTTranscriptEvent,
    VADAudioEvent,
    # Event utilities
    EventFilter,
    EventRegistry,
    get_global_registry,
)
from getstream.plugins.common.event_metrics import (
    calculate_stt_metrics,
    calculate_tts_metrics,
)
from getstream.plugins.common.event_serialization import (
    serialize_events,
    deserialize_event,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example STT Plugin Implementation
class ExampleSTTPlugin(STT):
    """Example STT plugin that simulates transcription."""

    def __init__(self):
        super().__init__(provider_name="example_stt")
        self.transcription_count = 0

    async def _process_audio_impl(self, pcm_data, user_metadata=None):
        """Simulate audio transcription with structured events."""
        # Simulate processing delay
        await asyncio.sleep(0.1)

        self.transcription_count += 1

        # Simulate different confidence levels
        confidence = 0.8 + (self.transcription_count % 3) * 0.1

        # Create metadata
        metadata = {
            "confidence": confidence,
            "processing_time_ms": 100.0,
            "model_name": "example_model_v1",
            "language": "en-US",
        }

        # Return results for synchronous mode
        text = f"This is transcript number {self.transcription_count}"
        return [(True, text, metadata)]

    async def close(self):
        logger.info(
            f"Closing STT plugin after {self.transcription_count} transcriptions"
        )
        await super().close()


# Example TTS Plugin Implementation
class ExampleTTSPlugin(TTS):
    """Example TTS plugin that simulates speech synthesis."""

    def __init__(self):
        super().__init__(provider_name="example_tts")
        self.synthesis_count = 0

    async def synthesize(self, text: str, *args, **kwargs):
        """Simulate text-to-speech synthesis."""
        self.synthesis_count += 1

        # Simulate processing delay
        await asyncio.sleep(0.2)

        # Generate fake audio data
        audio_size = len(text) * 100  # Simulate audio size based on text length
        fake_audio = b"\x00" * audio_size

        return fake_audio

    async def close(self):
        logger.info(f"Closing TTS plugin after {self.synthesis_count} syntheses")
        await super().close()


# Example VAD Plugin Implementation
class ExampleVADPlugin(VAD):
    """Example VAD plugin that simulates voice activity detection."""

    def __init__(self):
        super().__init__(provider_name="example_vad")
        self.frame_count = 0

    async def is_speech(self, frame):
        """Simulate speech detection."""
        self.frame_count += 1

        # Simulate varying speech probabilities
        # Every 5th frame is "speech", others are silence
        return 0.9 if self.frame_count % 5 == 0 else 0.1

    async def close(self):
        logger.info(f"Closing VAD plugin after processing {self.frame_count} frames")
        await super().close()


# Event Analysis Functions
def analyze_stt_performance(registry: EventRegistry):
    """Analyze STT performance from events."""
    # Get STT events
    stt_filter = EventFilter(
        event_types=[EventType.STT_TRANSCRIPT, EventType.STT_PARTIAL_TRANSCRIPT],
        time_window_ms=60000,  # Last minute
    )

    stt_events = registry.get_events(stt_filter)

    if not stt_events:
        print("No STT events found")
        return

    # Calculate metrics
    metrics = calculate_stt_metrics(stt_events)

    print("\n=== STT Performance Analysis ===")
    print(f"Total transcripts: {metrics['total_transcripts']}")
    print(f"Final transcripts: {metrics['final_transcripts']}")
    print(f"Partial transcripts: {metrics['partial_transcripts']}")

    if "avg_processing_time_ms" in metrics:
        print(f"Average processing time: {metrics['avg_processing_time_ms']:.2f}ms")
        print(f"Min processing time: {metrics['min_processing_time_ms']:.2f}ms")
        print(f"Max processing time: {metrics['max_processing_time_ms']:.2f}ms")

    if "avg_confidence" in metrics:
        print(f"Average confidence: {metrics['avg_confidence']:.2f}")
        print(f"Min confidence: {metrics['min_confidence']:.2f}")
        print(f"Max confidence: {metrics['max_confidence']:.2f}")


def analyze_tts_performance(registry: EventRegistry):
    """Analyze TTS performance from events."""
    tts_filter = EventFilter(
        event_types=[EventType.TTS_SYNTHESIS_COMPLETE], time_window_ms=60000
    )

    tts_events = registry.get_events(tts_filter)

    if not tts_events:
        print("No TTS completion events found")
        return

    metrics = calculate_tts_metrics(registry.get_events())

    print("\n=== TTS Performance Analysis ===")
    print(f"Total syntheses: {metrics['completed_syntheses']}")
    print(f"Total audio chunks: {metrics['total_audio_chunks']}")

    if "avg_synthesis_time_ms" in metrics:
        print(f"Average synthesis time: {metrics['avg_synthesis_time_ms']:.2f}ms")
        print(f"Min synthesis time: {metrics['min_synthesis_time_ms']:.2f}ms")
        print(f"Max synthesis time: {metrics['max_synthesis_time_ms']:.2f}ms")

    if "avg_real_time_factor" in metrics:
        print(f"Average real-time factor: {metrics['avg_real_time_factor']:.2f}")


def demonstrate_event_filtering(registry: EventRegistry):
    """Demonstrate various event filtering capabilities."""
    print("\n=== Event Filtering Examples ===")

    # Filter by event type
    error_filter = EventFilter(
        event_types=[EventType.STT_ERROR, EventType.TTS_ERROR, EventType.VAD_ERROR]
    )

    error_events = registry.get_events(error_filter)
    print(f"Total error events: {len(error_events)}")

    # Filter by confidence threshold
    high_confidence_filter = EventFilter(
        event_types=[EventType.STT_TRANSCRIPT], min_confidence=0.9
    )

    high_confidence_events = registry.get_events(high_confidence_filter)
    print(f"High confidence transcripts: {len(high_confidence_events)}")

    # Filter by time window
    recent_filter = EventFilter(time_window_ms=30000)  # Last 30 seconds
    recent_events = registry.get_events(recent_filter)
    print(f"Recent events (30s): {len(recent_events)}")

    # Filter by plugin name
    stt_plugin_filter = EventFilter(plugin_names=["example_stt"])
    stt_plugin_events = registry.get_events(stt_plugin_filter)
    print(f"STT plugin events: {len(stt_plugin_events)}")


def demonstrate_event_serialization(registry: EventRegistry):
    """Demonstrate event serialization and deserialization."""
    print("\n=== Event Serialization Example ===")

    # Get recent events
    recent_events = registry.get_events(limit=5)

    if not recent_events:
        print("No events to serialize")
        return

    # Serialize events
    events_json = serialize_events(recent_events)
    print(f"Serialized {len(recent_events)} events")
    print(f"JSON size: {len(events_json)} characters")

    # Save to file (example)
    filename = "example_events.json"
    with open(filename, "w") as f:
        f.write(events_json)
    print(f"Saved events to {filename}")

    # Load and deserialize
    with open(filename, "r") as f:
        loaded_json = f.read()

    events_data = json.loads(loaded_json)
    restored_events = [deserialize_event(event_data) for event_data in events_data]

    print(f"Restored {len(restored_events)} events")

    # Verify restoration
    for original, restored in zip(recent_events, restored_events):
        print(
            f"Original: {original.event_type.value} - Restored: {restored.event_type.value}"
        )


async def simulate_plugin_usage():
    """Simulate usage of different plugin types with events."""
    print("=== Starting Plugin Event System Demo ===\n")

    # Create plugin instances
    stt_plugin = ExampleSTTPlugin()
    tts_plugin = ExampleTTSPlugin()
    vad_plugin = ExampleVADPlugin()

    # Set up event handlers
    @stt_plugin.on("transcript")
    async def on_stt_transcript(event: STTTranscriptEvent):
        print(f"📝 STT Transcript: '{event.text}' (confidence: {event.confidence:.2f})")

    @tts_plugin.on("synthesis_complete")
    async def on_tts_complete(event):
        print(
            f"🔊 TTS Complete: {event.total_audio_bytes} bytes in {event.synthesis_time_ms:.1f}ms"
        )

    @vad_plugin.on("speech_start")
    async def on_speech_start(event):
        print(f"🎤 Speech started (probability: {event.speech_probability:.2f})")

    @vad_plugin.on("audio")
    async def on_vad_audio(event: VADAudioEvent):
        print(f"🎵 VAD Audio: {len(event.audio_data)} bytes, {event.duration_ms:.1f}ms")

    # Set up a mock audio track for TTS
    class MockAudioTrack:
        def __init__(self):
            self.framerate = 16000

        async def write(self, data):
            pass  # Mock write operation

    tts_plugin.set_output_track(MockAudioTrack())

    print("Plugins initialized. Starting simulation...\n")

    # Simulate STT processing
    print("--- STT Processing ---")
    from getstream.video.rtc.track_util import PcmData
    import numpy as np

    for i in range(3):
        # Create mock audio data
        mock_audio = np.random.randint(-32768, 32767, 1600, dtype=np.int16)
        pcm_data = PcmData(samples=mock_audio, sample_rate=16000, format="s16")

        await stt_plugin.process_audio(pcm_data, {"user_id": f"user_{i}"})
        await asyncio.sleep(0.1)

    # Simulate TTS processing
    print("\n--- TTS Processing ---")
    texts_to_synthesize = [
        "Hello, this is a test.",
        "The event system is working great!",
        "This is the final synthesis test.",
    ]

    for text in texts_to_synthesize:
        await tts_plugin.send(text, {"user_id": "tts_user"})
        await asyncio.sleep(0.1)

    # Simulate VAD processing
    print("\n--- VAD Processing ---")
    for i in range(10):
        # Create mock frame data
        frame_audio = np.random.randint(-32768, 32767, 512, dtype=np.int16)
        frame_pcm = PcmData(samples=frame_audio, sample_rate=16000, format="s16")

        await vad_plugin.process_audio(frame_pcm, {"user_id": "vad_user"})
        await asyncio.sleep(0.05)

    # Wait a bit for events to process
    await asyncio.sleep(0.5)

    # Analyze results
    registry = get_global_registry()

    print("\n=== Event Registry Statistics ===")
    stats = registry.get_statistics()
    print(f"Total events recorded: {stats['total_events']}")
    print(f"Active sessions: {stats['session_count']}")
    print(f"Events per second: {stats['events_per_second']:.2f}")
    print(f"Error rate: {stats['error_summary']['error_rate']:.2%}")

    # Demonstrate analysis functions
    analyze_stt_performance(registry)
    analyze_tts_performance(registry)
    demonstrate_event_filtering(registry)
    demonstrate_event_serialization(registry)

    # Clean up
    print("\n--- Cleaning Up ---")
    await stt_plugin.close()
    await tts_plugin.close()
    await vad_plugin.close()

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(simulate_plugin_usage())
