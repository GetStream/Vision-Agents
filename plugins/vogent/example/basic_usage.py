"""
Basic usage example for Vogent Turn Detection plugin.

This example demonstrates how to initialize and use the Vogent Turn Detection
plugin for detecting conversation turns.
"""

import asyncio
import logging
import numpy as np

from getstream.video.rtc.track_util import PcmData
from vision_agents.plugins.vogent import TurnDetection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def main():
    """Main example function."""
    
    # Initialize the turn detector
    detector = TurnDetection(
        model_name="vogent/Vogent-Turn-80M",
        buffer_duration=2.0,
        confidence_threshold=0.5,
        compile_model=True,
    )
    
    # Register event handlers
    @detector.on("turn_started")
    def on_turn_started(event_data):
        print(f"🎤 Turn started: {event_data.speaker_id}")
    
    @detector.on("turn_ended")
    def on_turn_ended(event_data):
        print(
            f"✅ Turn ended: {event_data.speaker_id} "
            f"(confidence: {event_data.confidence:.3f})"
        )
        if event_data.custom:
            print(f"   Details: {event_data.custom}")
    
    # Start detection
    detector.start()
    print("Turn detection started\n")
    
    # Simulate processing audio chunks
    # In a real application, this would be audio from a microphone or stream
    sample_rate = 48000  # Input sample rate (will be resampled to 16kHz)
    chunk_duration = 0.1  # 100ms chunks
    samples_per_chunk = int(sample_rate * chunk_duration)
    
    user_id = "user123"
    
    print("Simulating audio processing...")
    print("(In production, this would be real audio data)\n")
    
    # Process several chunks of simulated audio
    for i in range(30):  # ~3 seconds of audio
        # Create dummy audio data (in production, this would be real audio)
        audio_samples = np.random.randint(-1000, 1000, samples_per_chunk, dtype=np.int16)
        
        pcm_data = PcmData(
            samples=audio_samples,
            format="int16",
        )
        
        # Process the audio chunk
        await detector.process_audio(pcm_data, user_id=user_id)
        
        # Small delay to simulate real-time processing
        await asyncio.sleep(chunk_duration)
    
    print("\nFinished processing audio")
    
    # Stop detection
    detector.stop()
    print("Turn detection stopped")


if __name__ == "__main__":
    asyncio.run(main())

