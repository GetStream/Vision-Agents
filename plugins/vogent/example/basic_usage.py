"""
Basic usage example for Vogent Turn Detection plugin.

This example demonstrates how to initialize and use the Vogent Turn Detection
plugin for detecting conversation turns in a voice AI application.
"""

import asyncio
import logging
import numpy as np
import urllib.request

from getstream.video.rtc.track_util import PcmData
from vision_agents.plugins.vogent import VogentTurnDetection
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.turn_detection import TurnStartedEvent, TurnEndedEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def example_with_real_audio():
    """Example using real audio from the vogent-turn sample."""
    
    # Initialize the turn detector
    detector = VogentTurnDetection(
        whisper_model_size="tiny",      # Fast and efficient
        vogent_threshold=0.5,            # Balanced turn detection
        silence_duration_ms=1000,        # 1 second of silence ends turn
    )
    
    # Register event handlers
    @detector.events.subscribe
    async def on_turn_started(event: TurnStartedEvent):
        participant_id = event.participant.user_id if event.participant else "unknown"
        logger.info(f"🎤 Turn started: {participant_id}")
    
    @detector.events.subscribe
    async def on_turn_ended(event: TurnEndedEvent):
        participant_id = event.participant.user_id if event.participant else "unknown"
        logger.info(f"✅ Turn ended: {participant_id}")
    
    # Start detection (loads models)
    logger.info("Starting turn detection...")
    await detector.start()
    logger.info("Turn detection started\n")
    
    # Download sample audio
    logger.info("Downloading sample audio...")
    audio_url = "https://storage.googleapis.com/voturn-sample-recordings/incomplete_number_sample.wav"
    try:
        import soundfile as sf
        urllib.request.urlretrieve(audio_url, "sample.wav")
        audio_samples, sample_rate = sf.read("sample.wav")
        logger.info(f"Loaded audio: {len(audio_samples)} samples at {sample_rate}Hz\n")
    except ImportError:
        logger.warning("soundfile not installed, using random audio instead")
        audio_samples = np.random.randn(48000).astype(np.float32) * 0.1
        sample_rate = 16000
    
    # Create participant and conversation
    participant = Participant(user_id="user123", original={})
    conversation = InMemoryConversation(
        instructions="You are a helpful assistant",
        messages=[]
    )
    
    # Add a previous message for context
    await conversation.send_message(
        role="assistant",
        user_id="assistant",
        content="What is your phone number"
    )
    
    # Convert to PcmData
    pcm_data = PcmData(
        samples=audio_samples,
        format="f32",
        sample_rate=sample_rate
    )
    
    # Process the audio
    logger.info("Processing audio...")
    await detector.process_audio(pcm_data, participant, conversation)
    
    # Wait for processing to complete
    await asyncio.sleep(2)
    
    logger.info("\nFinished processing audio")
    
    # Stop detection
    await detector.stop()
    logger.info("Turn detection stopped")


async def example_with_simulated_audio():
    """Example using simulated audio data."""
    
    # Initialize the turn detector with custom settings
    detector = VogentTurnDetection(
        whisper_model_size="tiny",
        vad_reset_interval_seconds=5.0,
        speech_probability_threshold=0.5,
        pre_speech_buffer_ms=200,
        silence_duration_ms=1200,  # Slightly longer for simulated audio
        max_segment_duration_seconds=8,
        vogent_threshold=0.5,
    )
    
    # Register event handlers
    event_log = []
    
    @detector.events.subscribe
    async def on_turn_started(event: TurnStartedEvent):
        participant_id = event.participant.user_id if event.participant else "unknown"
        logger.info(f"🎤 Turn started for {participant_id}")
        event_log.append("start")
    
    @detector.events.subscribe
    async def on_turn_ended(event: TurnEndedEvent):
        participant_id = event.participant.user_id if event.participant else "unknown"
        logger.info(f"✅ Turn ended for {participant_id}")
        event_log.append("end")
    
    # Start detection
    logger.info("Starting turn detection...")
    await detector.start()
    logger.info("Turn detection started\n")
    
    # Create participant and conversation
    participant = Participant(user_id="user456", original={})
    conversation = InMemoryConversation(
        instructions="You are a helpful assistant",
        messages=[]
    )
    
    # Simulate audio chunks (in production, this would be real audio from a microphone)
    sample_rate = 16000
    chunk_duration = 0.1  # 100ms chunks
    samples_per_chunk = int(sample_rate * chunk_duration)
    
    logger.info("Simulating audio processing...")
    logger.info("(In production, this would be real audio data)\n")
    
    # Process several chunks of simulated audio
    for i in range(50):  # ~5 seconds of audio
        # Create dummy audio data (mix of silence and "speech-like" noise)
        if 10 <= i <= 40:  # Simulate speech in the middle
            audio_samples = np.random.randn(samples_per_chunk).astype(np.float32) * 0.1
        else:  # Silence
            audio_samples = np.random.randn(samples_per_chunk).astype(np.float32) * 0.01
        
        pcm_data = PcmData(
            samples=audio_samples,
            format="f32",
            sample_rate=sample_rate
        )
        
        # Process the audio chunk
        await detector.process_audio(pcm_data, participant, conversation)
        
        # Small delay to simulate real-time processing
        await asyncio.sleep(0.01)  # Faster than real-time for demo
    
    # Wait for final processing
    await asyncio.sleep(2)
    
    logger.info(f"\nEvent log: {event_log}")
    logger.info("Finished processing audio")
    
    # Stop detection
    await detector.stop()
    logger.info("Turn detection stopped")


async def main():
    """Main entry point - runs the example."""
    
    # Try the real audio example first, fall back to simulated if needed
    try:
        logger.info("=" * 60)
        logger.info("Example 1: Real Audio Sample")
        logger.info("=" * 60)
        await example_with_real_audio()
    except Exception as e:
        logger.error(f"Real audio example failed: {e}")
        logger.info("\nFalling back to simulated audio example...")
    
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Simulated Audio")
    logger.info("=" * 60)
    await example_with_simulated_audio()


if __name__ == "__main__":
    asyncio.run(main())
