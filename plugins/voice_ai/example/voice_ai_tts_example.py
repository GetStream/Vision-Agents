"""
Voice.ai TTS Example

This example demonstrates Voice.ai TTS integration with Vision Agents.

Requirements:
- VOICE_AI_API_KEY environment variable
- VOICE_AI_VOICE_ID environment variable
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core.tts.manual_test import manual_tts_to_wav
from vision_agents.plugins import voice_ai

logger = logging.getLogger(__name__)

load_dotenv()


async def main() -> None:
    tts = voice_ai.TTS()
    wav_path = await manual_tts_to_wav(tts, sample_rate=48000, channels=2)
    logger.info("Wrote WAV file: %s", wav_path)
    await tts.close()


if __name__ == "__main__":
    asyncio.run(main())
