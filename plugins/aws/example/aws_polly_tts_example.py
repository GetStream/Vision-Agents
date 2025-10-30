import asyncio
import os
from dotenv import load_dotenv

from vision_agents.plugins.aws import TTS
from vision_agents.core.tts.manual_test import manual_tts_to_wav


async def main():
    load_dotenv()
    tts = TTS(voice_id=os.environ.get("AWS_POLLY_VOICE", "Joanna"))
    await manual_tts_to_wav(tts, sample_rate=48000, channels=2)


if __name__ == "__main__":
    asyncio.run(main())
