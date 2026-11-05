"""Synthesize a WAV file using SIXTYDB_API_KEY and SIXTYDB_VOICE_ID."""

import asyncio
import wave

from vision_agents.plugins import sixtydb


async def main() -> None:
    provider = sixtydb.TTS()
    try:
        with wave.open("speech.wav", "wb") as output:
            output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            async for chunk in provider.send_iter("Hello from Vision Agents."):
                if chunk.data is not None:
                    output.writeframes(chunk.data.samples.tobytes())
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
