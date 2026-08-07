#!/usr/bin/env python3
"""
Smoke test: synthesize a few sentences with Palabra TTS and write a WAV file.

Unlike the full agent example, this does not join a Stream call – it drives
``palabra.TTS`` directly so you can confirm the API key works, listen to the
result, and see the latency you get.

It also shows the reason the plugin keeps one WebSocket open: ``start()`` pays
for the handshake and session ``init`` up front, and every sentence after that
reuses the same connection. The reported time-to-first-audio is therefore
synthesis latency only, not connection setup.

Usage::
    uv run tts_smoke.py
    uv run tts_smoke.py "Some other text to speak"

Requires ``PALABRA_API_KEY`` (see `.env.example`). The output is written to
``palabra_smoke.wav`` (mono, 16-bit PCM).
"""

import asyncio
import sys
import time
import wave

from dotenv import load_dotenv
from vision_agents.plugins import palabra

load_dotenv()

OUTPUT_PATH = "palabra_smoke.wav"

SENTENCES = [
    "The sun was setting over the mountains, casting long golden shadows.",
    "Birds were returning to their nests, filling the air with evening songs.",
    "A gentle breeze moved through the tall grass toward the horizon.",
]


async def main() -> None:
    sentences = sys.argv[1:] or SENTENCES

    tts = palabra.TTS()
    # Open the WebSocket before the first sentence so the handshake and the
    # session `init` are not counted in the latency below.
    await tts.start()

    audio = bytearray()
    try:
        for sentence in sentences:
            started = time.perf_counter()
            first_chunk_at = None
            chunks = 0

            async for chunk in tts.send_iter(sentence):
                if chunk.data is None:
                    continue
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter() - started
                audio += chunk.data.samples.tobytes()
                chunks += 1

            latency = f"{first_chunk_at * 1000:.0f}ms" if first_chunk_at else "n/a"
            print(f"{sentence[:48]!r}: {chunks} chunks, first audio in {latency}")
    finally:
        await tts.close()

    with wave.open(OUTPUT_PATH, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(tts.sample_rate)
        wav.writeframes(audio)

    duration = len(audio) / 2 / tts.sample_rate
    print(f"wrote {duration:.2f}s of audio to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
