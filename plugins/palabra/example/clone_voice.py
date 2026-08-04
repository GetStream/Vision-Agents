#!/usr/bin/env python3
"""
Clone a voice with Palabra, then speak with it.

Palabra needs at least 30 seconds of clean, single-speaker audio to clone from.
Pass your own recording, or pass nothing and the script records a sample by
reading a passage with Palabra's stock voice — which makes the example
self-contained, at the cost of cloning a synthetic voice rather than a human one.

Usage::
    uv run clone_voice.py                      # synthesize a sample, then clone it
    uv run clone_voice.py my_recording.wav     # clone from your own recording
    uv run clone_voice.py --keep               # don't delete the voice afterwards

Requires ``PALABRA_API_KEY`` (see `.env.example`). Cloned voices count against
your account quota, so the script deletes the voice on the way out unless you
pass ``--keep``.

Only clone someone's voice with their explicit consent.
"""

import asyncio
import sys
import wave
from pathlib import Path

from dotenv import load_dotenv
from vision_agents.plugins import palabra

load_dotenv()

SAMPLE_PATH = Path("palabra_voice_sample.wav")
OUTPUT_PATH = Path("palabra_cloned_voice.wav")

# ~40 seconds of speech: comfortably over Palabra's 30 second minimum.
SAMPLE_SCRIPT = [
    "Every morning the harbour wakes slowly, one boat at a time.",
    "The fishermen speak in short sentences, mostly about the weather.",
    "By seven the market is loud, and the gulls have taken the high walls.",
    "A woman sells coffee from a cart she has pushed to the same corner for years.",
    "She knows every regular by the way they hold their cup.",
    "Later the tide turns and the water goes flat and grey.",
    "Children run along the pier, daring each other to look over the edge.",
    "In the evening the boats come back heavier than they left.",
    "Someone always sings on the way in, badly, and nobody minds.",
    "The harbour sleeps again before the town does.",
]


async def write_sample(tts: palabra.TTS, path: Path) -> float:
    """Synthesize the passage into a WAV file and return its duration."""
    audio = bytearray()
    for line in SAMPLE_SCRIPT:
        async for chunk in tts.send_iter(line):
            if chunk.data is not None:
                audio += chunk.data.samples.tobytes()

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(tts.sample_rate)
        wav.writeframes(audio)
    return len(audio) / 2 / tts.sample_rate


async def speak(voice_id: str, path: Path) -> float:
    """Say a line with the cloned voice and write it to ``path``."""
    tts = palabra.TTS(voice_id=voice_id, deaccent_strength=0.7)
    audio = bytearray()
    try:
        async for chunk in tts.send_iter(
            "This is my cloned voice, generated from a short audio sample."
        ):
            if chunk.data is not None:
                audio += chunk.data.samples.tobytes()
        rate = tts.sample_rate
    finally:
        await tts.close()

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(audio)
    return len(audio) / 2 / rate


async def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--keep"]
    keep = "--keep" in sys.argv
    sample = Path(args[0]) if args else SAMPLE_PATH

    async with palabra.Voices() as voices:
        quota = await voices.limits()
        print(f"voice quota: {quota.total}/{quota.limit} used, {quota.remaining} left")
        if quota.remaining < 1:
            print("no quota left — delete a voice first (voices.delete(voice_id))")
            return

        if not args:
            tts = palabra.TTS()
            try:
                duration = await write_sample(tts, sample)
            finally:
                await tts.close()
            print(f"recorded {duration:.1f}s sample -> {sample}")

        print("cloning (this takes a moment)...")
        voice = await voices.clone("Vision Agents demo", sample, lang_code="en")
        print(f"voice {voice.voice_id} is {voice.processing_status}")
        if voice.warnings:
            print(f"warnings: {voice.warnings}")

        try:
            duration = await speak(voice.voice_id, OUTPUT_PATH)
            print(f"spoke {duration:.1f}s with the cloned voice -> {OUTPUT_PATH}")
        finally:
            if keep:
                print(f"keeping voice {voice.voice_id}")
            else:
                await voices.delete(voice.voice_id)
                print(f"deleted voice {voice.voice_id}")


if __name__ == "__main__":
    asyncio.run(main())
