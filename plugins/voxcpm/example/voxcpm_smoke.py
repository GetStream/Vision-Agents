"""Make one hosted VoxCPM request and write the streamed audio to WAV."""

import asyncio
import os
import sys
import time
import wave

from dotenv import load_dotenv
from getstream.video.rtc import PcmData
from vision_agents.plugins import voxcpm

load_dotenv()

OUTPUT_PATH: str = "voxcpm_smoke.wav"


async def main() -> None:
    text = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Hello from the Vision Agents VoxCPM plugin."
    )
    reference_audio = os.getenv("VOXCPM_REFERENCE_AUDIO") or None
    instance = voxcpm.TTS(
        api_key=os.getenv("MODELBEST_API_KEY"),
        model=os.getenv("MODELBEST_VOXCPM_MODEL_ID"),
        base_url=os.getenv("MODELBEST_VOXCPM_BASE_URL", voxcpm.DEFAULT_BASE_URL),
        ref_audio=reference_audio,
    )

    started = time.perf_counter()
    first_chunk_at: float | None = None
    pcm_chunks: list[PcmData] = []
    try:
        async for chunk in instance.send_iter(text):
            if chunk.data is not None:
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                pcm_chunks.append(chunk.data)
    finally:
        await instance.close()

    if not pcm_chunks:
        raise RuntimeError("VoxCPM returned no audio chunks")

    pcm = b"".join(chunk.to_bytes() for chunk in pcm_chunks)
    sample_rate = pcm_chunks[0].sample_rate
    channels = pcm_chunks[0].channels
    with wave.open(OUTPUT_PATH, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)

    elapsed = time.perf_counter() - started
    first_chunk_seconds = (first_chunk_at or started) - started
    duration = len(pcm) / (sample_rate * channels * 2)
    print(f"chunks: {len(pcm_chunks)}")
    print(f"sample rate: {sample_rate} Hz")
    print(f"time to first chunk: {first_chunk_seconds:.3f}s")
    print(f"duration: {duration:.2f}s")
    print(f"elapsed: {elapsed:.3f}s -> {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
