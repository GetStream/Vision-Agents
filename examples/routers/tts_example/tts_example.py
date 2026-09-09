import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData
from vision_agents.plugins import stream as acceleration

logging.basicConfig(level=logging.INFO)
# What this example prints is the point of it, and a request per HTTP call would bury that.
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

"""
Store a router config, then speak through it.

`routers/switchboard/router.yaml` says the voice has to be English and quick to start, but
never which model it is. Naming the config stores the folder and the router picks. Both
lines go over one socket that stays open, and the audio is written next to this file
because an example has no speaker.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it.
"""

HERE = Path(__file__).parent
SPOKEN = HERE / "spoken.wav"

LINES = [
    "Your table for four is booked for Saturday at seven thirty.",
    "You are on the patio, with a high chair and a note about the peanut allergy.",
]


async def main() -> None:
    router = acceleration.Router("switchboard")
    spoken: Optional[PcmData] = None

    async with router.tts.realtime() as tts:
        for line in LINES:
            asked = time.perf_counter()
            first = 0.0

            async for chunk in tts.send_iter(line):
                # The last chunk of an utterance carries no audio, only the news that
                # there is no more of it.
                if chunk.data is None:
                    continue

                first = first or time.perf_counter() - asked
                if spoken is None:
                    spoken = chunk.data.copy()
                else:
                    spoken.append(chunk.data)

            print(f"\n{line}\n  first audio after {first * 1000:.0f}ms\n")

    if spoken is None:
        raise RuntimeError("the router said nothing")

    await asyncio.to_thread(SPOKEN.write_bytes, spoken.to_wav_bytes())
    print(f"{SPOKEN.name}: {spoken.duration:.1f}s at {spoken.sample_rate}Hz")


if __name__ == "__main__":
    asyncio.run(main())
