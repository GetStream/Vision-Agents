import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.core.utils.examples import recorded_call
from vision_agents.plugins import stream as acceleration

logging.basicConfig(level=logging.INFO)
# What this example prints is the point of it, and a request per HTTP call would bury that.
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

"""
Store a router config, then transcribe through it.

`routers/clinic/router.yaml` says who to try and what may happen to the audio afterwards,
but never which model. Naming the config stores the folder and the router picks. The clip
next to this file is streamed a chunk at a time, the way a call arrives.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it.
"""

HERE = Path(__file__).parent
AUDIO = HERE / "saturday_seven_thirty.wav"

CALLER = Participant(original=None, user_id="caller", id="caller")


async def main() -> None:
    router = acceleration.Router("clinic")
    async with router.stt.realtime() as stt:
        async with recorded_call(AUDIO) as chunks:
            async for chunk in chunks:
                await stt.process_audio(chunk, CALLER)

        # The last words of a turn are transcribed after they are spoken. Most of what
        # arrives before then is the provider's current guess, superseded by the next one.
        for event in await stt.output.collect(timeout=3.0):
            if isinstance(event, Transcript) and event.final:
                print(f"\n{event.model_name}: {event.text}\n")


if __name__ == "__main__":
    asyncio.run(main())
