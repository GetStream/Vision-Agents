import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.realtime import (
    RealtimeAgentSpeechEnded,
    RealtimeAgentTranscript,
    RealtimeAudioOutput,
    RealtimeUserTranscript,
)
from vision_agents.core.utils.examples import recorded_call
from vision_agents.plugins import stream as acceleration

logging.basicConfig(level=logging.INFO)
# What this example prints is the point of it, and a request per HTTP call would bury that.
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

"""
Store a router config, then hold a conversation through it.

`routers/frontdesk/router.yaml` asks for one native audio model, quick to answer, that
writes down what it hears and says, but never which model. Naming the config stores the
folder and the router picks. The clip next to this file is streamed a chunk at a time, the
way a call arrives, and the model's reply is written next to it because an example has no
speaker.

There is no transcriber, turn detector or voice on this side of the socket. The model
hears the caller stop, answers in its own voice, and the transcripts are its own account of
the exchange.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it.
"""

HERE = Path(__file__).parent
AUDIO = HERE / "saturday_seven_thirty.wav"
SPOKEN = HERE / "spoken.wav"

CALLER = Participant(original=None, user_id="caller", id="caller")

# How long to wait for the model to finish its reply once the caller has stopped.
PATIENCE = 30.0


async def main() -> None:
    router = acceleration.Router("frontdesk")
    session = router.sts.realtime(
        instructions="You take restaurant bookings. Confirm what you heard in one short sentence.",
    )

    async with session as sts:
        stopped = time.perf_counter()
        async with recorded_call(AUDIO) as chunks:
            async for chunk in chunks:
                await sts.process_audio(chunk, CALLER)
                stopped = time.perf_counter()

        spoken = await asyncio.wait_for(replied(sts, stopped), PATIENCE)

    if spoken is None:
        raise RuntimeError("the model said nothing")

    await asyncio.to_thread(SPOKEN.write_bytes, spoken.to_wav_bytes())
    print(f"{SPOKEN.name}: {spoken.duration:.1f}s at {spoken.sample_rate}Hz")


async def replied(sts: acceleration.STS, stopped: float) -> Optional[PcmData]:
    """Read the model's reply off the session, printing the transcripts as they settle."""
    spoken: Optional[PcmData] = None
    first = 0.0

    async for event in sts.output:
        if isinstance(event, RealtimeAudioOutput):
            first = first or time.perf_counter() - stopped
            if spoken is None:
                spoken = event.data.copy()
            else:
                spoken.append(event.data)
        elif isinstance(event, RealtimeUserTranscript) and event.mode == "final":
            print(f"\nheard: {event.text}")
        elif isinstance(event, RealtimeAgentTranscript) and event.mode == "final":
            print(f"said:  {event.text}")
        elif isinstance(event, RealtimeAgentSpeechEnded):
            print(f"  first audio {first * 1000:.0f}ms after the caller stopped\n")
            return spoken
    return spoken


if __name__ == "__main__":
    asyncio.run(main())
