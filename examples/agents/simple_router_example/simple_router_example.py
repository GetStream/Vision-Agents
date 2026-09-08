import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
from vision_agents.plugins import stream as acceleration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# What this example prints is the point of it, and a request per poll of a transcription
# job would bury that.
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

"""
What a stored router config says about transcription, and what happens when it asks for
something nothing offers.

Every other example points a `Router` at a config somebody else wrote. This writes them:
`routers/` holds one YAML file per config, `sync_routers` stores the directory, and the
options inside are the interesting half - a priority list of who to try, a data policy
about what may happen to the audio afterwards, and per-provider settings the shared
vocabulary has no word for.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it.
"""

HERE = Path(__file__).parent

# Eight seconds of somebody booking a table, from the repository's own test assets, so the
# example transcribes something whose transcript can be checked by reading it.
AUDIO = HERE.parents[2] / "tests" / "test_assets" / "saturday_seven_thirty.wav"


async def store_the_configs() -> None:
    """Store every config in `routers/`, which is where routing that matters belongs.

    The same bargain as an agent's instructions being a file: a priority list written at a
    call site is one nobody reviews, and a data policy is exactly the kind of decision that
    should be read by a second person before it ships.
    """
    for config in await acceleration.sync_routers(HERE / "routers"):
        # to_dict leaves out what the file did not say, so this is the config as written
        # rather than every option transcription has.
        print(f"  {config.name}: {config.stt.to_dict()}")


async def transcribe_a_recording() -> None:
    """Transcribe through the podcast config and say who ended up serving it.

    Nothing here names a model. The config asked for diarization, word timings and a
    profanity filter, and only the batch half of Deepgram can express all three, so that
    is what the router picked - which is the difference between asking for a capability
    and asking for a vendor.
    """
    router = acceleration.Router("podcast")
    transcript = await router.stt.recording(AUDIO)

    print(f"  served by {transcript.provider}/{transcript.model}")
    print(f"  {transcript.text}")
    print(f"  {len(transcript.words)} words timed, speakers: {transcript.speakers}")


async def reconfigure_the_clinic() -> None:
    """Rewrite the clinic config's transcription from code rather than from YAML.

    `configure_stt` writes the same block `clinic.yaml` describes and leaves the other
    three modalities as they were stored, since saying how something is heard is not a
    statement about how it speaks.
    """
    router = acceleration.Router("clinic")
    config = await router.configure_stt(
        providers=["parakeet", "deepgram/flux-general-en"],
        data_policy={"allow_training": False, "retention": "none"},
        keyterms=["amoxicillin", "ondansetron"],
    )

    print(f"  clinic now tries {config.stt.providers} in that order")


async def ask_for_something_nothing_offers() -> None:
    """The two refusals, both of which arrive while the config is being written.

    This is the point of storing a config rather than passing keywords: a request nothing
    can serve is a thing to find out about now, from an exception with a message, rather
    than at three in the morning from a call that failed to start.
    """
    router = acceleration.Router("clinic")

    try:
        await router.configure_stt(mode="smart", diarize=True)
    except RuntimeError as refusal:
        print(f"  smart + diarize: {refusal}")

    try:
        await router.configure_stt(providers=["deepgramm"])
    except RuntimeError as refusal:
        print(f"  a misspelt vendor: {refusal}")


async def main() -> None:
    print("\nstored from routers/:")
    await store_the_configs()

    print("\ntranscribed:")
    await transcribe_a_recording()

    print("\nreconfigured:")
    await reconfigure_the_clinic()

    print("\nrefused:")
    await ask_for_something_nothing_offers()
    print()


if __name__ == "__main__":
    asyncio.run(main())
