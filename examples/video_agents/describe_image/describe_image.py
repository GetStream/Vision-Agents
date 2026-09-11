import asyncio
import os

from dotenv import load_dotenv
from vision_agents.core.llm.llm import ImageContent, LLMResponseDelta
from vision_agents.plugins import stream

load_dotenv()

# The capability shortcut routes to whatever vision model the tier prefers. Pin a
# provider/model here (or in STREAM_ACCELERATION_LLM) if the preferred one is rate limited.
TARGET = os.getenv("STREAM_ACCELERATION_LLM", "vlm")

"""
The smallest thing that sends an image to the router and reads back a description.

`responses.create` accepts bytes or URLs. The `vlm` alias selects an image-capable model
before the socket opens. `llm-fast` keeps its existing voice policy.

Needs a router: see acceleration/README.md, then set STREAM_ACCELERATION_URL and
STREAM_ACCELERATION_CUSTOMER_ID (a repo-root .env works).
"""

IMAGE_URL = "https://getstream.io/assets/images/Zf0EOs68zyqdRpyr_Large-Feature-Gaming-c1406c80.webp"


async def main() -> None:
    async with stream.LLM(target=TARGET) as llm:
        async for chunk in llm.responses.create(
            "Describe this image in one short sentence.",
            images=[ImageContent(url=IMAGE_URL)],
        ):
            if isinstance(chunk, LLMResponseDelta) and chunk.delta:
                print(chunk.delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
