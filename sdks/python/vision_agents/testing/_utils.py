import re
from typing import AsyncIterator

from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal


async def collect_simple_response(
    it: AsyncIterator[LLMResponseDelta | LLMResponseFinal],
) -> tuple[list[LLMResponseDelta], LLMResponseFinal]:
    """
    Iterate over LLM.simple_response() and collect the returned chunks.
    """
    deltas: list[LLMResponseDelta] = []
    final_response: LLMResponseFinal | None = None

    async for item in it:
        if isinstance(item, LLMResponseDelta):
            deltas.append(item)
        else:
            final_response = item

    if final_response is None:
        raise ValueError(
            "simple_response() ended without yielding an LLMResponseFinal chunk"
        )
    return deltas, final_response


def strip_code_fences(text: str) -> str:
    """Return ``text`` without a surrounding markdown code fence, if any."""
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    info_string, newline, body = cleaned.partition("\n")
    if newline and re.fullmatch(r"[\w-]*", info_string.strip()):
        cleaned = body
    return cleaned.strip()
