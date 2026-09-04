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
