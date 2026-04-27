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

    assert final_response is not None
    return deltas, final_response
