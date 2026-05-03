from typing import AsyncIterator

from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal


async def simple_response_final(
    it: AsyncIterator[LLMResponseDelta | LLMResponseFinal],
) -> LLMResponseFinal | None:
    """
    A helper to iterate over `llm.simple_response` and return only a final chunk.

    Args:
        it: An iterator over ``llm.simple_response``.

    Returns:
        LLMResponseFinal or ``None``.

    """
    async for chunk in it:
        if isinstance(chunk, LLMResponseFinal):
            return chunk
    return None
