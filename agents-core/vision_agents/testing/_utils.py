import json
from typing import Any, AsyncIterator

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


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from LLM output, tolerating a Markdown code fence.

    Raises:
        ValueError: If the text is not a JSON object.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Expected a JSON object, got: {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got: {text[:200]!r}")
    return data
