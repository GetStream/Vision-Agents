"""Scenario variations: reworded briefs that keep every fact intact."""

import dataclasses
import json
import logging

from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.llm import LLM

from ._scenario import Scenario
from ._utils import collect_simple_response, parse_json_object

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You rewrite test scenarios for conversational AI agents.\n"
    "Given a scenario, produce alternative wordings of its goal and constraints.\n\n"
    "Rules:\n"
    "- Keep every fact, number, name, date and requirement exactly as in the original.\n"
    "- Change only phrasing, sentence structure and tone.\n"
    "- Each variation must differ from the original and from the other variations.\n"
    "- Keep the same number of constraints, in the same order.\n"
    "- Respond with ONLY a JSON object in this exact format:\n"
    '  {"variations": [{"goal": "...", "constraints": ["...", "..."]}, ...]}\n'
    "- Do NOT include any other text before or after the JSON."
)


async def generate_variations(
    llm: LLM, scenario: Scenario, count: int
) -> list[Scenario]:
    """Return ``count`` scenarios: the original first, then reworded copies.

    Only ``goal`` and ``constraints`` are reworded. Persona and context are
    facts and are copied unchanged.

    Args:
        llm: LLM used to produce the rewordings. Use a fresh instance.
        scenario: The scenario to vary.
        count: Total number of scenarios to return, including the original.

    Raises:
        ValueError: If ``count`` is below 1 or the model returns unusable output.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if count == 1:
        return [scenario]

    llm.set_instructions(_SYSTEM_PROMPT)
    llm.set_conversation(InMemoryConversation(instructions=_SYSTEM_PROMPT, messages=[]))

    original = {"goal": scenario.goal, "constraints": scenario.constraints}
    prompt = (
        f"Produce exactly {count - 1} variation(s) of this scenario.\n\n"
        f"Original:\n{json.dumps(original, indent=2)}"
    )
    _, response = await collect_simple_response(llm.simple_response(text=prompt))
    data = parse_json_object(response.text)

    raw = data.get("variations")
    if not isinstance(raw, list) or len(raw) < count - 1:
        raise ValueError(
            f"Expected {count - 1} variation(s), model returned: {response.text[:200]!r}"
        )

    variants = [scenario]
    for item in raw[: count - 1]:
        if not isinstance(item, dict):
            raise ValueError(f"Variation must be an object, got: {item!r}")
        goal = item.get("goal")
        constraints = item.get("constraints", scenario.constraints)
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError(f"Variation is missing a goal: {item!r}")
        if not isinstance(constraints, list) or not all(
            isinstance(c, str) for c in constraints
        ):
            raise ValueError(f"Variation constraints must be strings: {item!r}")
        if len(constraints) != len(scenario.constraints):
            raise ValueError(
                f"Variation must keep {len(scenario.constraints)} constraint(s), got {len(constraints)}"
            )
        variants.append(
            dataclasses.replace(scenario, goal=goal, constraints=list(constraints))
        )
    return variants
