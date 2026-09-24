"""LLM-driven simulated user for multi-turn simulations."""

import asyncio
import logging

from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.llm import LLM

from ._scenario import Scenario
from ._utils import collect_simple_response, parse_json_object

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are role-playing a human user talking to an AI agent in a text chat.\n"
    "Stay in character at all times. Never reveal that you are an AI or that "
    "this is a test.\n\n"
    "{brief}\n\n"
    "Rules:\n"
    "- Write one short message per turn, the way a real person types in chat.\n"
    "- Only share facts when they are relevant or when the agent asks for them.\n"
    "- Keep pursuing your goal until the agent has actually accomplished it.\n"
    '- Set "done" to true only once your goal is fully achieved, or once it is '
    "clear the agent cannot achieve it.\n"
    "- Respond with ONLY a JSON object in this exact format:\n"
    '  {{"message": "<what you say next>", "done": false}}\n'
    "- Do NOT include any other text before or after the JSON."
)

_OPENING_PROMPT = "The conversation starts now. Write your opening message."

_REPLY_PROMPT = (
    "The agent replied:\n{reply}\n\n"
    'Write your next message, or set "done" to true if your goal has been '
    "achieved or you want to end the conversation."
)

_EMPTY_REPLY = "(the agent did not reply)"


class SimulatedUserError(Exception):
    """The simulated user could not produce its next message."""


class SimulatedUser:
    """Plays the user side of a conversation from a scenario brief.

    The LLM receives the scenario as its system prompt, reads each agent
    reply and decides both the next line and whether the goal has been met.

    Args:
        llm: LLM that plays the user. Use a fresh instance per conversation.
        scenario: Brief describing persona, facts, goal and constraints.
        max_turns: Hard cap on the number of user messages.
        turn_timeout: Seconds to wait for the LLM to produce a message.
    """

    def __init__(
        self,
        llm: LLM,
        scenario: Scenario,
        max_turns: int = 10,
        turn_timeout: float = 60.0,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if turn_timeout <= 0:
            raise ValueError("turn_timeout must be positive")
        self._llm = llm
        self._scenario = scenario
        self._max_turns = max_turns
        self._turn_timeout = turn_timeout
        self._conversation: InMemoryConversation | None = None
        self._turns_taken = 0
        self._done = False
        self._started = False

    async def start(self) -> None:
        """Prime the LLM with the scenario brief."""
        if self._started:
            return
        instructions = _SYSTEM_PROMPT.format(brief=self._scenario.brief)
        self._llm.set_instructions(instructions)
        self._conversation = InMemoryConversation(
            instructions=instructions, messages=[]
        )
        self._llm.set_conversation(self._conversation)
        self._started = True

    @property
    def turns_taken(self) -> int:
        """Number of messages sent to the agent so far."""
        return self._turns_taken

    @property
    def done(self) -> bool:
        """Whether the user decided the conversation is over."""
        return self._done

    async def next_message(self, agent_reply: str | None) -> str | None:
        """Return the next user message, or ``None`` when the conversation is over.

        The conversation ends when the user declares the goal met or when
        ``max_turns`` messages have been sent, whichever comes first.

        Args:
            agent_reply: The agent's last reply, or ``None`` for the opening turn.

        Raises:
            SimulatedUserError: If the LLM fails, times out or returns malformed output.
        """
        if not self._started:
            await self.start()
        if self._done or self._turns_taken >= self._max_turns:
            return None

        if agent_reply is None:
            prompt = _OPENING_PROMPT
        else:
            prompt = _REPLY_PROMPT.format(reply=agent_reply or _EMPTY_REPLY)

        if self._conversation is not None:
            await self._conversation.send_message(
                role="user", user_id="simulation", content=prompt
            )
        try:
            _, response = await asyncio.wait_for(
                collect_simple_response(self._llm.simple_response(text=prompt)),
                timeout=self._turn_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise SimulatedUserError(
                f"Simulated user did not respond within {self._turn_timeout}s"
            ) from exc
        except Exception as exc:
            logger.exception("Simulated user LLM failed")
            raise SimulatedUserError(f"Simulated user LLM failed: {exc}") from exc
        if self._conversation is not None and response.text:
            await self._conversation.send_message(
                role="assistant", user_id="simulated-user", content=response.text
            )

        try:
            data = parse_json_object(response.text)
        except ValueError as exc:
            raise SimulatedUserError(
                f"Simulated user returned invalid output: {exc}"
            ) from exc

        message = data.get("message")
        done = data.get("done", False)
        if not isinstance(message, str) or not isinstance(done, bool):
            raise SimulatedUserError(
                f"Simulated user returned invalid output: {response.text[:200]!r}"
            )

        if done or not message.strip():
            self._done = True
            return None

        self._turns_taken += 1
        return message
