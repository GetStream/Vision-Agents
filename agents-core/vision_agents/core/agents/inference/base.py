import abc

from vision_agents.core.edge.types import Participant


class InferenceFlow(abc.ABC):
    @abc.abstractmethod
    async def start(self): ...

    @abc.abstractmethod
    async def stop(self): ...

    @abc.abstractmethod
    async def interrupt(self): ...

    @abc.abstractmethod
    async def simple_response(
        self,
        text: str,
        participant: Participant,
        interrupt: bool = True,
    ) -> None:
        """Ask the LLM to reply to an injected system instruction.

        Args:
            text: Instruction or message to inject into the conversation.
            participant: Participant that the injected message is attributed to.
            interrupt: If True, preempt any in-flight LLM turn. If False, drop
                silently when a turn is already in flight (best-effort).
        """
        ...

    @abc.abstractmethod
    async def say(self, text: str, interrupt: bool = False) -> None:
        """Speak ``text`` directly through TTS, bypassing the LLM.

        Args:
            text: The utterance to speak.
            interrupt: If True, preempt any in-flight turn and clear the TTS
                pipeline first. If False (default), queue behind ongoing speech.
        """
        ...
