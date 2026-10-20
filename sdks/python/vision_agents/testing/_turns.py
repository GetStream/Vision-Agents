from vision_agents.core.agents.conversation import Conversation
from vision_agents.core.agents.inference.llm_turn import LLMTurn
from vision_agents.core.edge.types import Participant
from vision_agents.core.events import EventManager
from vision_agents.core.llm.events import LLMResponseFinalEvent
from vision_agents.core.llm.llm import LLM


async def answer(
    llm: LLM, conversation: Conversation, participant: Participant, text: str
) -> None:
    """What the agent does with a participant's words: keep them, then ask the LLM.

    The LLM's reply is kept as "Ok", so the next request carries a known history.
    """
    await conversation.upsert_message(
        role="user", user_id=participant.user_id, content=text, participant=participant
    )
    events = EventManager()
    events.register(LLMResponseFinalEvent)
    turn = LLMTurn(
        transcript=text,
        participant=participant,
        events=events,
        speaker=conversation.speaker_label(participant),
    )
    turn.start(llm)
    await turn.stream.collect()
    await conversation.send_message(role="assistant", user_id="agent", content="Ok")
