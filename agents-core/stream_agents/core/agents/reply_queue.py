from stream_agents.core.agents.conversation import Message
from stream_agents.core.llm.llm import LLMResponse


class ReplyQueue:
    """
    When a user interrupts the LLM, there are a few different behaviours that should be supported.
    1. Cancel/stop the audio playback, STT and LLM
    2. Pause and resume. Update context. Maybe reply the same
    3. Pause and refresh.

    Generating a reply, should write on this queue


    """

    def __init__(self, agent):
        self.agent = agent

    def pause(self):
        # TODO: some audio fade
        pass

    def _clear(self):
        pass

    async def say_text(self, text: str, user_id: str):
        # TODO: Stream and buffer
        await self.agent.conversation.add_message(Message(content=text, user_id=user_id))
        if self.agent.tts is not None:
            await self.agent.tts.send(text)

    async def send_audio(self, pcm):
        # TODO: stream & buffer
        if self.agent._audio_track is not None:
            await self.agent._audio_track.send_audio(pcm)
