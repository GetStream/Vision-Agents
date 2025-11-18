# Realtime LLM Plugin Development

Here is a minimal example of developing a new Realtime LLM.

```python
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.events import LLMResponseCompletedEvent, LLMResponseChunkEvent
from vision_agents.core.processors import Processor
from vision_agents.core.llm import realtime

class MyRealtime(realtime.Realtime):
    def __init__(self, model: str, client: Optional[ClientType]):
        # it should be possible to pass the client (makes it easier for users to customize things)
        # settings that are common to change, like model should be specified as well
        super().__init__()
        self.model = model
        self.client = client

    async def connect(self):
        # create the websocket or webrtc connection to the realtime LLM
        pass

    async def _handle_events(self):
        # handle the events from the connect method

        # when receiving audio do this
        audio_event = RealtimeAudioOutputEvent(
            plugin_name="gemini",
            audio_data=audio_content,
            sample_rate=24000
        )
        self.events.send(audio_event)

        await self.output_track.write(audio_content)

        # for transcriptions...
        # TODO document this
        pass

    async def close(self):
        pass

    # native method wrapped. wrap the native method, every llm has its own name for this
    # openai calls it create response, anthropic create message. so the name depends on your llm
    async def mynativemethod(self, *args, **kwargs):

        # some details to get right here...
        # ensure conversation history is maintained. typically by passing it ie:
        if self._instructions:
            kwargs["system"] = [{"text": self._instructions}]

        response_iterator = await self.client.mynativemethod(self, *args, **kwargs)

        # while receiving streaming do this
        total_text = ""
        for chunk in response_iterator:
            self.events.send(LLMResponseChunkEvent(
                    plugin_name="gemini",
                    content_index=0,
                    item_id="",
                    output_index=0,
                    sequence_number=0,
                    delta=chunk.text,
                ))
            total_text += chunk.text

        llm_response = LLMResponseEvent(response_iterator, total_text)
        # and when completed
        self.events.send(LLMResponseCompletedEvent(
            plugin_name="gemini",
            original=llm_response.original,
            text=llm_response.text
        ))

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[Processor]] = None,
        participant: Participant = None,
    ):
        # call the LLM with the given text
        # be sure to use the streaming version
        self.mynativemethod(...)

    async def simple_audio_response(self, pcm: PcmData):
        # respond to this audio
        pass

```

## Things to get right

* Use the streaming API/version in your native method
* Have 1 endpoint wrap the native method (with *args, **kwargs)
* Simple response is the standardized way. this should call mynativemethod
* Messages are standardized in _normalize_message

## Other examples

If you need more examples look in

- gemini_llm.py
- aws_llm.py (AWS Bedrock implementation)

## PCM / Audio management

Use `PcmData` and other utils available from the `getstream.video.rtc.track_util` module. Do not write code that directly manipulates PCM, use the audio utilities instead.
