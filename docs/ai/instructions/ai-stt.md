## STT

```python
from vision_agents.core import stt
from vision_agents.core.stt.events import TranscriptResponse

class MySTT(stt.STT):

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[MyClient] = None,
    ):
        super().__init__(provider_name="my_stt")
        # be sure to allow the passing of the client object
        # if client is not passed, create one
        # pass the most common settings for the client in the init (like api key)


    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        parts = self.client.stt(pcm_data, stream=True)
        full_text = ""
        for part in parts:
            response = TranscriptResponse(
                confidence=0.9,
                language='en',
                processing_time_ms=300,
                audio_duration_ms=2000,
                other={}
            )
            # parts that aren't finished
            self._emit_partial_transcript_event(part, participant, response)
            full_text += part

        # the full text
        self._emit_transcript_event(full_text, participant, response)

```

## Testing the STT

A good example of testing the STT can be found in plugins/fish/tests/test_fish_stt.py

## PCM / Audio management

Use `PcmData` and other utils available from the `getstream.video.rtc.track_util` module.
Do not write code that directly manipulates PCM, use the audio utilities instead.

## Turn keeping

If your STT supports Turn detection/turn events do the following

```
class MySTT(stt.STT):
    turn_detection: bool = True
    
    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        ...
        self._emit_turn_ended_event(participant=participant, eager_end_of_turn=eager_end_of_turn)
        ...
```
