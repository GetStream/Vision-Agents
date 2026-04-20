## STT

```python
from vision_agents.core import stt
from vision_agents.core.stt import Transcript, TranscriptResponse

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
            # partials — mode depends on provider semantics:
            #   "replacement" → each partial resends the full running transcript
            #   "delta"       → each partial is a new chunk appended to prior text
            self.output.send_nowait(
                Transcript(
                    text=part,
                    participant=participant,
                    response=response,
                    mode="delta",
                )
            )
            full_text += part

        # the full text
        self.output.send_nowait(
            Transcript(
                text=full_text,
                participant=participant,
                response=response,
                mode="final",
            )
        )

```

## Testing the STT

A good example of testing the STT can be found in plugins/fish/tests/test_fish_stt.py

## PCM / Audio management

Use `PcmData` and other utils available from the `getstream.video.rtc.track_util` module.
Do not write code that directly manipulates PCM, use the audio utilities instead.

## Turn keeping

If your STT supports Turn detection/turn events do the following

```
from vision_agents.core.turn_detection import TurnEnded, TurnStarted

class MySTT(stt.STT):
    turn_detection: bool = True

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        ...
        self.output.send_nowait(
            TurnEnded(
                participant=participant,
                confidence=0.9,
                eager=eager_end_of_turn,
            )
        )
        ...
```
