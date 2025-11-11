## Turn Detector

Here's a minimal example

```python

class MyTurnDetector(TurnDetector):
    async def process_audio(
        self,
        audio_data: PcmData,
        participant: Participant,
        conversation: Optional[Conversation],
    ) -> None:
    
        self._emit_start_turn_event(TurnStartedEvent(participant=participant))
        self._emit_end_turn_event(participant=participant, confidence=0.7)

    def start(self):
        super().start()
        # Any custom model loading/ or other heavy prep steps go here
        
    def stop(self):
        super().stop()
        # cleanup time. start and stop are optional

```

### Loading models

Loading models blocks async work. Here's an example of how to properly load a model

```
async def _prepare_smart_turn(self):
    await ensure_model(SMART_TURN_ONNX_PATH, SMART_TURN_ONNX_URL)
    self._whisper_extractor = await asyncio.to_thread(WhisperFeatureExtractor, chunk_length=8)
    # Load ONNX session in thread pool to avoid blocking event loop
    self.smart_turn = await asyncio.to_thread(build_session, SMART_TURN_ONNX_PATH)
```

### Testing turn detection

An example test suite for turn detection can be found in `smart_turn/tests/test_smart_turn.py`