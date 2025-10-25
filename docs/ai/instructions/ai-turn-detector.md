## Turn Detector

Here's a minimal example

```python

class MyTurnDetector(TurnDetector):
    async def process_audio(
        self,
        audio_data: PcmData,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
    
        # turn end
        self._emit_turn_event(TurnEvent.TURN_ENDED, event_data)

        # turn start
        self._emit_turn_event(TurnEvent.TURN_STARTED, event_data)


```