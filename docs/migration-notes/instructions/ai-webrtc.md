Joining a call using WebRTC works likes this:

```python

call = stream_client.video.call("default", str(uuid.uuid4()))

async with call.join("transcription-client", timeout=6.0) as connection:
    async def on_audio_packet(event):
        # do something with the audio event
        pass

    await on_event(connection, "audio_packet", on_audio_packet)

    async for event in connection:
        # this will yield until the call ends or if call is called
        pass
```

The `call.join` method loads python code under `getstream/video/rtc` package which requires special python dependencies.

The `pyproject.toml` groups all webrtc related dependencies under the `webrtc` group, when adding a new dependency that is only needed by the `rtc` module add that dependency to the `webrtc` group.
