## Gemini Live Speech-to-Speech Plugin

Google Gemini Live Speech-to-Speech (STS) plugin for GetStream. It connects a realtime Gemini Live session to a Stream video call so your assistant can speak and listen in the same call.

### Installation

```bash
pip install getstream-plugins-gemini
```

### Requirements

- **Python**: 3.10+
- **Dependencies**: `getstream[webrtc"]`, `getstream-plugins-common`, `google-genai`
- **API key**: `GOOGLE_API_KEY` or `GEMINI_API_KEY` set in your environment

### Quick Start

Below is a minimal example that attaches the Gemini Live output audio track to a Stream call and streams microphone audio into Gemini. The assistant will speak back into the call, and you can also send text messages to the assistant.

```python
import asyncio
import os

from getstream import Stream
from getstream.plugins.gemini.live import GeminiLive
from getstream.video import rtc
from getstream.video.rtc.track_util import PcmData


async def main():
    # Ensure your key is set: export GOOGLE_API_KEY=... (or GEMINI_API_KEY)
    gemini = GeminiLive(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-live-2.5-flash-preview",
    )

    client = Stream.from_env()
    call = client.video.call("default", "your-call-id")

    async with await rtc.join(call, user_id="assistant-bot") as connection:
        # Route Gemini's synthesized speech back into the call
        await connection.add_tracks(audio=gemini.output_track)

        # Forward microphone PCM frames to Gemini in realtime
        @connection.on("audio")
        async def on_audio(pcm: PcmData):
            await gemini.send_audio_pcm(pcm, target_rate=48000)

        # Optionally send a kick-off text message
        await gemini.send_text("Give a short greeting to the participants.")

        # Keep the session running
        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
```

Optional: forward remote participant video frames to Gemini for multimodal context:

```python
# Forward remote video frames to Gemini (optional)
@connection.on("track_added")
async def _on_track_added(track_id, kind, user):
    if kind == "video" and connection.subscriber_pc:
        track = connection.subscriber_pc.add_track_subscriber(track_id)
        if track:
            await gemini._watch_video_track(track)
```

For a full runnable example, see `examples/gemini_live/main.py`.

### Features

- **Bidirectional audio**: Streams microphone PCM to Gemini, and plays Gemini speech into the call using `output_track`.
- **Video frame forwarding**: Sends remote participant video frames to Gemini Live for multimodal understanding. Use `start_video_sender` with a remote `MediaStreamTrack`.
- **Text messages**: Use `send_text` to add text turns directly to the conversation.
- **Barge-in (interruptions)**: When the user starts speaking, current playback is interrupted so Gemini can focus on the new input. Playback automatically resumes after brief silence.
- **Auto resampling**: `send_audio_pcm` will resample input frames to the target rate when needed.
- **Events**: Subscribe to `"audio"` for synthesized audio chunks and `"text"` for assistant text.

### API Overview

- **`GeminiLive(api_key: str | None = None, model: str = "gemini-live-2.5-flash-preview", config: LiveConnectConfigDict | None = None)`**: Create a new Gemini Live session. If `api_key` is not provided, the plugin reads `GOOGLE_API_KEY` or `GEMINI_API_KEY` from the environment.
- **`output_track`**: An `AudioStreamTrack` you can publish in your call via `add_tracks(audio=...)`.
- **`await send_text(text: str)`**: Send a user text message to the current turn.
- **`await send_audio_pcm(pcm: PcmData, target_rate: int = 48000)`**: Stream PCM frames to Gemini. Frames are converted to the required format and resampled if necessary.
- **`await wait_until_ready(timeout: float | None = None) -> bool`**: Wait until the underlying live session is connected.
- **`await interrupt_playback()` / `resume_playback()`**: Manually stop or resume synthesized audio playback. Useful if you want to manage barge-in behavior yourself.
- **`await start_video_sender(track: MediaStreamTrack, fps: int = 1)`**: Start forwarding video frames from a remote `MediaStreamTrack` to Gemini Live at the given frame rate.
- **`await stop_video_sender()`**: Stop the background video sender task, if running.
- **`await close()`**: Close the session and background tasks.

### Environment Variables

- **`GOOGLE_API_KEY` / `GEMINI_API_KEY`**: Gemini API key. One must be set.
- **`GEMINI_LIVE_MODEL`**: Optional override for the model name if you need a different variant.

### Notes on Interruptions

- **How it works**: The plugin detects user speech activity in incoming PCM and interrupts any ongoing playback. After a short period of silence, playback is enabled again so the assistant can speak.
- **Why it matters**: This enables natural barge-in experiences, where users can cut off the assistant mid-sentence and ask follow-up questions.

### Troubleshooting

- **No audio playback**: Ensure you publish `output_track` to your call and the call is subscribed to the assistant’s audio.
- **No responses**: Verify `GOOGLE_API_KEY`/`GEMINI_API_KEY` is set and has access to the chosen model. Try a different model via `model=`.
- **Sample-rate issues**: Use `send_audio_pcm(..., target_rate=48000)` to normalize input frames.
