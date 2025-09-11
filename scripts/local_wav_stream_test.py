import asyncio
import os
import sys
import wave

from dotenv import load_dotenv
import numpy as np
from getstream.audio.utils import resample_audio
from stream_agents.plugins.openai.realtime import Realtime
import logging


async def main():
    # Load environment variables (expects OPENAI_API_KEY)
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    wav_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/Users/mkahan/Development/stream-agents/plugins/test_assets/speech_48k.wav"
    )

    if not os.path.exists(wav_path):
        print(f"WAV not found: {wav_path}", file=sys.stderr)
        sys.exit(1)

    # Instantiate the OpenAI Realtime LLM
    llm = Realtime(model="gpt-4o-realtime-preview-2024-12-17")

    @llm.on("connected")
    async def _on_connected(event):
        print("[connected] OpenAI Realtime session established")

    @llm.on("transcript")
    async def _on_transcript(event):
        role = "User" if event.is_user else "Assistant"
        print(f"[transcript:{role}] {event.text}")

    @llm.on("response")
    async def _on_response(event):
        print(f"[response] {event.text}")

    @llm.on("error")
    async def _on_error(event):
        print(f"[error] {getattr(event, 'error_message', str(event))}", file=sys.stderr)

    total_audio_bytes = 0

    @llm.on("audio_output")
    async def _on_audio_output(event):
        nonlocal total_audio_bytes
        data = getattr(event, "audio_data", b"")
        total_audio_bytes += len(data)
        # Print occasionally to avoid spam
        if total_audio_bytes and total_audio_bytes % (24_000 * 2) == 0:
            # roughly each 1s at 24kHz s16 mono
            print(f"[audio_output] received ~{total_audio_bytes} bytes so far")

    # Try native wave first; if format unsupported, fall back to soundfile
    data_bytes_iter = None
    try:
        with wave.open(wav_path, "rb") as w:
            sample_rate = w.getframerate()
            channels = w.getnchannels()
            sample_width = w.getsampwidth()
            print(
                f"[info] WAV: {sample_rate} Hz, channels={channels}, width={sample_width}"
            )

            frames_per_chunk = 960  # 20 ms @ 48 kHz
            print("[info] Streaming audio (wave module)...")

            def _iter_bytes():
                while True:
                    raw = w.readframes(frames_per_chunk)
                    if not raw:
                        break
                    # If not 48 kHz mono s16, convert
                    arr = np.frombuffer(raw, dtype=np.int16)
                    if channels > 1:
                        arr = arr.reshape(-1, channels).mean(axis=1).astype(np.int16)
                    if sample_rate != 48000:
                        arr = resample_audio(arr, sample_rate, 48000).astype(np.int16)
                    yield arr.tobytes()

            data_bytes_iter = _iter_bytes()
    except wave.Error:
        try:
            import soundfile as sf  # type: ignore

            print("[info] Streaming audio (soundfile fallback)...")
            audio, sr = sf.read(wav_path, always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            # Convert to int16
            if audio.dtype != np.int16:
                audio = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
            if sr != 48000:
                audio = resample_audio(audio, sr, 48000).astype(np.int16)

            frame_len = 960

            def _iter_bytes_sf():
                for i in range(0, len(audio), frame_len):
                    chunk = audio[i : i + frame_len]
                    if chunk.size == 0:
                        break
                    if chunk.size < frame_len:
                        pad = np.zeros(frame_len - chunk.size, dtype=np.int16)
                        chunk = np.concatenate([chunk, pad])
                    yield chunk.tobytes()

            data_bytes_iter = _iter_bytes_sf()
        except Exception as e:
            print(f"[error] Unable to decode audio: {e}", file=sys.stderr)
            sys.exit(1)

    # Optionally send a text to force a spoken response
    await llm.send_text("Please respond out loud with a short greeting.")
    # Stream
    print("[info] Streaming audio...")
    await _async_iter(data_bytes_iter, llm)
    # Give time for assistant response
    await asyncio.sleep(8)

    print(f"[result] total audio bytes received: {total_audio_bytes}")
    await llm.close()


async def _async_iter(it, llm: Realtime):
    for data in it:
        # stream each chunk
        await llm.send_audio_pcm(data, target_rate=48000)
        await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(main())
