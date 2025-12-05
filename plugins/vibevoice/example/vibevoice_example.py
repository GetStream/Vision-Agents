import asyncio

import argparse
from vision_agents.plugins.vibevoice import TTS


async def main():
    parser = argparse.ArgumentParser(description="VibeVoice TTS Example")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of VibeVoice TTS.",
        help="Text to speak",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "remote"],
        default="local",
        help="Execution mode",
    )
    parser.add_argument(
        "--url", type=str, default="ws://localhost:3000", help="Remote WebSocket URL"
    )

    args = parser.parse_args()

    print(f"Initializing VibeVoice TTS in {args.mode} mode...")

    if args.mode == "remote":
        tts_engine = TTS(base_url=args.url)
    else:
        print(
            "Note: Local mode requires VibeVoice dependencies and model download (approx 1GB)."
        )
        tts_engine = TTS()
        print("Warming up model...")
        tts_engine.warmup()

    print(f"Streaming audio for: '{args.text}'")

    chunk_count = 0
    total_bytes = 0

    try:
        async for chunk in tts_engine.stream_audio(args.text):
            chunk_count += 1
            total_bytes += len(chunk.samples.tobytes())
            print(
                f"Received chunk {chunk_count}: {len(chunk.samples.tobytes())} bytes",
                end="\r",
            )

        print(f"\nFinished! Total chunks: {chunk_count}, Total bytes: {total_bytes}")

    except Exception as e:
        print(f"\nError encountered: {e}")


if __name__ == "__main__":
    asyncio.run(main())
