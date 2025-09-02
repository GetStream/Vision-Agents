import os
import asyncio

import pytest

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # If python-dotenv is not installed, ignore and skip the test
    pass


@pytest.mark.asyncio
async def test_gemini_live_with_real_api():
    """
    Optional smoke test: requires GOOGLE_API_KEY and google-genai installed.
    Connects, sends a short text, and asserts we receive audio or text back.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key is None:
        pytest.skip("GOOGLE_API_KEY not set – skipping live Gemini test")

    # Require google-genai to be present
    try:
        import google.genai  # noqa: F401
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"Required Google packages not available: {e}")

    from getstream.plugins.gemini.live.live import GeminiLive

    # Set up instance and event capture
    events = {"audio": [], "text": []}
    sts = GeminiLive(api_key=api_key)

    @sts.on("audio")  # type: ignore[arg-type]
    async def _on_audio(data: bytes):
        events["audio"].append(data)

    @sts.on("text")  # type: ignore[arg-type]
    async def _on_text(text: str):
        events["text"].append(text)

    ready = await sts.wait_until_ready(timeout=10.0)
    if not ready:
        await sts.close()
        raise RuntimeError("Gemini Live did not become ready in time")

    # Send a very short prompt and wait briefly for any response
    await sts.send_text("Speak a short sentence.")

    # Wait up to 10s for any audio or text response
    for _ in range(50):
        if events["audio"] or events["text"]:
            break
        await asyncio.sleep(0.2)

    try:
        assert (
            events["audio"] or events["text"]
        ), "No response received from Gemini Live"
    finally:
        await sts.close()
