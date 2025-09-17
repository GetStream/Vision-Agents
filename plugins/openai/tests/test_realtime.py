import asyncio
import pytest
from dotenv import load_dotenv
from stream_agents.plugins.openai.realtime import Realtime

load_dotenv()


class TestRealtime:
    @pytest.fixture
    def realtime(self) -> Realtime:
        return Realtime(model="gpt-realtime", voice="marin")

    @pytest.mark.integration
    async def test_realtime(self, realtime: Realtime):
        await realtime.connect()
        assert realtime.rtc.token is not None
        assert realtime.rtc._mic_track is not None
        assert realtime.is_connected
        # listen on the data channel via rtc event callback
        events = []
        done = asyncio.Event()

        def on_evt(evt: dict):
            events.append(evt)
            if evt.get("type") == "conversation.item.created":
                item = evt.get("item", {}) or {}
                content_list = item.get("content", []) or []
                for c in content_list:
                    if c.get("type") in ("input_text", "output_text") and c.get("text") == "Hello, how are you?":
                        try:
                            done.set()
                        except Exception:
                            pass
                        break

        realtime.rtc.set_event_callback(on_evt)
        await realtime.send_text("Hello, how are you?")
        await asyncio.wait_for(done.wait(), timeout=10.0)
        assert any(e.get("type") == "conversation.item.created" for e in events)
