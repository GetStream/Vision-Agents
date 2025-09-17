import asyncio
import pytest
from dotenv import load_dotenv
from os import getenv
from stream_agents.plugins.openai.realtime import Realtime

load_dotenv()


class TestRealtime:
    @pytest.fixture
    def realtime(self) -> Realtime:
        return Realtime(model="gpt-realtime", api_key=getenv("OPENAI_API_KEY"), voice="marin")

    async def test_realtime(self, realtime: Realtime):
        await realtime.connect()
        assert realtime.rtc.token is not None
        assert realtime.rtc._mic_track is not None
        assert 0