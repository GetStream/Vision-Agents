import asyncio
import logging
from typing import Any, Optional

from getstream.video.rtc.track_util import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLMResponseEvent
from vision_agents.core.llm.realtime import Realtime
from vision_agents.core.utils.video_forwarder import VideoForwarder

import aiortc


class ConcreteRealtime(Realtime):
    """Minimal concrete implementation for testing the base class."""

    async def connect(self) -> None: ...

    async def simple_response(
        self, text: str, participant: Optional[Participant] = None
    ) -> LLMResponseEvent[Any]: ...

    async def simple_audio_response(
        self, pcm: PcmData, participant: Optional[Participant] = None
    ) -> None: ...

    async def watch_video_track(
        self,
        track: aiortc.mediastreams.MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None: ...

    async def close(self) -> None: ...


class TestRealtimeRunToolInBackground:
    async def test_tool_runs_without_blocking(self):
        rt = ConcreteRealtime()
        started = asyncio.Event()
        finish = asyncio.Event()

        async def tool():
            started.set()
            await finish.wait()

        rt._run_tool_in_background(tool())

        await started.wait()
        assert len(rt._tool_tasks) == 1

        finish.set()
        await asyncio.sleep(0)  # let tool complete
        await asyncio.sleep(0)  # let done_callback fire
        assert len(rt._tool_tasks) == 0

    async def test_tool_exception_is_logged(self, caplog):
        rt = ConcreteRealtime()

        async def failing_tool():
            raise ValueError("tool failed")

        with caplog.at_level(logging.ERROR, logger="vision_agents.core.llm.realtime"):
            rt._run_tool_in_background(failing_tool())
            await asyncio.sleep(0)  # let tool raise
            await asyncio.sleep(0)  # let done_callback fire

        assert "tool failed" in caplog.text

    async def test_await_pending_tools_waits_for_completion(self):
        rt = ConcreteRealtime()
        completed = asyncio.Event()

        async def tool():
            await asyncio.sleep(0.1)
            completed.set()

        rt._run_tool_in_background(tool())
        assert not completed.is_set()

        await rt._await_pending_tools()

        assert completed.is_set()
        assert len(rt._tool_tasks) == 0
