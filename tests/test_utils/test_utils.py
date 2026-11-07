import asyncio
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pytest
from vision_agents.core.utils.examples import get_weather_by_location, recorded_call
from vision_agents.core.utils.utils import cancel_and_wait, ensure_model

from tests.base_test import BaseTest

AUDIO = Path(__file__).parent / "test_assets" / "saturday_seven_thirty.wav"
MODEL = b"onnx weights"


class ModelServer(ThreadingHTTPServer):
    """Serves MODEL, first answering with each status in `failures`."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), ModelRequest)
        self.failures: list[int] = []
        self.requests = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/model.onnx"


class ModelRequest(BaseHTTPRequestHandler):
    server: ModelServer

    def do_GET(self) -> None:
        self.server.requests += 1
        if self.server.failures:
            self.send_error(self.server.failures.pop(0))
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(MODEL)))
        self.end_headers()
        self.wfile.write(MODEL)

    def log_message(self, format: str, *args: object) -> None:
        pass


class TestRecordedCall:
    async def test_delivers_the_clip_and_the_silence_after_it(self):
        chunk_ms, quiet_ms = 500, 1000

        async with recorded_call(AUDIO, chunk_ms=chunk_ms, quiet_ms=quiet_ms) as chunks:
            delivered = [chunk async for chunk in chunks]

        spoken = sum(chunk.duration_ms for chunk in delivered)
        assert spoken == pytest.approx(8016 + quiet_ms, abs=50)
        assert all(chunk.sample_rate == 16000 for chunk in delivered)
        assert not delivered[-1].samples.any()

    async def test_paces_the_chunks_rather_than_sending_them_at_once(self):
        loop = asyncio.get_running_loop()
        started = loop.time()

        async with recorded_call(AUDIO, chunk_ms=100) as chunks:
            delivered = 0
            async for _ in chunks:
                delivered += 1
                if delivered == 5:
                    break

        assert loop.time() - started >= 0.4


class TestWeatherUtils(BaseTest):
    @pytest.mark.integration
    async def test_get_weather_by_location_integration(self):
        """Integration test with real API call."""
        result = await get_weather_by_location("London")

        assert "current_weather" in result
        assert "temperature" in result["current_weather"]
        assert isinstance(result["current_weather"]["temperature"], (int, float))

    @pytest.mark.integration
    async def test_get_weather_by_location_boulder_colorado(self):
        """Integration test for Boulder, Colorado with real API call."""
        result = await get_weather_by_location("Boulder")

        assert "current_weather" in result
        assert "temperature" in result["current_weather"]
        assert "windspeed" in result["current_weather"]
        assert isinstance(result["current_weather"]["temperature"], (int, float))
        assert isinstance(result["current_weather"]["windspeed"], (int, float))


class TestCancelAndWait:
    async def test_cancel_single_task(self):
        async def long_running():
            await asyncio.sleep(10)

        task = asyncio.create_task(long_running())
        await cancel_and_wait(task)

        assert task.cancelled()

    async def test_cancel_multiple_tasks(self):
        async def slow_task():
            await asyncio.sleep(100)

        tasks = [asyncio.create_task(slow_task()) for _ in range(3)]
        await cancel_and_wait(*tasks)

        assert all(t.cancelled() for t in tasks)

    async def test_cancel_already_done_task(self):
        async def quick():
            return 42

        task = asyncio.create_task(quick())
        await task

        await cancel_and_wait(task)

        assert task.done()
        assert not task.cancelled()

    async def test_cancel_mix_of_done_and_pending(self):
        async def quick():
            return 1

        async def slow():
            await asyncio.sleep(100)

        done_task = asyncio.create_task(quick())
        await done_task

        pending_task = asyncio.create_task(slow())
        await cancel_and_wait(done_task, pending_task)

        assert done_task.done()
        assert not done_task.cancelled()
        assert pending_task.cancelled()

    async def test_cancel_no_futures(self):
        await cancel_and_wait()


class TestEnsureModel:
    @pytest.fixture
    def server(self) -> Iterator[ModelServer]:
        server = ModelServer()
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        yield server
        server.shutdown()
        server.server_close()
        thread.join()

    async def test_a_failure_that_may_pass_is_tried_again(
        self, server: ModelServer, tmp_path: Path
    ):
        server.failures = [503]
        path = tmp_path / "model.onnx"

        await ensure_model(str(path), server.url)

        assert await asyncio.to_thread(path.read_bytes) == MODEL
        assert server.requests == 2
        assert await asyncio.to_thread(os.listdir, tmp_path) == ["model.onnx"]

    async def test_a_model_that_is_not_there_is_asked_for_once(
        self, server: ModelServer, tmp_path: Path
    ):
        server.failures = [404]

        with pytest.raises(RuntimeError, match="Failed to download model.onnx"):
            await ensure_model(str(tmp_path / "model.onnx"), server.url)

        assert server.requests == 1
        assert await asyncio.to_thread(os.listdir, tmp_path) == []
