import asyncio

import pytest
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.core.utils.utils import cancel_and_wait

from tests.base_test import BaseTest


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
