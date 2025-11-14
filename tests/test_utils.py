import pytest
from vision_agents.core.utils.examples import get_weather_by_location

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
