from typing import Dict, Any

import httpx


async def get_weather_by_location(location: str) -> Dict[str, Any]:
    """
    Get current weather for a location using Open-Meteo API.

    Args:
        location: Name of the location (city, place, etc.)

    Returns:
        Weather data dictionary containing current weather information

    Raises:
        ValueError: If location not found or API response is invalid
        httpx.HTTPError: If API request fails
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Get geocoding data for the location
        geo_response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search", params={"name": location}
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            raise ValueError(f"Location '{location}' not found")

        # Get weather for the location
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        weather_response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        return weather_data
