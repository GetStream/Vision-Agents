import asyncio
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict, Any, Union

import httpx
from getstream.video.rtc.track_util import PcmData


@asynccontextmanager
async def recorded_call(
    path: Union[str, Path],
    chunk_ms: int = 100,
    quiet_ms: int = 2000,
) -> AsyncIterator[AsyncIterator[PcmData]]:
    """A wav file delivered the way a call delivers audio, for examples without a speaker.

    Sending a whole clip at once has the model see all of it before anybody has finished
    talking, which is not the problem a streaming model solves.

    Args:
        path: The wav file to read.
        chunk_ms: How much audio a call delivers at a time.
        quiet_ms: Silence streamed after the clip. Not padding: a streaming model decides
            a turn is over by hearing the caller stop, and a clip that ends the instant
            the speech does never gives it that.

    Yields:
        The clip and then the silence, a chunk at a time, paced as they were spoken.
    """
    audio = await asyncio.to_thread(_read_wav, path)
    quiet = PcmData.from_bytes(
        bytes(audio.sample_rate * quiet_ms // 1000 * 2), sample_rate=audio.sample_rate
    )

    async def paced() -> AsyncIterator[PcmData]:
        for clip in (audio, quiet):
            for chunk in clip.chunks(clip.sample_rate * chunk_ms // 1000):
                yield chunk
                await asyncio.sleep(chunk_ms / 1000)

    yield paced()


def _read_wav(path: Union[str, Path]) -> PcmData:
    with wave.open(str(path)) as f:
        return PcmData.from_bytes(
            f.readframes(f.getnframes()), sample_rate=f.getframerate()
        )


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
