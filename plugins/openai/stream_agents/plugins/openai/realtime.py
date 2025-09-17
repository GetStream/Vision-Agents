import asyncio
from os import getenv
from typing import Any, Optional
from httpx import AsyncClient
from stream_agents.core.llm import realtime
import logging

logger = logging.getLogger(__name__)

OPENAI_API_URL = "https://api.openai.com/v1/realtime/sessions"

class RTCManager:
    def __init__(self, api_key: str, model: str, voice: str):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.token = None


    async def connect(self):
        self.token = await self._get_session_token()

        # get sdp
        

        # connect
              
    async def _get_session_token(self) -> str:
        url = OPENAI_API_URL
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "voice": self.voice}
        async with AsyncClient() as client:
            for attempt in range(2):
                try:
                    resp = await client.post(url, headers=headers, json=payload, timeout=15)
                    resp.raise_for_status()
                    data: dict = resp.json()
                    secret = data.get("client_secret")
                    return secret.get("value")
                except Exception as e:
                    if attempt == 0:
                        await asyncio.sleep(1.0)
                        continue
                    logger.error(f"Failed to get OpenAI Realtime session token: {e}")


class Realtime(realtime.Realtime):
    def __init__(self, api_key: str, model: str, voice: str):
        super().__init__()
        self.rtc = RTCManager(api_key, model, voice)

    async def connect(self):
        await self.rtc.connect()

    # def send_audio_pcm(self, audio):
    #     pass

    async def send_text(self, text):
        pass

    async def native_send_realtime_input(
        self,
        *,
        text: Optional[str] = None,
        audio: Optional[Any] = None,
        media: Optional[Any] = None,
    ) -> None:
        ...

    async def _close_impl(self):
        ...