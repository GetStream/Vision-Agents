"""
Stream Agents Utilities Package

This package provides utility functions and scripts for Stream Agents.
"""

import os
import webbrowser
from urllib.parse import urlencode
from uuid import uuid4

from getstream.models import UserRequest
from getstream.video.call import Call

import logging

logger = logging.getLogger(__name__)


def open_demo(call: Call) -> str:
    client = call.client.stream

    # Create a human user for testing
    human_id = f"user-{uuid4()}"

    client.upsert_users(UserRequest(id=human_id, name="Human User"))

    # Create user token for browser access
    token = client.create_token(human_id, expiration=3600)

    """Helper function to open browser with Stream call link."""
    base_url = (
        f"{os.getenv('EXAMPLE_BASE_URL', 'https://getstream.io/video/demos')}/join/"
    )
    params = {
        "api_key": client.api_key,
        "token": token,
        "skip_lobby": "true",
        "video_encoder": "vp8",
        "bitrate": 2000000,
        "w": 1920,
        "h": 1080,
    }

    url = f"{base_url}{call.id}?{urlencode(params)}"
    print(f"🌐 Opening browser to: {url}")

    try:
        webbrowser.open(url)
        print("✅ Browser opened successfully!")
    except Exception as e:
        print(f"❌ Failed to open browser: {e}")
        print(f"Please manually open this URL: {url}")

    return url


def open_pronto(api_key: str, token: str, call_id: str):
    """Open browser with the video call URL."""
    # Use the same URL pattern as the working workout assistant example
    base_url = (
        f"{os.getenv('EXAMPLE_BASE_URL', 'https://pronto-staging.getstream.io')}/join/"
    )
    params = {
        "api_key": api_key,
        "token": token,
        "skip_lobby": "true",
        "video_encoder": "vp8",
    }

    url = f"{base_url}{call_id}?{urlencode(params)}"
    logger.info(f"🌐 Opening browser: {url}")

    try:
        webbrowser.open(url)
        logger.info("✅ Browser opened successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to open browser: {e}")
        logger.info(f"Please manually open this URL: {url}")


__all__ = [
    "open_demo",
    "open_pronto",
]
