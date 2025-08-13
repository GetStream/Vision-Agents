"""
Utility functions for Stream Agents examples.
"""

import logging
import os
import webbrowser
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


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
