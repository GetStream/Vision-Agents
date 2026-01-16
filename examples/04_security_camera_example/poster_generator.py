"""Wanted poster generation and X posting utilities."""

import io
import logging
import os
from typing import Optional

import cv2
import numpy as np
import tweepy
from google.genai.client import Client
from google.genai.types import GenerateContentConfig, Part

logger = logging.getLogger(__name__)


async def post_to_x(image_data: bytes, caption: str) -> Optional[str]:
    """Post an image to X with a caption. Returns tweet URL or None."""
    consumer_key = os.getenv("X_CONSUMER_KEY")
    consumer_secret = os.getenv("X_CONSUMER_SECRET")
    access_token = os.getenv("X_ACCESS_TOKEN")
    access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")

    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        logger.warning("⚠️ X credentials not configured")
        return None

    auth = tweepy.OAuth1UserHandler(
        consumer_key, consumer_secret, access_token, access_token_secret
    )
    api_v1 = tweepy.API(auth)
    client_v2 = tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )

    media = api_v1.media_upload(filename="poster.png", file=io.BytesIO(image_data))
    response = client_v2.create_tweet(text=caption, media_ids=[media.media_id])

    me = client_v2.get_me()
    tweet_url = f"https://x.com/{me.data.username}/status/{response.data['id']}"
    logger.info(f"🐦 Posted to X: {tweet_url}")
    return tweet_url


WANTED_POSTER_PROMPT = """
Create a vintage western wanted poster as a frame around the image.
Use sepia tone and aged paper texture.
You'll be given the name {name} to work with. 
Add a western-style nickname to the name for the wanted poster. Do not change the name. 
    e.g. if the name is "John", you could say "Rattlesnake John" or "Six-Gun John" or "Desperado John", etc.
The nickname MUST MUST MUST be wild-west themed. Do not refer to silliness or sausages.
Add text 'WANTED: {name}' at the top and 'FOR BEING A SILLY SAUSAGE ' at the bottom.
Add 'REWARD: MY UNDYING GRATITUDE' at the bottom.
Add no other text. Make it fun.
Don't change anything else about the image, except resizing it to fit the frame and changing the colour scheme to fit the western theme.
Don't modify the image in any other way.
"""


async def generate_wanted_poster(face_image: np.ndarray, name: str) -> Optional[bytes]:
    """Generate a wanted poster using Gemini's image generation API."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    client = Client(api_key=api_key).aio

    _, buffer = cv2.imencode(".png", face_image)
    image_part = Part.from_bytes(data=buffer.tobytes(), mime_type="image/png")

    logger.info(f"🎨 Generating wanted poster for {name}...")
    response = await client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[image_part, WANTED_POSTER_PROMPT.format(name=name)],
        config=GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )

    if not response.candidates or response.candidates[0].content is None:
        logger.warning("⚠️ Gemini returned no image")
        return None

    for part in response.candidates[0].content.parts or []:
        inline = part.inline_data
        if (
            inline
            and inline.mime_type
            and inline.mime_type.startswith("image/")
            and inline.data
        ):
            logger.info(f"✅ Got poster image ({len(inline.data)} bytes)")
            return inline.data

    logger.warning("⚠️ No image found in response")
    return None


async def generate_and_post_poster(
    face_image: np.ndarray,
    name: str,
    post_to_x_enabled: bool = False,
    tweet_caption: str = "🤠 WANTED: This dangerous AI tester! #VisionAgents",
) -> tuple[Optional[bytes], Optional[str]]:
    """Generate a wanted poster and optionally post it to X."""
    image_data = await generate_wanted_poster(face_image, name)
    if image_data is None:
        return None, None

    tweet_url = (
        await post_to_x(image_data, tweet_caption) if post_to_x_enabled else None
    )
    return image_data, tweet_url
