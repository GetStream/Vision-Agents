import pytest
import os


@pytest.fixture(scope="session")
def deepgram_api_key():
    """Get the Deepgram API key from environment variables."""
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        pytest.skip(
            "DEEPGRAM_API_KEY environment variable not set. Add it to your .env file."
        )
    return api_key


@pytest.fixture(scope="session")
def deepgram_model():
    """Get the Deepgram model name from environment variables."""
    return os.environ.get("DEEPGRAM_MODEL", "nova-2")


@pytest.fixture(scope="session")
def deepgram_language():
    """Get the Deepgram language from environment variables."""
    return os.environ.get("DEEPGRAM_LANGUAGE", "en-US")
