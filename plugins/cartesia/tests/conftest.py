import os

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def cartesia_api_key_required() -> str:
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        pytest.fail(
            "Cartesia integration tests require CARTESIA_API_KEY. "
            "Set CARTESIA_API_KEY in the environment or in a .env file before "
            "running tests marked with @pytest.mark.integration.",
            pytrace=False,
        )
    return api_key
