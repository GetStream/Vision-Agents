"""
Integration tests for the TavusClient and TavusProcessor classes.

These tests make actual API calls to the Tavus API and require a valid TAVUS_KEY
environment variable to be set.
"""

import pytest
import os
from dotenv import load_dotenv

from stream_agents.core.processors.tavus_processor import TavusClient


# Load environment variables
load_dotenv()


class TestTavusClientIntegration:
    """Integration tests for TavusClient that make actual API calls."""

    @pytest.fixture(scope="class")
    def api_key(self):
        """Get the Tavus API key from environment variables."""
        api_key = os.getenv("TAVUS_KEY")
        if not api_key:
            pytest.skip("TAVUS_KEY environment variable not set")
        return api_key

    @pytest.fixture(scope="class")
    def client(self, api_key):
        """Create a TavusClient instance for testing."""
        return TavusClient(api_key)

    @pytest.fixture(scope="class")
    def test_replica_id(self):
        """Test replica ID - you may need to update this with a valid replica ID."""
        # This is the example from the API docs - replace with your actual replica ID
        return os.getenv("TAVUS_TEST_REPLICA_ID", "rfe12d8b9597")

    @pytest.fixture(scope="class")
    def test_persona_id(self):
        """Test persona ID - you may need to update this with a valid persona ID."""
        # This is the example from the API docs - replace with your actual persona ID
        return os.getenv("TAVUS_TEST_PERSONA_ID", "pdced222244b")

    @pytest.mark.integration
    def test_client_initialization(self, api_key):
        """Test that TavusClient initializes correctly."""
        client = TavusClient(api_key)
        assert client.api_key == api_key
        assert client.base_url == "https://tavusapi.com/v2"
        assert "x-api-key" in client.session.headers
        assert client.session.headers["x-api-key"] == api_key
        assert client.session.headers["Content-Type"] == "application/json"

    def test_one_plus_one(self, api_key):
        assert 2 == 2
