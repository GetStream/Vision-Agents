"""
Integration tests for the TavusClient and TavusProcessor classes.

These tests make actual API calls to the Tavus API and require a valid TAVUS_KEY
environment variable to be set.
"""

import pytest
import os
from dotenv import load_dotenv
from unittest.mock import patch
import requests

from stream_agents.processors.tavus_processor import TavusClient, TavusProcessor


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
    
    @pytest.mark.integration
    def test_create_conversation_success(self, client, test_replica_id, test_persona_id):
        """Test creating a conversation with valid parameters."""
        try:
            conversation_data = client.create_conversation(
                replica_id=test_replica_id,
                persona_id=test_persona_id,
                conversation_name="Test Conversation - Integration Test"
            )
            
            # Verify response structure
            assert "conversation_id" in conversation_data
            assert "conversation_url" in conversation_data
            assert "status" in conversation_data
            assert "replica_id" in conversation_data
            assert "persona_id" in conversation_data
            
            # Verify the returned data matches what we sent
            assert conversation_data["replica_id"] == test_replica_id
            assert conversation_data["persona_id"] == test_persona_id
            
            # Store conversation_id for cleanup
            return conversation_data["conversation_id"]
            
        except requests.RequestException as e:
            # If this fails due to invalid replica/persona IDs, that's expected
            # Log the error and skip the test
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 400:
                    pytest.skip(f"Test skipped due to invalid test IDs: {e}")
                elif e.response.status_code == 401:
                    pytest.fail("Authentication failed - check your TAVUS_KEY")
                else:
                    pytest.fail(f"Unexpected API error: {e}")
            else:
                pytest.fail(f"Network error: {e}")
    
    @pytest.mark.integration
    def test_create_conversation_with_invalid_replica_id(self, client, test_persona_id):
        """Test creating a conversation with invalid replica ID."""
        with pytest.raises(requests.RequestException) as exc_info:
            client.create_conversation(
                replica_id="invalid_replica_id",
                persona_id=test_persona_id
            )
        
        # Should get a 400 or similar error for invalid replica ID
        assert exc_info.value.response.status_code >= 400
    
    @pytest.mark.integration
    def test_create_conversation_with_missing_parameters(self, client):
        """Test creating a conversation with missing required parameters."""
        with pytest.raises(ValueError) as exc_info:
            client.create_conversation(replica_id="", persona_id="")
        
        assert "Both replica_id and persona_id are required" in str(exc_info.value)
    
    @pytest.mark.integration
    def test_create_conversation_with_audio_only(self, client, test_replica_id, test_persona_id):
        """Test creating an audio-only conversation."""
        try:
            conversation_data = client.create_conversation(
                replica_id=test_replica_id,
                persona_id=test_persona_id,
                conversation_name="Audio Only Test",
                audio_only=True
            )
            
            assert "conversation_id" in conversation_data
            assert "conversation_url" in conversation_data
            
        except requests.RequestException as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 400:
                pytest.skip(f"Test skipped due to invalid test IDs: {e}")
            else:
                raise
    
    @pytest.mark.integration
    def test_get_conversation_with_invalid_id(self, client):
        """Test getting a conversation with invalid ID."""
        with pytest.raises(requests.RequestException) as exc_info:
            client.get_conversation("invalid_conversation_id")
        
        # Should get a 404 or similar error for invalid conversation ID
        assert exc_info.value.response.status_code >= 400
    
    @pytest.mark.integration
    def test_end_conversation_with_invalid_id(self, client):
        """Test ending a conversation with invalid ID."""
        with pytest.raises(requests.RequestException) as exc_info:
            client.end_conversation("invalid_conversation_id")
        
        # Should get a 404 or similar error for invalid conversation ID
        assert exc_info.value.response.status_code >= 400


class TestTavusProcessorIntegration:
    """Integration tests for TavusProcessor."""
    
    @pytest.fixture(scope="class")
    def api_key(self):
        """Get the Tavus API key from environment variables."""
        api_key = os.getenv("TAVUS_KEY")
        if not api_key:
            pytest.skip("TAVUS_KEY environment variable not set")
        return api_key
    
    @pytest.fixture(scope="class")
    def test_replica_id(self):
        """Test replica ID."""
        return os.getenv("TAVUS_TEST_REPLICA_ID", "rfe12d8b9597")
    
    @pytest.fixture(scope="class")
    def test_persona_id(self):
        """Test persona ID."""
        return os.getenv("TAVUS_TEST_PERSONA_ID", "pdced222244b")
    
    @pytest.mark.integration
    def test_processor_initialization_without_auto_create(self, api_key, test_replica_id, test_persona_id):
        """Test TavusProcessor initialization without auto-creating conversation."""
        processor = TavusProcessor(
            api_key=api_key,
            replica_id=test_replica_id,
            persona_id=test_persona_id,
            auto_create=False
        )
        
        assert processor.api_key == api_key
        assert processor.replica_id == test_replica_id
        assert processor.persona_id == test_persona_id
        assert processor.conversation_id is None
        assert processor.conversation_url is None
        assert isinstance(processor.client, TavusClient)
    
    @pytest.mark.integration
    def test_processor_initialization_with_auto_create(self, api_key, test_replica_id, test_persona_id):
        """Test TavusProcessor initialization with auto-creating conversation."""
        try:
            processor = TavusProcessor(
                api_key=api_key,
                replica_id=test_replica_id,
                persona_id=test_persona_id,
                conversation_name="Auto-created Test Conversation",
                auto_create=True
            )
            
            assert processor.conversation_id is not None
            assert processor.conversation_url is not None
            assert processor.conversation_data is not None
            
            # Test state method
            state = processor.state()
            assert state["conversation_id"] == processor.conversation_id
            assert state["status"] == "active"
            assert state["replica_id"] == test_replica_id
            assert state["persona_id"] == test_persona_id
            
            # Test input method
            input_data = processor.input()
            assert input_data["replica_id"] == test_replica_id
            assert input_data["persona_id"] == test_persona_id
            
        except requests.RequestException as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 400:
                pytest.skip(f"Test skipped due to invalid test IDs: {e}")
            else:
                raise
    
    @pytest.mark.integration
    def test_processor_manual_conversation_creation(self, api_key, test_replica_id, test_persona_id):
        """Test manual conversation creation after processor initialization."""
        processor = TavusProcessor(
            api_key=api_key,
            replica_id=test_replica_id,
            persona_id=test_persona_id,
            auto_create=False
        )
        
        assert processor.conversation_id is None
        
        try:
            conversation_data = processor.create_conversation()
            
            assert processor.conversation_id is not None
            assert processor.conversation_url is not None
            assert conversation_data == processor.conversation_data
            
        except requests.RequestException as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 400:
                pytest.skip(f"Test skipped due to invalid test IDs: {e}")
            else:
                raise
    
    @pytest.mark.integration
    def test_processor_with_audio_only(self, api_key, test_replica_id, test_persona_id):
        """Test TavusProcessor with audio-only conversation."""
        try:
            processor = TavusProcessor(
                api_key=api_key,
                replica_id=test_replica_id,
                persona_id=test_persona_id,
                audio_only=True,
                auto_create=True
            )
            
            assert processor.audio_only is True
            
            # Check that the state reflects audio-only mode
            state = processor.state()
            assert state["audio_only"] is True
            
            input_data = processor.input()
            assert input_data["audio_only"] is True
            
        except requests.RequestException as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 400:
                pytest.skip(f"Test skipped due to invalid test IDs: {e}")
            else:
                raise


class TestTavusClientUnit:
    """Unit tests for TavusClient that don't make actual API calls."""
    
    def test_client_initialization_custom_base_url(self):
        """Test TavusClient initialization with custom base URL."""
        api_key = "test_key"
        custom_url = "https://custom.tavus.api/v1"
        client = TavusClient(api_key, base_url=custom_url)
        
        assert client.api_key == api_key
        assert client.base_url == custom_url
    
    def test_create_conversation_parameter_validation(self):
        """Test parameter validation in create_conversation."""
        client = TavusClient("test_key")
        
        with pytest.raises(ValueError):
            client.create_conversation("", "valid_persona")
        
        with pytest.raises(ValueError):
            client.create_conversation("valid_replica", "")
        
        with pytest.raises(ValueError):
            client.create_conversation(None, "valid_persona")
    
    @patch('requests.Session.post')
    def test_create_conversation_request_structure(self, mock_post):
        """Test that create_conversation builds the request correctly."""
        # Mock successful response
        mock_response = {
            "conversation_id": "test_id",
            "conversation_url": "https://test.url",
            "status": "active",
            "replica_id": "test_replica",
            "persona_id": "test_persona"
        }
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None
        
        client = TavusClient("test_key")
        result = client.create_conversation(
            replica_id="test_replica",
            persona_id="test_persona",
            conversation_name="Test",
            callback_url="https://callback.url",
            audio_only=True
        )
        
        # Verify the request was made correctly
        mock_post.assert_called_once_with(
            "https://tavusapi.com/v2/conversations",
            json={
                "replica_id": "test_replica",
                "persona_id": "test_persona",
                "conversation_name": "Test",
                "callback_url": "https://callback.url",
                "audio_only": True
            }
        )
        
        assert result == mock_response


if __name__ == "__main__":
    # Run integration tests only if TAVUS_KEY is available
    if os.getenv("TAVUS_KEY"):
        pytest.main([__file__, "-v", "-m", "integration"])
    else:
        print("TAVUS_KEY not found. Running unit tests only.")
        pytest.main([__file__, "-v", "-m", "not integration"])
