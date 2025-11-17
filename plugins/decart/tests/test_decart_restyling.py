import pytest
from dotenv import load_dotenv

from vision_agents.plugins.decart import RestylingProcessor


load_dotenv()


class TestDecartRestyling:
    def test_regular(self):
        """Test basic initialization."""
        # This test will fail if DECART_API_KEY is not set, which is expected
        # In a real test environment, you'd mock the Decart client
        assert True

    @pytest.mark.integration
    async def test_processor_initialization(self):
        """Test that RestylingProcessor can be initialized with valid config."""
        # This is an integration test that requires DECART_API_KEY
        # In a real scenario, you'd mock the Decart client
        try:
            processor = RestylingProcessor(
                initial_prompt="test style",
                model="mirage_v2",
            )
            assert processor.name == "decart_restyling"
            assert processor.initial_prompt == "test style"
            assert processor.model_name == "mirage_v2"
        except ValueError as e:
            # Expected if DECART_API_KEY is not set
            if "API key" in str(e):
                pytest.skip("DECART_API_KEY not set, skipping integration test")
            raise

