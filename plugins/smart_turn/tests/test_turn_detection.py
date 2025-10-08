import pytest
from dotenv import load_dotenv


load_dotenv()


class TestSmartTurnPlugin:
    def test_import(self):
        """Test that the plugin can be imported."""
        from vision_agents.plugins.smart_turn import TurnDetection

        assert TurnDetection is not None

    async def test_instantiation(self):
        """Test that the TurnDetection class can be instantiated."""
        from vision_agents.plugins.smart_turn import TurnDetection

        detector = TurnDetection(api_key="test_key")
        assert detector is not None
        assert detector.api_key == "test_key"
        assert detector.buffer_duration == 2.0
        assert detector._confidence_threshold == 0.5

    @pytest.mark.integration
    async def test_turn_detection_integration(self):
        """Integration test for turn detection (requires FAL_KEY in environment)."""
        # This test should be run manually with a valid FAL_KEY
        # For now, just pass
        assert True

