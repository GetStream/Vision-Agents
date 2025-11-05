"""
Tests for the roboflow plugin.
"""

import pytest
from pathlib import Path
from typing import Iterator
from PIL import Image
import numpy as np

from vision_agents.plugins.roboflow import RoboflowDetectionProcessor


class TestRoboflowDetectionProcessor:
    """Test cases for RoboflowDetectionProcessor."""

    def test_processor_initialization_with_params(self):
        """Test processor can be instantiated with valid parameters."""
        processor = RoboflowDetectionProcessor(
            api_key="test_key",
            workspace_id="test_workspace",
            project_id="test_project",
            model_version=1,
        )
        assert processor is not None
        assert processor.workspace_id == "test_workspace"
        assert processor.project_id == "test_project"
        assert processor.model_version == 1

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError, match="API key required"):
            RoboflowDetectionProcessor(
                workspace_id="test",
                project_id="test",
            )

    def test_missing_workspace_raises_error(self):
        """Test that missing workspace_id raises ValueError."""
        with pytest.raises(ValueError, match="workspace_id and project_id are required"):
            RoboflowDetectionProcessor(
                api_key="test_key",
                project_id="test",
            )

    def test_missing_project_raises_error(self):
        """Test that missing project_id raises ValueError."""
        with pytest.raises(ValueError, match="workspace_id and project_id are required"):
            RoboflowDetectionProcessor(
                api_key="test_key",
                workspace_id="test",
            )

    def test_processor_cleanup(self):
        """Test processor can be closed cleanly."""
        processor = RoboflowDetectionProcessor(
            api_key="test_key",
            workspace_id="test_workspace",
            project_id="test_project",
        )
        processor.close()
        assert processor._shutdown is True

