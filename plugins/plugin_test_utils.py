"""
Test utility functions for GetStream.

This module provides utility functions for accessing test assets and other
resources from the main test directory.

IMPORTANT: When writing tests for plugins, consider the following:

1. Pytest automatically loads fixtures from conftest.py files in parent directories,
   but they may not be directly accessible from plugin subdirectories.

2. If your test requires fixtures from the root conftest.py:
   - Run tests from the project root directory using: uv run pytest path/to/test.py
   - For complex fixtures, consider placing tests in the main test directory
   - Or duplicate the fixture functionality in your plugin tests

3. For accessing test assets, use the get_asset_path(), get_audio_asset(),
   and get_json_metadata() functions provided in this module.
"""

import json
import os
import sys
from typing import Any, Dict

# Load environment variables from .env files
try:
    from dotenv import load_dotenv

    # First try to load from the project root .env
    load_dotenv()

    # Then check for plugin-specific .env files and load them, overriding root settings if needed
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    for plugin_type in ["stt", "tts", "vad"]:
        plugins_dir = os.path.join(current_file_dir, plugin_type)
        if os.path.isdir(plugins_dir):
            for plugin_dir in os.listdir(plugins_dir):
                plugin_env_path = os.path.join(plugins_dir, plugin_dir, ".env")
                if os.path.isfile(plugin_env_path):
                    load_dotenv(plugin_env_path, override=True)
except ImportError:
    # dotenv is not available, so environment variables won't be loaded
    pass

# Import fixtures from the root conftest
try:
    # Add project root to path to ensure imports work
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Re-export fixtures
    __all__ = [
        "get_asset_path",
        "get_audio_asset",
        "get_json_metadata",
    ]

except ImportError:
    # If fixtures can't be imported, just continue
    __all__ = ["get_asset_path", "get_audio_asset", "get_json_metadata"]


def get_asset_path(file_name: str) -> str:
    """
    Get the path to an asset file in the tests/assets directory.

    Args:
        file_name: The name of the asset file

    Returns:
        The full path to the asset file
    """
    # Find the project root (where tests/test_assets is located)
    project_root = _find_project_root()

    # Return the path to the asset
    return os.path.join(project_root, "tests", "test_assets", file_name)


def get_audio_asset(file_name: str) -> str:
    """
    Get the path to an audio asset file.

    Args:
        file_name: The name of the audio file (e.g., "mia.mp3")

    Returns:
        The full path to the audio file
    """
    return get_asset_path(file_name)


def get_json_metadata(file_name: str) -> Dict[str, Any]:
    """
    Load and return the contents of a JSON metadata file.

    Args:
        file_name: The name of the JSON file (e.g., "mia.json")

    Returns:
        The parsed JSON content as a dictionary
    """
    json_path = get_asset_path(file_name)
    with open(json_path, "r") as f:
        return json.load(f)


def _find_project_root() -> str:
    """
    Find the project root directory by looking for the 'tests/test_assets' directory.

    Returns:
        The path to the project root directory
    """
    # Start from the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Walk up the directory tree until we find the project root
    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
        # Check if tests/test_assets exists in this directory
        if os.path.isdir(os.path.join(current_dir, "tests", "test_assets")):
            return current_dir

        # Not found, go up one level
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached filesystem root
            break
        current_dir = parent_dir

    # If not found by walking up, try a fixed path from the current file
    # This assumes the module is at plugins/plugin_test_utils.py
    possible_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(possible_root, "tests", "test_assets")):
        return possible_root

    # Last resort: try to locate from the current working directory
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, "tests", "test_assets")):
        return cwd

    raise FileNotFoundError(
        "Could not find the project root directory with 'tests/test_assets'. "
        "Make sure you're running tests from the project root or a subdirectory."
    )
