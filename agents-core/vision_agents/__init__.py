"""
Vision Agents - Open video agents for building low latency video and voice agents.
"""

import os
import sys
from pathlib import Path

# Version will be set by hatch-vcs
__version__ = "0.0.0"

# Auto-create .env file on first import if it doesn't exist
def _setup_env_file():
    """Automatically create .env file from template if it doesn't exist."""
    try:
        # Get the package directory
        package_dir = Path(__file__).parent
        
        # Template file path
        template_path = package_dir / "env_template.txt"
        
        # Target .env file path (in current working directory)
        env_path = Path.cwd() / ".env"
        
        # Check if .env already exists
        if env_path.exists():
            return
        
        # Check if template exists
        if not template_path.exists():
            return
        
        # Only create .env if we're in a project directory (not in site-packages)
        # This prevents creating .env files in unexpected places
        if "site-packages" in str(package_dir) or "dist-packages" in str(package_dir):
            return
        
        # Copy template to .env
        import shutil
        shutil.copy2(template_path, env_path)
        
        # Print helpful message
        print("🎉 Vision Agents: Created .env file with example configuration!")
        print()
        print("📁 File location:")
        print(f"   {env_path.absolute()}")
        print()
        print("📝 Please edit the .env file and add your actual API keys")
        print("🔗 See the comments in the .env file for where to get API keys")
        print("💡 Run 'vision-agents-setup' command for more setup options")
        
    except Exception:
        # Silently fail - don't break the import if env setup fails
        pass

# Run the setup function
_setup_env_file()

# Import core components
from .core.agents import Agent
from .core.edge.types import User

__all__ = ["Agent", "User", "__version__"]
