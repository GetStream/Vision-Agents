#!/usr/bin/env python3
"""
Vision Agents Setup Script
Creates a .env file from template and provides setup guidance.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def create_env_file(target_dir=None, force=False):
    """Create .env file from template if it doesn't exist."""
    # Get the package directory
    package_dir = Path(__file__).parent
    
    # Template file path
    template_path = package_dir / "env_template.txt"
    
    # Target directory
    if target_dir is None:
        target_dir = Path.cwd()
    else:
        target_dir = Path(target_dir)
    
    # Target .env file path
    env_path = target_dir / ".env"
    
    # Check if .env already exists
    if env_path.exists() and not force:
        print(f"✓ .env file already exists at {env_path}")
        print("💡 Use --force to overwrite existing .env file")
        return True
    
    # Check if template exists
    if not template_path.exists():
        print(f"❌ Template file not found at {template_path}")
        return False
    
    try:
        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy template to .env
        shutil.copy2(template_path, env_path)
        print(f"✓ Created .env file at: {env_path}")
        print()
        print("📁 File location:")
        print(f"   {env_path.absolute()}")
        print()
        print("📝 Next steps:")
        print("1. Edit the .env file and add your actual API keys")
        print("2. See the comments in the .env file for where to get API keys")
        print("3. Start building your vision agent!")
        print()
        print("🔗 Quick links for API keys:")
        print("  • Stream: https://getstream.io/")
        print("  • OpenAI: https://platform.openai.com/api-keys")
        print("  • Deepgram: https://console.deepgram.com/")
        print("  • Cartesia: https://cartesia.ai/")
        print("  • FAL (Smart Turn): https://fal.ai/")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def show_setup_guide():
    """Show comprehensive setup guide."""
    print("🚀 Vision Agents Setup Guide")
    print("=" * 50)
    print()
    print("✅ Package already installed!")
    print()
    print("📝 Next steps:")
    print("1. Add your API keys to the .env file")
    print("2. Start building your vision agent:")
    print()
    print("   from vision_agents import Agent")
    print("   from vision_agents.plugins import openai, deepgram, cartesia")
    print()
    print("🔧 Setup commands:")
    print("   vision-agents-setup              # Create .env file")
    print("   vision-agents-setup --force      # Overwrite existing .env")
    print("   vision-agents-setup --guide      # Show this guide")
    print()
    print("📚 Documentation: https://visionagents.ai/")
    print("💬 Examples: https://github.com/GetStream/Vision-Agents/tree/main/examples")


def main():
    """Main entry point for setup script."""
    parser = argparse.ArgumentParser(
        description="Vision Agents setup script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vision-agents-setup                    # Create .env in current directory
  vision-agents-setup --dir /path/to/project  # Create .env in specific directory
  vision-agents-setup --force           # Overwrite existing .env file
  vision-agents-setup --guide           # Show setup guide
        """
    )
    
    parser.add_argument(
        "--dir", "-d",
        help="Directory to create .env file in (default: current directory)"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing .env file"
    )
    
    parser.add_argument(
        "--guide", "-g",
        action="store_true",
        help="Show setup guide"
    )
    
    args = parser.parse_args()
    
    if args.guide:
        show_setup_guide()
        return
    
    try:
        success = create_env_file(target_dir=args.dir, force=args.force)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Setup script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
