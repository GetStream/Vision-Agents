"""
Generic CLI runner for Vision Agents examples.

Provides a Click-based CLI with common options for debugging and logging.
"""

import asyncio
import logging
import warnings
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

import click
from vision_agents.core.utils.logging import configure_sdk_logger

if TYPE_CHECKING:
    from vision_agents.core.agents.agent_launcher import AgentLauncher


asyncio_logger = logging.getLogger("asyncio")

logger = logging.getLogger(__name__)


def cli(launcher: "AgentLauncher") -> None:
    """
    Create and run a CLI from an AgentLauncher.

    Usage:
        if __name__ == "__main__":
            cli(AgentLauncher(create_agent=create_agent, join_call=join_call))

    Args:
        launcher: AgentLauncher instance with create_agent and join_call functions
    """

    @click.command()
    @click.option(
        "--call-type",
        type=str,
        default="default",
        help="Call type for the video call",
    )
    @click.option(
        "--call-id",
        type=str,
        default=None,
        help="Call ID for the video call (auto-generated if not provided)",
    )
    @click.option(
        "--debug",
        is_flag=True,
        default=False,
        help="Enable debug mode",
    )
    @click.option(
        "--log-level",
        type=click.Choice(
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
        ),
        default="INFO",
        help="Set the logging level",
    )
    @click.option(
        "--no-demo",
        is_flag=True,
        default=False,
        help="Disable opening the demo UI",
    )
    @click.option(
        "--video-track-override",
        type=click.Path(dir_okay=False, exists=True, resolve_path=True),
        default=None,
        help="Optional local video track override for debugging. "
        "This track will play instead of any incoming video track.",
    )
    def run_agent(
        call_type: str,
        call_id: Optional[str],
        debug: bool,
        log_level: str,
        no_demo: bool,
        video_track_override: Optional[str],
    ) -> None:
        """Run the agent with the specified configuration."""
        # Configure logging
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        configure_sdk_logger(level=numeric_level)

        # Suppress dataclasses_json missing value RuntimeWarnings.
        # They pollute the output and cannot be fixed by the users.
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="dataclasses_json.core"
        )

        # Generate call ID if not provided
        if call_id is None:
            call_id = str(uuid4())

        async def _run():
            logger.info("🚀 Launching agent...")

            try:
                # Start the agent launcher.
                await launcher.start()

                # Create the agent
                agent = await launcher.launch()
                if video_track_override:
                    agent.set_video_track_override_path(video_track_override)

                logger.info("✅ Agent warmed up and ready")

                # Open demo UI by default
                if (
                    not no_demo
                    and hasattr(agent, "edge")
                    and hasattr(agent.edge, "open_demo_for_agent")
                ):
                    logger.info("🌐 Opening demo UI...")
                    await agent.edge.open_demo_for_agent(agent, call_type, call_id)

                # Join call if join_call function is provided
                if launcher.join_call:
                    logger.info(f"📞 Joining call: {call_type}/{call_id}")
                    result = launcher.join_call(agent, call_type, call_id)
                    if asyncio.iscoroutine(result):
                        await result
                else:
                    logger.warning(
                        '⚠️ No "join_call" function provided; the agent is created but will not join the call'
                    )
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal, shutting down gracefully...")
            except Exception as e:
                logger.error(f"❌ Error running agent: {e}", exc_info=True)
                raise
            finally:
                await launcher.stop()

        asyncio_logger_level = asyncio_logger.level

        try:
            asyncio.run(_run(), debug=debug)
        except KeyboardInterrupt:
            # Temporarily suppress asyncio error logging during cleanup
            asyncio_logger_level = asyncio_logger.level
            # Suppress KeyboardInterrupt and asyncio errors during cleanup
            asyncio_logger.setLevel(logging.CRITICAL)
            logger.info("👋 Agent shutdown complete")
        finally:
            # Restore original logging level
            asyncio_logger.setLevel(asyncio_logger_level)

    # Invoke the click command
    run_agent()
