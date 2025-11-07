"""
Generic CLI runner for Vision Agents examples.

Provides a Click-based CLI with common options for debugging and logging.
"""

import asyncio
import logging
from typing import Awaitable, Callable, Optional, TYPE_CHECKING
from uuid import uuid4

import click

if TYPE_CHECKING:
    from vision_agents.core.agents.agent_launcher import AgentLauncher


def run_example(
    async_main: Callable[[], Awaitable[None]],
    debug: bool = False,
    log_level: str = "INFO",
    agent_launcher: Optional["AgentLauncher"] = None,
) -> None:
    """
    Run an async example with optional debug and logging configuration.

    Args:
        async_main: Async function to run
        debug: Enable debug mode (BlockBuster + asyncio debug)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        agent_launcher: Optional agent launcher to use for warmup
    """
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Enable BlockBuster in debug mode
    if debug:
        loop = asyncio.get_running_loop()
        loop.slow_callback_duration = 0.005  # warn if blocking >5ms
        from blockbuster import BlockBuster

        blockbuster = BlockBuster()
        blockbuster.activate()
        logging.info("BlockBuster activated")

    # Run warmup if agent launcher provided
    async def _run_with_launcher():
        if agent_launcher:
            logging.info("Warming up agent via launcher...")
            await agent_launcher.launch()
        await async_main()

    # Run the async main function
    asyncio.run(_run_with_launcher(), debug=debug)


def example_cli(
    func: Optional[Callable[[], Awaitable[None]]] = None,
    agent_launcher: Optional["AgentLauncher"] = None,
) -> Callable:
    """
    Decorator to add standard CLI options to an example.

    Usage:
        @example_cli
        async def main():
            # Your example code here
            pass

        if __name__ == "__main__":
            main()
    
    Or with agent launcher:
        @example_cli(agent_launcher=launcher)
        async def main():
            # Your example code here
            pass

        if __name__ == "__main__":
            main()
    """

    def decorator(f: Callable[[], Awaitable[None]]) -> click.Command:
        @click.command()
        @click.option(
            "--debug",
            is_flag=True,
            default=False,
            help="Enable debug mode (BlockBuster + asyncio debug)",
        )
        @click.option(
            "--log-level",
            type=click.Choice(
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
            ),
            default="INFO",
            help="Set the logging level",
        )
        def wrapper(debug: bool, log_level: str) -> None:
            run_example(f, debug=debug, log_level=log_level, agent_launcher=agent_launcher)

        wrapper.__doc__ = f.__doc__
        return wrapper

    # Support both @example_cli and @example_cli(agent_launcher=...)
    if func is not None:
        return decorator(func)
    return decorator


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
    def run_agent(call_type: str, call_id: Optional[str], debug: bool, log_level: str, no_demo: bool) -> None:
        """Run the agent with the specified configuration."""
        # Configure logging
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Generate call ID if not provided
        if call_id is None:
            call_id = str(uuid4())
        
        async def _run():
            logger = logging.getLogger(__name__)
            logger.info("🚀 Launching agent...")
            
            try:
                # Launch agent with warmup
                agent = await launcher.launch(call_type=call_type, call_id=call_id)
                logger.info("✅ Agent warmed up and ready")
                
                # Open demo UI by default
                if not no_demo and hasattr(agent, 'edge') and hasattr(agent.edge, 'open_demo_for_agent'):
                    logger.info("🌐 Opening demo UI...")
                    await agent.edge.open_demo_for_agent(agent, call_type, call_id)
                
                # Join call if join_call function is provided
                if launcher.join_call:
                    logger.info(f"📞 Joining call: {call_type}/{call_id}")
                    result = launcher.join_call(agent, call_type, call_id)
                    if asyncio.iscoroutine(result):
                        await result
                else:
                    logger.warning("No join_call function provided, agent created but not joined to call")
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal, shutting down gracefully...")
            except Exception as e:
                logger.error(f"❌ Error running agent: {e}", exc_info=True)
                raise
        
        try:
            # Temporarily suppress asyncio error logging during cleanup
            asyncio_logger = logging.getLogger("asyncio")
            original_level = asyncio_logger.level
            
            asyncio.run(_run(), debug=debug)
        except KeyboardInterrupt:
            # Suppress KeyboardInterrupt and asyncio errors during cleanup
            asyncio_logger.setLevel(logging.CRITICAL)
            logger = logging.getLogger(__name__)
            logger.info("👋 Agent shutdown complete")
        finally:
            # Restore original logging level
            if 'asyncio_logger' in locals() and 'original_level' in locals():
                asyncio_logger.setLevel(original_level)
    
    # Invoke the click command
    run_agent()

