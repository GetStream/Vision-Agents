"""
Generic CLI runner for Vision Agents examples.

Provides a Click-based CLI with common options for debugging and logging.
"""

import asyncio
import logging
from typing import Awaitable, Callable

import click


def run_example(
    async_main: Callable[[], Awaitable[None]],
    debug: bool = False,
    log_level: str = "INFO",
) -> None:
    """
    Run an async example with optional debug and logging configuration.

    Args:
        async_main: Async function to run
        debug: Enable debug mode (BlockBuster + asyncio debug)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Enable BlockBuster in debug mode
    if debug:
        try:
            from blockbuster import BlockBuster

            blockbuster = BlockBuster()
            blockbuster.activate()
            logging.info("BlockBuster activated")
        except ImportError:
            logging.warning("BlockBuster not available - install it for debug mode")

    # Run the async main function
    asyncio.run(async_main(), debug=debug)


def example_cli(func: Callable[[], Awaitable[None]]) -> click.Command:
    """
    Decorator to add standard CLI options to an example.

    Usage:
        @example_cli
        async def main():
            # Your example code here
            pass

        if __name__ == "__main__":
            main()
    """

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
        run_example(func, debug=debug, log_level=log_level)

    wrapper.__doc__ = func.__doc__
    return wrapper

