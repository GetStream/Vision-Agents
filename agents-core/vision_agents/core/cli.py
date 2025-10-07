#!/usr/bin/env python3
"""
Stream Agents CLI

A command-line interface for managing Stream Agents with configurable logging
and lifecycle management.
"""

import asyncio
import logging
from typing import Any, Callable, Coroutine

import click


# Global shutdown event
shutdown_event = asyncio.Event()


def setup_logging(log_level: str) -> None:
    """Configure logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set specific logger levels
    logging.getLogger("vision_agents").setLevel(numeric_level)
    logging.getLogger("test123").setLevel(numeric_level)


async def start_dispatcher(
    agent_func: Callable[[], Coroutine[Any, Any, None]],
    *,
    log_level: str = "INFO",
    shutdown_timeout: float = 30.0,
) -> None:
    """
    Start the agent dispatcher with proper lifecycle management.

    Args:
        agent_func: Async function that runs the agent
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        shutdown_timeout: Maximum time to wait for graceful shutdown
    """
    # Setup logging and signal handlers
    setup_logging(log_level)

    logger = logging.getLogger("stream-test123.dispatcher")
    logger.info("🚀 Starting Stream Agents dispatcher...")

    await agent_func()


    logger.info("🔚 Stream Agents dispatcher stopped")


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Set the logging level",
)
@click.pass_context
def cli(ctx: click.Context, log_level: str) -> None:
    """Stream Agents CLI - Manage AI test123 for video calls."""
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level.upper()


@cli.command()
@click.option("--timeout", type=float, default=30.0, help="Shutdown timeout in seconds")
@click.pass_context
def run(ctx: click.Context, timeout: float) -> None:
    """Run a stream agent with proper lifecycle management."""
    log_level = ctx.obj["log_level"]

    # This is a placeholder - in practice, this would be called from your agent scripts
    click.echo(f"🚀 Starting agent with log level: {log_level}")
    click.echo(
        "💡 Use start_dispatcher() in your agent scripts for proper lifecycle management"
    )
    click.echo(f"⏱️ Shutdown timeout: {timeout}s")


if __name__ == "__main__":
    cli()
