import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable

from vision_agents.core.utils.logging import logger


@contextmanager
def log_exceptions(
    logger: logging.Logger | logging.LoggerAdapter,
    message: str,
    *exceptions: type[Exception],
    reraise: bool = False,
) -> Iterator[None]:
    """Catch specified exceptions within the block and log them with traceback.

    Args:
        logger: Logger to emit the traceback to (uses ``logger.exception``).
        message: Message logged alongside the traceback.
            *exceptions: One or more exception classes to catch.
            Defaults to ``Exception``.
        reraise: If True, re-raise after logging.
    """
    if not exceptions:
        exceptions = (Exception,)
    try:
        yield
    except exceptions:
        logger.exception(message)
        if reraise:
            raise


def log_task_exception(message: str) -> Callable[[asyncio.Task], None]:
    """
    A done_callback to log exceptions in the failed tasks
    Args:
        message: An error message to log.

    """

    def wrapper(task: asyncio.Task[None]) -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error(message, exc_info=task.exception())

    return wrapper
