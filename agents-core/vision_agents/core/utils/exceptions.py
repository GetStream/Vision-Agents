import logging
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def log_exceptions(
    logger: logging.Logger,
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
