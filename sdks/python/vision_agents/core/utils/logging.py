import logging
import sys
from contextvars import ContextVar, Token
from dataclasses import dataclass

import colorlog
from uvicorn.logging import AccessFormatter

logger = logging.getLogger(__name__)

call_id_ctx: ContextVar[str | None] = ContextVar("call_id", default=None)
_CURRENT_CALL_ID: str | None = None
_ORIGINAL_FACTORY = logging.getLogRecordFactory()
_CALL_ID_ENABLED = True


MAIN_LOGGER = logging.getLogger("vision_agents")


def _get_colored_formatter(
    fmt: str, datefmt: str = "%H:%M:%S"
) -> colorlog.ColoredFormatter:
    return colorlog.ColoredFormatter(
        fmt=fmt,
        datefmt=datefmt,
        log_colors={
            "DEBUG": "white",
            "INFO": "light_white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )


def configure_sdk_logger(level: int) -> None:
    """
    Sets the default configuration for "vision_agents" logger to have a nice formatting
    if it's not already configured.
    """

    default_handler = logging.StreamHandler(sys.stderr)
    default_handler.setLevel(level)
    formatter = _get_colored_formatter(
        fmt="%(log_color)s%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s"
    )
    default_handler.setFormatter(formatter)

    for _logger in [MAIN_LOGGER]:
        # Set the default handler only if it's not already configured
        if not _logger.handlers:
            _logger.handlers = [default_handler]
            _logger.propagate = False
        # Do not override the level if it's set
        if _logger.level == logging.NOTSET:
            _logger.setLevel(level)


def configure_fastapi_loggers(level: int) -> None:
    """
    Configure fastapi loggers with the provided level to follow the same format if it's not already configured.
    Args:
        level: log level as int

    Returns:
        None
    """
    # Make the access log format look similar to the other logs
    access_log_formatter = AccessFormatter(
        fmt='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(client_addr)s | "%(request_line)s" | %(status_code)s',
        datefmt="%H:%M:%S",
    )
    generic_formatter = _get_colored_formatter(
        fmt="%(log_color)s%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s"
    )

    loggers_formatters = [
        (logging.getLogger("fastapi"), generic_formatter),
        (logging.getLogger("uvicorn"), generic_formatter),
        (logging.getLogger("uvicorn.access"), access_log_formatter),
        (logging.getLogger("uvicorn.error"), generic_formatter),
    ]

    for _logger, formatter in loggers_formatters:
        default_handler = logging.StreamHandler(sys.stderr)
        default_handler.setLevel(level)
        default_handler.setFormatter(formatter)

        # Set the default handler only if it's not already configured
        if not _logger.handlers:
            _logger.handlers = [default_handler]
            _logger.propagate = False
        # Do not override the level if it's set
        if _logger.level == logging.NOTSET:
            _logger.setLevel(level)


@dataclass(slots=True)
class CallContextToken:
    """Token capturing prior state for restoring logging context."""

    context_token: Token
    previous_global: str | None


def _contextual_record_factory(*args, **kwargs) -> logging.LogRecord:
    """Attach the call ID from context to every log record."""
    if not _CALL_ID_ENABLED:
        return _ORIGINAL_FACTORY(*args, **kwargs)

    record = _ORIGINAL_FACTORY(*args, **kwargs)
    call_id = call_id_ctx.get()
    if not call_id:
        call_id = _CURRENT_CALL_ID

    message = record.getMessage()

    if call_id:
        record.msg = f"[call:{call_id}] {message}"
        record.args = ()
    else:
        record.msg = message
        record.args = ()

    record.call_id = call_id or "-"
    return record


def set_call_context(call_id: str) -> CallContextToken:
    """Store the call ID into the logging context."""

    global _CURRENT_CALL_ID

    token = CallContextToken(
        context_token=call_id_ctx.set(call_id),
        previous_global=_CURRENT_CALL_ID,
    )
    _CURRENT_CALL_ID = call_id
    return token


def clear_call_context(token: CallContextToken) -> None:
    """Reset the call context using the provided token."""

    global _CURRENT_CALL_ID

    # failing TODO: fix
    # call_id_ctx.reset(token.context_token)
    _CURRENT_CALL_ID = token.previous_global


def configure_call_id_logging(enabled: bool) -> None:
    """Configure whether call ID logging is enabled."""

    logger.info(f"Configuring call ID logging to {enabled}")
    global _CALL_ID_ENABLED
    _CALL_ID_ENABLED = enabled
