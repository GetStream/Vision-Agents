from contextvars import ContextVar, Token
import logging

call_id_ctx: ContextVar[str] = ContextVar("call_id", default="-")
_original_factory = logging.getLogRecordFactory()


def _contextual_record_factory(*args, **kwargs) -> logging.LogRecord:
    """Attach the call ID from context to every log record."""

    record = _original_factory(*args, **kwargs)
    record.call_id = call_id_ctx.get()
    return record


def initialize_logging_context() -> None:
    """Ensure the logging system populates call ID on every record."""

    logging.setLogRecordFactory(_contextual_record_factory)


def set_call_context(call_id: str) -> Token:
    """Store the call ID into the logging context."""

    initialize_logging_context()
    return call_id_ctx.set(call_id)


def clear_call_context(token: Token) -> None:
    """Reset the call context using the provided token."""

    call_id_ctx.reset(token)
