import asyncio
import logging

import pytest

from vision_agents.core.utils.exceptions import log_exceptions


logger = logging.getLogger("test_log_exceptions")


class TestLogExceptions:
    def test_catches_exception_by_default(self, caplog):
        """A subclass of Exception inside the block is caught and logged, not propagated."""
        with caplog.at_level(logging.ERROR):
            with log_exceptions(logger, "boom happened"):
                raise ValueError("boom")
        # Reaching this line means the exception was swallowed.
        assert any("boom happened" in r.message for r in caplog.records)
        assert any(r.exc_info is not None for r in caplog.records)

    def test_does_not_log_when_block_completes_cleanly(self, caplog):
        with caplog.at_level(logging.ERROR):
            with log_exceptions(logger, "should not appear"):
                pass
        assert not any("should not appear" in r.message for r in caplog.records)

    def test_reraise_propagates_after_logging(self, caplog):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="boom"):
                with log_exceptions(logger, "boom happened", reraise=True):
                    raise ValueError("boom")
        assert any("boom happened" in r.message for r in caplog.records)

    def test_only_catches_specified_exception_types(self, caplog):
        """Exceptions not in the configured tuple must propagate."""
        with caplog.at_level(logging.ERROR):
            with pytest.raises(KeyError):
                with log_exceptions(logger, "would log", ValueError):
                    raise KeyError("not_caught")
        assert not any("would log" in r.message for r in caplog.records)

    def test_catches_any_specified_exception_type(self, caplog):
        with caplog.at_level(logging.ERROR):
            with log_exceptions(logger, "caught it", ValueError, KeyError):
                raise KeyError("inner")
        assert any("caught it" in r.message for r in caplog.records)

    def test_base_exception_subclasses_propagate_by_default(self, caplog):
        """Default filter is ``Exception``; ``BaseException`` subclasses must pass through.

        This is load-bearing for callers that rely on ``CancelledError`` /
        ``KeyboardInterrupt`` reaching outer scopes during cancellation.
        """
        with caplog.at_level(logging.ERROR):
            with pytest.raises(KeyboardInterrupt):
                with log_exceptions(logger, "should not log"):
                    raise KeyboardInterrupt
        assert not any("should not log" in r.message for r in caplog.records)

    def test_cancelled_error_propagates_by_default(self, caplog):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(asyncio.CancelledError):
                with log_exceptions(logger, "should not log"):
                    raise asyncio.CancelledError()
        assert not any("should not log" in r.message for r in caplog.records)
