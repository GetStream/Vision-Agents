"""Unit tests for the platform-conditional liteav bindings module."""

import pytest

from vision_agents.plugins.tencent.bindings import (
    LITEAV_IMPORT_ERROR,
    require_liteav,
)


class TestRequireLiteav:
    @pytest.mark.skipif(
        LITEAV_IMPORT_ERROR is None,
        reason="liteav is installed; the friendly-error branch can't trigger here",
    )
    def test_raises_with_user_facing_message_on_unsupported_platform(self) -> None:
        with pytest.raises(RuntimeError) as excinfo:
            require_liteav()
        msg = str(excinfo.value)
        # The message must point the user at the install path and at the
        # Docker fallback — both regress easily if someone "simplifies" it.
        assert "manylinux" in msg
        assert "Linux" in msg
        assert "plugins/tencent/README.md" in msg

    @pytest.mark.skipif(
        LITEAV_IMPORT_ERROR is not None,
        reason="liteav is not installed; require_liteav() can't return cleanly",
    )
    def test_returns_silently_when_liteav_is_available(self) -> None:
        # No raise, no return value — just exercises the no-op path on
        # Linux so the test suite has at least one assertion that proves
        # require_liteav() doesn't false-positive when the SDK is loaded.
        assert require_liteav() is None
