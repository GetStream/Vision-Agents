import asyncio

import pytest
from vision_agents.core.utils.logging import (
    CallContextToken,
    call_id_ctx,
    clear_call_context,
    set_call_context,
)


class TestCallContext:
    @pytest.fixture(autouse=True)
    def _reset_call_state(self):
        """Snapshot and restore module-level call state around each test."""
        import vision_agents.core.utils.logging as logging_mod

        previous_context_value = call_id_ctx.get()
        previous_global = logging_mod._CURRENT_CALL_ID
        try:
            yield
        finally:
            call_id_ctx.set(previous_context_value)
            logging_mod._CURRENT_CALL_ID = previous_global

    def test_set_call_context_returns_token_capturing_prior_state(self):
        token = set_call_context("call-A")

        assert isinstance(token, CallContextToken)
        assert token.previous_context is None
        assert token.previous_global is None

    def test_set_call_context_makes_call_id_visible(self):
        set_call_context("call-A")

        assert call_id_ctx.get() == "call-A"

    def test_clear_call_context_restores_unset_state(self):
        token = set_call_context("call-A")

        clear_call_context(token)

        assert call_id_ctx.get() is None

    def test_clear_call_context_restores_global_call_id(self):
        import vision_agents.core.utils.logging as logging_mod

        token = set_call_context("call-A")

        clear_call_context(token)

        assert logging_mod._CURRENT_CALL_ID is None

    def test_nested_set_clear_restores_outer_call_id(self):
        outer = set_call_context("outer")
        inner = set_call_context("inner")

        assert call_id_ctx.get() == "inner"
        clear_call_context(inner)
        assert call_id_ctx.get() == "outer"
        clear_call_context(outer)
        assert call_id_ctx.get() is None

    async def test_clear_works_in_different_task_than_set(self):
        """The set_call_context return value must be usable from a sibling
        asyncio task. Tokens that bind to the originating contextvars.Context
        cannot be reset from a different task; this test pins that the
        public API does not regress to that behaviour."""

        token_holder: list[CallContextToken] = []

        async def setter():
            token_holder.append(set_call_context("call-A"))

        async def clearer():
            clear_call_context(token_holder[0])

        await asyncio.create_task(setter())
        await asyncio.create_task(clearer())

        # No exception raised. In whatever context observes the contextvar
        # next, the cleared value is what set_call_context recorded as prior.
        assert call_id_ctx.get() is None
