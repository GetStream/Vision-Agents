import asyncio

from vision_agents.core.agents import Agent


class AgentSessionContextManager:
    """Context manager that owns an `Agent` session lifecycle.

    This wrapper keeps the underlying RTC connection (if any) open for the
    duration of the context, and guarantees a best-effort asynchronous cleanup of
    both the RTC connection and the `Agent` resources on exit.

    It accepts an optional connection context manager (e.g., a WebRTC join
    context) that may implement an async `__aexit__`. On exit, we shield the
    asynchronous teardown so it completes even as the loop shuts down. Any
    exception that caused the context to exit is propagated.

    Typical usage:
        agent = Agent(...)
        with await agent.join(call):
            await agent.finish()

    Args:
        agent: The `Agent` whose resources and event wiring should be managed.
        connection_cm: Optional provider-specific connection context manager
            returned by the edge transport (kept open during the context).
    """
    def __init__(self, agent: Agent, connection_cm=None):
        self.agent = agent
        self._connection_cm = connection_cm

    def __enter__(self):
        """Enter the session context.

        Returns:
            AgentSessionContextManager: The context manager itself.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the session context and trigger cleanup.

        Ensures the provider connection (if provided) and the `Agent` are closed.
        Cleanup coroutines are shielded so they are not cancelled by loop
        shutdown. Exceptions are not suppressed.

        Args:
            exc_type: Exception type causing exit, if any.
            exc_value: Exception instance, if any.
            traceback: Traceback object, if any.

        Returns:
            False, so any exception is propagated to the caller.
        """
        loop = asyncio.get_running_loop()

        # ------------------------------------------------------------------
        # Close the RTC connection context if one was started.
        # ------------------------------------------------------------------
        if self._connection_cm is not None:
            aexit = getattr(self._connection_cm, "__aexit__", None)
            if aexit is not None:
                if asyncio.iscoroutinefunction(aexit):
                    # Shield the aexit coroutine so it runs to completion even if the loop is closing.
                    asyncio.shield(loop.create_task(aexit(None, None, None)))
                else:
                    # Fallback for a sync __aexit__ (unlikely, but safe).
                    aexit(None, None, None)

        # ------------------------------------------------------------------
        # Close the agent's own resources.
        # ------------------------------------------------------------------
        if hasattr(self, "agent") and self.agent is not None:
            coro = self.agent.close()
            if asyncio.iscoroutine(coro):
                # Shield the close coroutine so it runs to completion even if the loop is closing.
                if loop.is_running():
                    asyncio.shield(loop.create_task(coro))
                else:
                    # If we are outside the loop, we can block until done.
                    loop.run_until_complete(coro)

        # ------------------------------------------------------------------
        # Handle any exception that caused the context manager to exit.
        # ------------------------------------------------------------------
        if exc_type:
            print(f"An exception occurred: {exc_value}")

        # Returning False propagates the exception (if any); True would suppress it.
        return False
