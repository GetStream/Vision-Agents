import asyncio

from stream_agents.core.agents import Agent


class AgentSessionContextManager:
    def __init__(self, agent: Agent, connection_cm=None):
        self.agent = agent
        self._connection_cm = connection_cm

    def __enter__(self):
        # return a resource if needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
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
