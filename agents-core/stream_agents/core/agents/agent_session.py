from stream_agents.core.agents import Agent


class AgentSessionContextManager:
    def __init__(self, agent: Agent, connection_cm=None):
        self.agent = agent
        self._connection_cm = connection_cm

    def __enter__(self):
        print("Entering context")
        # return a resource if needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting context")

        # Close RTC connection context if present
        try:
            if self._connection_cm is not None:
                ae = getattr(self._connection_cm, "__aexit__", None)
                if callable(ae):
                    try:
                        import asyncio

                        if asyncio.iscoroutinefunction(ae):
                            asyncio.get_event_loop().run_until_complete(
                                ae(None, None, None)
                            )
                        else:
                            ae(None, None, None)
                    except Exception:
                        pass
        finally:
            # Close agent resources
            try:
                import asyncio

                coro = self.agent.close()
                if asyncio.iscoroutine(coro):
                    asyncio.get_event_loop().run_until_complete(coro)
            except Exception:
                pass

        # Handle exceptions if any
        if exc_type:
            print(f"An exception occurred: {exc_value}")
        # Return True to suppress exception, False (or None) to propagate it

        return False
