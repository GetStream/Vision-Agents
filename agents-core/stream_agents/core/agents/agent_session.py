from stream_agents.core.agents import Agent


class AgentSessionContextManager:
    def __init__(self, agent: Agent):
        self.agent = agent

    def __enter__(self):
        print("Entering context")
        # return a resource if needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting context")

        self.agent.close()
        # Handle exceptions if any
        if exc_type:
            print(f"An exception occurred: {exc_value}")
        # Return True to suppress exception, False (or None) to propagate it

        return False
