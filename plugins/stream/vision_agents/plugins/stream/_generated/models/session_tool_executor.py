from enum import StrEnum


class SessionToolExecutor(StrEnum):
    SESSION_TOOL_EXECUTOR_CLIENT = "client"
    SESSION_TOOL_EXECUTOR_SERVER = "server"

    def __str__(self) -> str:
        return str(self.value)
