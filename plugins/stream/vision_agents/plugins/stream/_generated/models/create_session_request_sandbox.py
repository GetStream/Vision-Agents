from enum import Enum


class CreateSessionRequestSandbox(str, Enum):
    DAYTONA = "daytona"

    def __str__(self) -> str:
        return str(self.value)
