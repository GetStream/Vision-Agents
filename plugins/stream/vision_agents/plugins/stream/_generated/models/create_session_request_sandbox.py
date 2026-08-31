from enum import StrEnum


class CreateSessionRequestSandbox(StrEnum):
    DAYTONA = "daytona"

    def __str__(self) -> str:
        return str(self.value)
