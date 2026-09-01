from enum import StrEnum


class SimulationCaseEnded(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"
    TIMEOUT = "timeout"
    TURNS = "turns"

    def __str__(self) -> str:
        return str(self.value)
