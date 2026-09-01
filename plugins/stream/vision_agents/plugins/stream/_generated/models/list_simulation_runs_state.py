from enum import StrEnum


class ListSimulationRunsState(StrEnum):
    CANCELLED = "cancelled"
    ERRORED = "errored"
    FAILED = "failed"
    PASSED = "passed"
    RUNNING = "running"

    def __str__(self) -> str:
        return str(self.value)
