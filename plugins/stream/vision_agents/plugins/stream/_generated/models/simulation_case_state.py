from enum import StrEnum


class SimulationCaseState(StrEnum):
    CANCELLED = "cancelled"
    ERRORED = "errored"
    FAILED = "failed"
    PASSED = "passed"
    PENDING = "pending"
    RUNNING = "running"

    def __str__(self) -> str:
        return str(self.value)
