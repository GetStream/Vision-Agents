from enum import StrEnum


class DataChangeOp(StrEnum):
    DELETE = "delete"
    INSERT = "insert"
    UPDATE = "update"

    def __str__(self) -> str:
        return str(self.value)
