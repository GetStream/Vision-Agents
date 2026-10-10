from enum import StrEnum


class InvocationArgumentType(StrEnum):
    ARRAY = "array"
    BOOLEAN = "boolean"
    NULL = "null"
    NUMBER = "number"
    OBJECT = "object"
    STRING = "string"

    def __str__(self) -> str:
        return str(self.value)
