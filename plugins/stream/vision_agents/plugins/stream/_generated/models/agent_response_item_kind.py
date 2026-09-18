from enum import StrEnum


class AgentResponseItemKind(StrEnum):
    ANSWER = "answer"
    BLOCKED = "blocked"
    ERROR = "error"
    SAID = "said"
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    def __str__(self) -> str:
        return str(self.value)
