from enum import StrEnum


class DecisionKind(StrEnum):
    ANSWER = "answer"
    ASK = "ask"
    BACKCHANNEL = "backchannel"
    COMPACT = "compact"
    DELEGATE = "delegate"
    FAIL = "fail"
    IGNORE = "ignore"
    INTERRUPT = "interrupt"
    QUEUE = "queue"
    SETTLE = "settle"
    SHORTEN = "shorten"
    SUPERSEDE = "supersede"
    WAIT = "wait"

    def __str__(self) -> str:
        return str(self.value)
