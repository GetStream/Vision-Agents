from enum import Enum


class DecisionKind(str, Enum):
    ANSWER = "answer"
    ASK = "ask"
    BACKCHANNEL = "backchannel"
    COMPACT = "compact"
    DELEGATE = "delegate"
    FAIL = "fail"
    IGNORE = "ignore"
    INTERRUPT = "interrupt"
    QUEUE = "queue"
    SHORTEN = "shorten"
    SUPERSEDE = "supersede"
    WAIT = "wait"

    def __str__(self) -> str:
        return str(self.value)
