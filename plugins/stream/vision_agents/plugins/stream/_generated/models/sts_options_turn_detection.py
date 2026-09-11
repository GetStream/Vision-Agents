from enum import StrEnum


class StsOptionsTurnDetection(StrEnum):
    NONE = "none"
    SEMANTIC = "semantic"
    SERVER_VAD = "server_vad"

    def __str__(self) -> str:
        return str(self.value)
