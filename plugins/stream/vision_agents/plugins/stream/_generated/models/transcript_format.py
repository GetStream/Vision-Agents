from enum import StrEnum


class TranscriptFormat(StrEnum):
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"

    def __str__(self) -> str:
        return str(self.value)
