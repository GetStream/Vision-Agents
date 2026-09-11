from enum import StrEnum


class Modality(StrEnum):
    KNOWLEDGE = "knowledge"
    LLM = "llm"
    MEMORY = "memory"
    PHONE = "phone"
    SEARCH = "search"
    STS = "sts"
    STT = "stt"
    TTS = "tts"

    def __str__(self) -> str:
        return str(self.value)
