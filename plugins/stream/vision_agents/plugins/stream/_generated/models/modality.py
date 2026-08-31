from enum import StrEnum


class Modality(StrEnum):
    KNOWLEDGE = "knowledge"
    LLM = "llm"
    MEMORY = "memory"
    PHONE = "phone"
    SEARCH = "search"
    STT = "stt"
    TTS = "tts"

    def __str__(self) -> str:
        return str(self.value)
