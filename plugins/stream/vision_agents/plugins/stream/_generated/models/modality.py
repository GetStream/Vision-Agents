from enum import StrEnum


class Modality(StrEnum):
    DECISION_MODEL = "decision_model"
    IMAGE = "image"
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
