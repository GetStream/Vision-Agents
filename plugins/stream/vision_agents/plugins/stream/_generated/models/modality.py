from enum import StrEnum


class Modality(StrEnum):
    DECISION_MODEL = "decision_model"
    DETECTION = "detection"
    IMAGE = "image"
    KNOWLEDGE = "knowledge"
    LLM = "llm"
    MEMORY = "memory"
    OCR = "ocr"
    PHONE = "phone"
    SEARCH = "search"
    STS = "sts"
    STT = "stt"
    TTS = "tts"

    def __str__(self) -> str:
        return str(self.value)
