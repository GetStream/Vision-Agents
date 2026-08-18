from enum import Enum


class Modality(str, Enum):
    LLM = "llm"
    MEMORY = "memory"
    PHONE = "phone"
    STT = "stt"
    TTS = "tts"

    def __str__(self) -> str:
        return str(self.value)
