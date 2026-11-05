from enum import StrEnum


class KnowledgeDocumentState(StrEnum):
    FAILED = "failed"
    INDEXED = "indexed"
    PENDING = "pending"

    def __str__(self) -> str:
        return str(self.value)
