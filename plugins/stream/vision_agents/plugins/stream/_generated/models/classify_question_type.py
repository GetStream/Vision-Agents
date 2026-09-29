from enum import StrEnum


class ClassifyQuestionType(StrEnum):
    CHOICE = "choice"
    NOUL = "noul"
    SCORE = "score"

    def __str__(self) -> str:
        return str(self.value)
