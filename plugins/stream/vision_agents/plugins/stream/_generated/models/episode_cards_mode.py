from enum import StrEnum


class EpisodeCardsMode(StrEnum):
    OFF = "off"
    ON = "on"
    RECORD = "record"

    def __str__(self) -> str:
        return str(self.value)
