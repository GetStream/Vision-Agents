from enum import Enum


class Tier(str, Enum):
    HIGH_QUALITY = "high-quality"
    LOW_LATENCY = "low-latency"

    def __str__(self) -> str:
        return str(self.value)
