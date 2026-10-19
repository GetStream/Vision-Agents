from enum import StrEnum


class SandboxOptionsRegion(StrEnum):
    SANDBOX_REGION_EU = "eu"
    SANDBOX_REGION_US = "us"

    def __str__(self) -> str:
        return str(self.value)
