from enum import StrEnum


class AuditSource(StrEnum):
    API = "api"
    CLI = "cli"
    DASHBOARD = "dashboard"
    SDK = "sdk"

    def __str__(self) -> str:
        return str(self.value)
