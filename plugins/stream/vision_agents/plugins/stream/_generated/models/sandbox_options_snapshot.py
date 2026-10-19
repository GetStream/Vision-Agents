from enum import StrEnum


class SandboxOptionsSnapshot(StrEnum):
    SANDBOX_SNAPSHOT_DAYTONA_LARGE = "daytona-large"
    SANDBOX_SNAPSHOT_DAYTONA_MEDIUM = "daytona-medium"
    SANDBOX_SNAPSHOT_DAYTONA_SMALL = "daytona-small"
    SANDBOX_SNAPSHOT_DAYTONA_VM_LARGE = "daytona-vm-large"
    SANDBOX_SNAPSHOT_DAYTONA_VM_MEDIUM = "daytona-vm-medium"
    SANDBOX_SNAPSHOT_DAYTONA_VM_SMALL = "daytona-vm-small"

    def __str__(self) -> str:
        return str(self.value)
