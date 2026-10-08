from enum import StrEnum


class StreamWritesInto(StrEnum):
    DEPLOYMENT_APP = "deployment_app"
    NOWHERE = "nowhere"
    THIS_APP = "this_app"

    def __str__(self) -> str:
        return str(self.value)
