from enum import StrEnum


class ImageContentPartType(StrEnum):
    IMAGE_URL = "image_url"

    def __str__(self) -> str:
        return str(self.value)
