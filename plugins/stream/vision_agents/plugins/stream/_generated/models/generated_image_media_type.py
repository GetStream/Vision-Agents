from enum import StrEnum


class GeneratedImageMediaType(StrEnum):
    IMAGEJPEG = "image/jpeg"
    IMAGEPNG = "image/png"

    def __str__(self) -> str:
        return str(self.value)
