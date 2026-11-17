from typing import Optional


def sanitize_text(text: str) -> str:
    """Remove markdown and special characters that don't speak well."""
    return text.replace("*", "").replace("#", "")


def mask_phone_number(number: Optional[str]) -> str:
    """Hide all but the last four digits of a phone number, for logs."""
    if not number:
        return "unknown"
    digits = sum(character.isdigit() for character in number)
    hidden = digits - 4 if digits > 4 else digits
    masked = []
    for character in number:
        if character.isdigit() and hidden > 0:
            masked.append("*")
            hidden -= 1
        else:
            masked.append(character)
    return "".join(masked)
