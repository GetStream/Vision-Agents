def sanitize_text(text: str) -> str:
    """Remove markdown and special characters that don't speak well."""
    return text.replace("*", "").replace("#", "")
