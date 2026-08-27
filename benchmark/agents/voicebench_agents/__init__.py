"""Shared helpers for the vertical reference agents."""

from pathlib import Path


def pack_prompt(pack: str) -> str:
    """Return the contract system prompt for a vertical pack."""
    path = Path(__file__).resolve().parent.parent / "contracts" / f"{pack}.prompt"
    return path.read_text(encoding="utf-8").strip()
