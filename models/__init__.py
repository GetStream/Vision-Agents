"""
Models Package

This package provides AI model interfaces and implementations for Stream Agents.
"""

from .model import Model
from .openai import OpenAIModel

__all__ = ["Model", "OpenAIModel"]

__version__ = "0.1.0"
