"""
Model Protocol

This module defines the Model protocol interface for AI models used in Stream Agents.
"""

from __future__ import annotations

from typing import Protocol, Any, Dict, List, Optional, Union
from abc import abstractmethod


class Model(Protocol):
    """
    Protocol for AI models used in Stream Agents.

    This protocol defines the interface that all AI model implementations
    must follow to be compatible with the Agent system.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the model name."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The input prompt to generate a response for
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            stop: Stop sequences to end generation
            **kwargs: Additional model-specific parameters

        Returns:
            Generated response text

        Raises:
            Exception: If generation fails
        """
        ...

    @abstractmethod
    async def generate_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a chat response from the model.

        Args:
            messages: List of chat messages in format [{"role": "user", "content": "..."}]
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            stop: Stop sequences to end generation
            **kwargs: Additional model-specific parameters

        Returns:
            Generated response text

        Raises:
            Exception: If generation fails
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> Any:  # AsyncIterator[str] - avoiding import for simplicity
        """
        Generate a streaming response from the model.

        Args:
            prompt: The input prompt to generate a response for
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            stop: Stop sequences to end generation
            **kwargs: Additional model-specific parameters

        Yields:
            Chunks of generated response text

        Raises:
            Exception: If generation fails
        """
        ...

    @abstractmethod
    async def generate_chat_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> Any:  # AsyncIterator[str] - avoiding import for simplicity
        """
        Generate a streaming chat response from the model.

        Args:
            messages: List of chat messages in format [{"role": "user", "content": "..."}]
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            stop: Stop sequences to end generation
            **kwargs: Additional model-specific parameters

        Yields:
            Chunks of generated response text

        Raises:
            Exception: If generation fails
        """
        ...
