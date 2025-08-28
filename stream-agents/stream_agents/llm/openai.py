"""
OpenAI Model Implementation

This module provides an OpenAI implementation of the Model protocol.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union, AsyncIterator

from openai import OpenAI, AsyncOpenAI








class OpenAILLM:
    """
    OpenAI implementation of the Model protocol.

    This class provides a lightweight wrapper around the OpenAI Python SDK
    that implements the Model protocol for use with Stream Agents.

    Example usage:
        # Using environment variables (OPENAI_API_KEY)
        model = OpenAIModel(name="gpt-4o")

        # Using explicit client
        client = OpenAI(api_key="your-api-key")
        model = OpenAIModel(name="gpt-4o", client=client)

        # Using async client
        async_client = AsyncOpenAI(api_key="your-api-key")
        model = OpenAIModel(name="gpt-4o", client=async_client)
    """

    def __init__(
        self,
        name: str,
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        instructions: Optional[str] = None,
        # Default generation parameters
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        default_stop: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI model.

        Args:
            name: Model name (e.g., "gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini")
            client: Pre-configured OpenAI or AsyncOpenAI client
            api_key: OpenAI API key (if not using pre-configured client)
            base_url: Base URL for OpenAI API (if not using pre-configured client)
            organization: OpenAI organization ID (if not using pre-configured client)
            project: OpenAI project ID (if not using pre-configured client)
            instructions: System instructions for the model
            default_temperature: Default temperature for generation
            default_max_tokens: Default max tokens for generation
            default_stop: Default stop sequences for generation
            **kwargs: Additional client configuration options
        """
        self._name = name
        self.instructions = instructions or "You are a helpful AI assistant."
        self.logger = logging.getLogger(f"OpenAIModel[{name}]")
        
        # Mark this as a traditional LLM (not STS)
        self.sts = False

        # Store default parameters
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._default_stop = default_stop

        if client is not None:
            self._client = client
            # Check if it's an async client (including mocks)
            self._is_async = (
                isinstance(client, AsyncOpenAI)
                or hasattr(client, "__aenter__")
                or str(type(client).__name__).startswith("AsyncMock")
            )
        else:
            # Filter out generation parameters from client kwargs
            client_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ["temperature", "max_tokens", "stop"]
            }

            # Create async client by default for better compatibility
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                project=project,
                **client_kwargs,
            )
            self._is_async = True

        self.logger.info(f"Initialized OpenAI model: {name}")

    async def conversation_started(self, agent):
        """Called when the conversation starts."""
        try:
            response = await self.generate("Say a brief greeting to welcome the user to the call.")
            if hasattr(agent, 'tts') and agent.tts:
                await agent.tts.send(response)
        except Exception as e:
            self.logger.error(f"Error in conversation_started: {e}")

    @property
    def name(self) -> str:
        """Get the model name."""
        return self._name

    @property
    def client(self) -> Union[OpenAI, AsyncOpenAI]:
        """Get the underlying OpenAI client."""
        return self._client

    @property
    def is_async(self) -> bool:
        """Check if the client is async."""
        return self._is_async

    async def generate(
        self,
        prompt: Union[str, List[Any]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from the model using a simple prompt or structured input.

        Args:
            prompt: The input prompt (string) or list of ResponseInputItemParam objects
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            stop: Stop sequences to end generation
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Generated response text

        Raises:
            Exception: If generation fails
        """
        if isinstance(prompt, str):
            # Convert simple prompt to chat format with system instructions
            messages = [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": prompt}
            ]
        else:
            # Handle list of ResponseInputItemParam objects
            # Check if this looks like the official OpenAI responses.create format
            has_image_content = any(
                isinstance(item, dict) and 
                isinstance(item.get("content"), list) and
                any(content_item.get("type") == "input_image" for content_item in item.get("content", []))
                for item in prompt
            )
            
            if has_image_content:
                # Use responses.create for image inputs
                return await self._generate_with_responses_api(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    **kwargs,
                )
            else:
                # Convert to chat format for text-only inputs
                messages = [{"role": "system", "content": self.instructions}]
                
                for item in prompt:
                    if isinstance(item, dict):
                        # Convert ResponseInputItemParam to chat message format
                        if item.get("type") == "message":
                            messages.append({
                                "role": item.get("role", "user"),
                                "content": item.get("content", "")
                            })
                        else:
                            # Handle other types if needed
                            self.logger.warning(f"Unknown input item type: {item.get('type')}")
                    elif isinstance(item, str):
                        # Handle string items
                        messages.append({"role": "user", "content": item})
                    else:
                        self.logger.warning(f"Unknown input item format: {type(item)}")
                
                return await self.generate_chat(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    **kwargs,
                )

    async def _generate_with_responses_api(
        self,
        input_items: List[Any],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response using OpenAI's responses.create API for image inputs.

        Args:
            input_items: List of input items in OpenAI responses.create format
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            stop: Stop sequences to end generation
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Generated response text

        Raises:
            Exception: If generation fails
        """
        try:
            # Prepare parameters with defaults
            params = {"model": self._name, "input": input_items, **kwargs}

            # Use provided parameters or fall back to defaults
            if max_tokens is not None:
                params["max_output_tokens"] = max_tokens
            elif self._default_max_tokens is not None:
                params["max_output_tokens"] = self._default_max_tokens

            if temperature is not None:
                params["temperature"] = temperature
            elif self._default_temperature is not None:
                params["temperature"] = self._default_temperature

            # Note: responses.create doesn't support stop sequences in the same way
            # as chat completions, so we skip that parameter

            self.logger.debug(
                f"Generating response with responses.create API for {len(input_items)} input items"
            )

            # Make the API call
            if self._is_async:
                response = await self._client.responses.create(**params)
            else:
                # For sync client, we need to wrap in an async context
                import asyncio

                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._client.responses.create(**params)
                )

            # Extract the response content
            if hasattr(response, 'output_text') and response.output_text:
                content = response.output_text
                self.logger.debug(f"Generated response: {len(content)} characters")
                return content
            elif hasattr(response, 'output') and response.output:
                # Handle the response format with output array
                output_items = response.output
                if output_items and len(output_items) > 0:
                    first_output = output_items[0]
                    if hasattr(first_output, 'content') and first_output.content:
                        content_items = first_output.content
                        if content_items and len(content_items) > 0:
                            first_content = content_items[0]
                            if hasattr(first_content, 'text'):
                                content = first_content.text
                                self.logger.debug(f"Generated response: {len(content)} characters")
                                return content
                
                self.logger.warning("No text content found in response output")
                return str(response)
            else:
                self.logger.warning("No content in responses.create response")
                return ""

        except Exception as e:
            self.logger.error(f"Error generating response with responses.create: {e}")
            raise

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
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Generated response text

        Raises:
            Exception: If generation fails
        """
        try:
            # Prepare parameters with defaults
            params = {"model": self._name, "messages": messages, **kwargs}

            # Use provided parameters or fall back to defaults
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            elif self._default_max_tokens is not None:
                params["max_tokens"] = self._default_max_tokens

            if temperature is not None:
                params["temperature"] = temperature
            elif self._default_temperature is not None:
                params["temperature"] = self._default_temperature

            if stop is not None:
                params["stop"] = stop
            elif self._default_stop is not None:
                params["stop"] = self._default_stop

            self.logger.debug(
                f"Generating chat completion with {len(messages)} messages"
            )

            # Make the API call
            if self._is_async:
                response = await self._client.chat.completions.create(**params)
            else:
                # For sync client, we need to wrap in an async context
                import asyncio

                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._client.chat.completions.create(**params)
                )

            # Extract the response content
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                self.logger.debug(f"Generated response: {len(content)} characters")
                return content
            else:
                self.logger.warning("No content in response")
                return ""

        except Exception as e:
            self.logger.error(f"Error generating chat response: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from the model using a simple prompt.

        Args:
            prompt: The input prompt to generate a response for
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            stop: Stop sequences to end generation
            **kwargs: Additional OpenAI-specific parameters

        Yields:
            Chunks of generated response text

        Raises:
            Exception: If generation fails
        """
        # Convert simple prompt to chat format
        messages = [{"role": "user", "content": prompt}]
        async for chunk in self.generate_chat_stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs,
        ):
            yield chunk

    async def generate_chat_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming chat response from the model.

        Args:
            messages: List of chat messages in format [{"role": "user", "content": "..."}]
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            stop: Stop sequences to end generation
            **kwargs: Additional OpenAI-specific parameters

        Yields:
            Chunks of generated response text

        Raises:
            Exception: If generation fails
        """
        try:
            # Prepare parameters with defaults
            params = {
                "model": self._name,
                "messages": messages,
                "stream": True,
                **kwargs,
            }

            # Use provided parameters or fall back to defaults
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            elif self._default_max_tokens is not None:
                params["max_tokens"] = self._default_max_tokens

            if temperature is not None:
                params["temperature"] = temperature
            elif self._default_temperature is not None:
                params["temperature"] = self._default_temperature

            if stop is not None:
                params["stop"] = stop
            elif self._default_stop is not None:
                params["stop"] = self._default_stop

            self.logger.debug(
                f"Generating streaming chat completion with {len(messages)} messages"
            )

            # Make the streaming API call
            if self._is_async:
                stream = await self._client.chat.completions.create(**params)
            else:
                # For sync client, we need to wrap in an async context
                import asyncio

                stream = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._client.chat.completions.create(**params)
                )

            # Process the stream
            if self._is_async:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                # Handle sync stream in async context
                import asyncio

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                        # Allow other async tasks to run
                        await asyncio.sleep(0)

        except Exception as e:
            self.logger.error(f"Error generating streaming chat response: {e}")
            raise

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"OpenAIModel(name='{self._name}', async={self._is_async})"
