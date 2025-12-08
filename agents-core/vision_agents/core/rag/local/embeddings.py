"""Embedding providers for local RAG."""

import abc
from typing import Optional


class EmbeddingProvider(abc.ABC):
    """Abstract base class for embedding providers.

    Embedding providers convert text into vector representations
    for semantic similarity search.
    """

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""

    @abc.abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Default implementation calls embed() for each text.
        Override for more efficient batch processing.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return [await self.embed(text) for text in texts]


class OpenAIEmbeddings(EmbeddingProvider):
    """Embedding provider using OpenAI's embedding models.

    Example:
        ```python
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector = await embeddings.embed("Hello world")
        ```
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """Initialize OpenAI embeddings provider.

        Args:
            model: The embedding model to use. Defaults to text-embedding-3-small.
            api_key: Optional API key. By default loads from OPENAI_API_KEY.
            dimensions: Optional dimension for the embeddings (for models that support it).
        """
        from openai import AsyncOpenAI

        self._model = model
        self._dimensions = dimensions
        self._client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()

        # Default dimensions for known models
        self._default_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        if self._dimensions:
            return self._dimensions
        return self._default_dimensions.get(self._model, 1536)

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        kwargs: dict = {"model": self._model, "input": text}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        response = await self._client.embeddings.create(**kwargs)
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently."""
        if not texts:
            return []

        kwargs: dict = {"model": self._model, "input": texts}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        response = await self._client.embeddings.create(**kwargs)

        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
