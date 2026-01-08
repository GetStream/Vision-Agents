import abc
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """A document to be indexed in the RAG system."""

    text: str
    source: str
    metadata: dict | None = None


class RAG(abc.ABC):
    """
    Abstract base class for RAG (Retrieval Augmented Generation) implementations.

    The full complexities of RAG are beyond the scope of this project.
    We ship with examples including TurboPuffer RAG with hybrid search.

    The documentation explains in greater detail how to build RAG.
    """

    @abc.abstractmethod
    async def add_documents(self, documents: list[Document]) -> int:
        """
        Add documents to the RAG index.

        Args:
            documents: List of documents to index.

        Returns:
            Number of chunks indexed.
        """

    async def add_directory(
        self,
        path: str | Path,
        extensions: list[str] | None = None,
    ) -> int:
        """
        Add all files from a directory to the RAG index.

        Args:
            path: Path to directory containing files.
            extensions: File extensions to include (e.g., ['.md', '.txt']).
                       Defaults to ['.md', '.txt'].

        Returns:
            Total number of chunks indexed.
        """
        directory = Path(path)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        if extensions is None:
            extensions = [".md", ".txt"]

        # Normalize extensions
        extensions = [
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in extensions
        ]

        files = [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]

        if not files:
            return 0

        documents = [Document(text=f.read_text(), source=f.name) for f in files]

        return await self.add_documents(documents)

    @abc.abstractmethod
    async def search(self, query: str, top_k: int = 3) -> str:
        """
        Search the knowledge base.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            Formatted string with search results.
        """

    @abc.abstractmethod
    async def clear(self) -> None:
        """Clear all indexed documents."""

    async def close(self) -> None:
        """Close any open resources. Override if needed."""
