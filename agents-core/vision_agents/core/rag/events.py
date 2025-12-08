"""RAG-specific events for observability."""

from dataclasses import dataclass, field
from typing import Any, Optional

from vision_agents.core.events import BaseEvent

from .types import RetrievalResult


@dataclass
class RAGRetrievalStartEvent(BaseEvent):
    """Emitted when RAG retrieval begins."""

    type: str = field(default="rag.retrieval.start", init=False)
    plugin_name: str = "rag"
    query: str = ""
    top_k: int = 5


@dataclass
class RAGRetrievalCompleteEvent(BaseEvent):
    """Emitted when RAG retrieval completes."""

    type: str = field(default="rag.retrieval.complete", init=False)
    plugin_name: str = "rag"
    query: str = ""
    results: list[RetrievalResult] = field(default_factory=list)
    retrieval_time_ms: float = 0.0

    @property
    def result_count(self) -> int:
        return len(self.results)


@dataclass
class RAGDocumentAddedEvent(BaseEvent):
    """Emitted when a document is added to the RAG system."""

    type: str = field(default="rag.document.added", init=False)
    plugin_name: str = "rag"
    document_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0


@dataclass
class RAGFileAddedEvent(BaseEvent):
    """Emitted when a file is uploaded to the RAG system."""

    type: str = field(default="rag.file.added", init=False)
    plugin_name: str = "rag"
    file_path: str = ""
    file_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
