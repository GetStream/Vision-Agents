# Local RAG Demo

Demonstrates self-managed RAG with pluggable components.

## What makes it "local"?

Unlike managed RAG (Gemini/OpenAI Vector Store), LocalRAG gives you control:

| Component | What happens |
|-----------|--------------|
| **Chunking** | Done locally by your chosen chunker |
| **Embeddings** | API call returns vectors to you (text not stored remotely) |
| **Vector Store** | Stored in local memory |
| **Search** | Local cosine similarity (no API call) |

Your documents never leave your machine - only text chunks are sent to the embedding API.

## Components

- **Chunkers**: `SentenceChunker`, `FixedSizeChunker`
- **Embeddings**: `OpenAIEmbeddings` (or implement your own)
- **Vector Stores**: `InMemoryVectorStore` (or implement your own)

## Setup

```bash
export OPENAI_API_KEY=your-key-here
```

## Run

```bash
cd examples/other_examples/local_rag_demo
uv run python local_rag_demo.py
```

## Demos included

1. **SentenceChunker** - Natural text boundaries
2. **FixedSizeChunker** - Fixed size with overlap
3. **LLM Integration** - Automatic context injection
4. **File ingestion** - Add files directly

## Making it fully offline

To eliminate all API calls, implement a local embedding provider:

```python
class LocalEmbeddings(EmbeddingProvider):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
    
    @property
    def dimension(self) -> int:
        return 384
    
    async def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()
```

