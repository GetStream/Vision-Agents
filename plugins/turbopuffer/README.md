# TurboPuffer RAG Plugin

Hybrid search RAG (Retrieval Augmented Generation) implementation using TurboPuffer for vector + BM25 full-text search, with Gemini for embeddings.

## Features

- **Hybrid Search**: Combines vector (semantic) and BM25 (keyword) search for better retrieval quality
- **Reciprocal Rank Fusion**: Merges results from multiple search strategies
- **Gemini Embeddings**: Uses Google's Gemini embedding model for high-quality vectors
- **Low-latency Queries**: Supports cache warming for fast query responses
- **Implements RAG Interface**: Compatible with Vision Agents RAG base class

## Installation

```bash
uv add vision-agents[turbopuffer]
```

## Usage

```python
from vision_agents.plugins import turbopuffer

# Initialize RAG
rag = turbopuffer.TurboPufferRAG(namespace="my-knowledge")
await rag.add_directory("./knowledge")

# Hybrid search (default)
results = await rag.search("How does the chat API work?")

# Vector-only search
results = await rag.search("How does the chat API work?", mode="vector")

# BM25-only search  
results = await rag.search("chat API pricing", mode="bm25")

# Or use convenience function
rag = await turbopuffer.create_rag(
    namespace="product-knowledge",
    knowledge_dir="./knowledge"
)
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `namespace` | TurboPuffer namespace for storing vectors | Required |
| `embedding_model` | Gemini embedding model | `models/gemini-embedding-001` |
| `chunk_size` | Size of text chunks for splitting documents | `10000` |
| `chunk_overlap` | Overlap between chunks for context continuity | `200` |
| `region` | TurboPuffer region | `gcp-us-central1` |

## Environment Variables

- `TURBO_PUFFER_KEY`: TurboPuffer API key
- `GOOGLE_API_KEY`: Google API key (for Gemini embeddings)

## Dependencies

- `turbopuffer`: TurboPuffer vector database client
- `langchain-google-genai`: Gemini embeddings
- `langchain-text-splitters`: Text chunking utilities

## References

- [TurboPuffer Hybrid Search](https://turbopuffer.com/docs/hybrid)
- [MTEB Embedding Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
