# Qdrant RAG Plugin

Hybrid search RAG (Retrieval Augmented Generation) using Qdrant's built-in fastembed integration for dense and BM25 sparse embeddings.

## Features

- **Hybrid Search**: Dense vector (semantic) + BM25 sparse (keyword) via native Qdrant RRF fusion
- **fastembed Native**: No external embedding dependencies — Qdrant client handles everything
- **Implements RAG Interface**: Compatible with Vision Agents RAG base class

## Installation

```bash
uv add "vision-agents[qdrant]"
# or directly
uv add vision-agents-plugins-qdrant
```

## Usage

```python
from vision_agents.plugins import qdrant

# Initialize RAG (connects to local Qdrant by default)
rag = qdrant.QdrantRAG(collection="my-knowledge")
await rag.add_directory("./knowledge")

# Hybrid search (default)
results = await rag.search("How does the chat API work?")

# Vector-only search
results = await rag.search("How does the chat API work?", mode="vector")

# BM25 search
results = await rag.search("chat API pricing", mode="bm25")

# Or use convenience function
rag = await qdrant.create_rag(
    collection="product-knowledge",
    knowledge_dir="./knowledge"
)
```

## Configuration

| Parameter      | Description                                   | Default                                    |
|----------------|-----------------------------------------------|--------------------------------------------|
| `collection`   | Qdrant collection name                        | Required                                   |
| `url`          | Qdrant server URL                             | `http://localhost:6333`                    |
| `api_key`      | Qdrant API key (for Qdrant Cloud)             | `QDRANT_API_KEY` env var                   |
| `dense_model`  | fastembed dense model for semantic search     | `sentence-transformers/all-MiniLM-L6-v2`  |
| `sparse_model` | fastembed sparse model for BM25 search        | `Qdrant/bm25`                              |
| `chunk_size`   | Size of text chunks for splitting documents   | `10000`                                    |
| `chunk_overlap`| Overlap between chunks for context continuity | `200`                                      |
| `cloud_inference` | Use Qdrant Cloud server-side inference instead of local fastembed | `False`               |

## Environment Variables

- `QDRANT_API_KEY`: Qdrant API key (for Qdrant Cloud; not needed for local)

## Running Qdrant locally

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Dependencies

- `qdrant-client[fastembed]`: Qdrant async client with built-in fastembed support
- `langchain-text-splitters`: Text chunking utilities

## References

- [Qdrant Hybrid Queries](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [fastembed Models](https://qdrant.github.io/fastembed/examples/Supported_Models/)
