# OpenAI RAG Demo

Demonstrates Retrieval-Augmented Generation using OpenAI's Vector Store API.

## Features

- Upload documents to OpenAI's managed Vector Store
- Automatic chunking and embedding by OpenAI
- Semantic search with relevance scores
- Automatic context injection into LLM queries
- Event-based observability

## Setup

```bash
# Set your API key
export OPENAI_API_KEY=your-key-here

# Or use a .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

## Run

```bash
cd examples/other_examples/openai_rag_demo
uv run python openai_rag_demo.py
```

## How it Works

1. Creates sample documents (company policy, product FAQ, technical specs)
2. Uploads them to OpenAI's Vector Store
3. Attaches the RAG provider to a GPT-4o-mini LLM
4. Demonstrates Q&A with automatic context retrieval

