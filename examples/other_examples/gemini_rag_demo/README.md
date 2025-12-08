# Gemini RAG Demo

This example demonstrates **Retrieval-Augmented Generation (RAG)** using Google's native [File Search Tool](https://blog.google/technology/developers/file-search-gemini-api/) in the Gemini API.

## What is RAG?

RAG (Retrieval-Augmented Generation) enhances LLM responses by:
1. **Retrieving** relevant information from a knowledge base
2. **Augmenting** the query with this context
3. **Generating** more accurate, grounded responses

## Features

- **Fully Managed**: Google handles chunking, embeddings, and vector search
- **Multiple File Formats**: Supports PDF, TXT, MD, code files, JSON, and more
- **Built-in Citations**: Responses include source references
- **Automatic Context Injection**: Just attach the RAG provider to your LLM

## Setup

1. Set your Google API key:
   ```bash
   export GOOGLE_API_KEY=your-api-key
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Run the demo:
   ```bash
   uv run python gemini_rag_demo.py
   ```

## How It Works

```python
from vision_agents.plugins.gemini import GeminiFileSearchRAG, LLM

# 1. Create RAG provider
rag = GeminiFileSearchRAG()

# 2. Upload documents to the knowledge base
await rag.add_file("docs/manual.pdf")
await rag.add_file("docs/faq.md")

# 3. Attach RAG to LLM for automatic context injection
llm = LLM(model="gemini-2.0-flash")
llm.set_rag_provider(rag, top_k=5, include_citations=True)

# 4. Query - context is automatically retrieved and injected
response = await llm.simple_response("How do I reset my password?")
```

## Demo Output

The demo creates sample documents (company policy, product FAQ, technical specs) and demonstrates querying the knowledge base:

```
🔍 Gemini RAG Demo - File Search with Retrieval-Augmented Generation

📦 Initializing Gemini File Search RAG provider...

📄 Creating sample documents...

☁️  Uploading documents to Gemini File Search Store...
   ✅ Uploaded: company_policy.txt
   ✅ Uploaded: product_faq.txt
   ✅ Uploaded: technical_specs.txt

💬 Demo: Asking questions about the knowledge base

❓ Question: How many vacation days do employees get per year?
💡 Answer: Full-time employees are entitled to 25 days of paid vacation per year...
```

## Advanced Usage

### Direct File Search Tool Access

For more control, you can get the native Gemini Tool:

```python
# Get the file search tool for native integration
tool = rag.get_file_search_tool(top_k=10)

# Use directly with generate_content
response = await client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Find information about...",
    config=GenerateContentConfig(tools=[tool])
)
```

### Event Observability

Subscribe to RAG events for monitoring:

```python
from vision_agents.core.rag.events import (
    RAGFileAddedEvent,
    RAGRetrievalCompleteEvent,
)

@rag.events.subscribe
async def on_retrieval(event: RAGRetrievalCompleteEvent):
    print(f"Retrieved {event.result_count} results in {event.retrieval_time_ms}ms")
```

## Supported File Types

| Extension | MIME Type |
|-----------|-----------|
| `.txt` | text/plain |
| `.md` | text/markdown |
| `.pdf` | application/pdf |
| `.py` | text/x-python |
| `.js` | text/javascript |
| `.json` | application/json |
| `.html` | text/html |
| And many more... | |

## Pricing

According to Google's File Search documentation:
- **Storage**: Free
- **Embedding generation at query time**: Free
- **Initial indexing**: $0.15 per 1 million tokens

