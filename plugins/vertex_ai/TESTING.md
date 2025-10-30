# Testing Vertex AI Plugin Locally

## Prerequisites

### 1. Install Dependencies

Make sure you have the project dependencies installed:

```bash
# From project root
uv sync
```

### 2. Set Up GCP Credentials

You need Google Cloud credentials to use Vertex AI. Choose one of these methods:

#### Option A: Application Default Credentials (Recommended)

```bash
gcloud auth application-default login
```

This will use your default GCP project and credentials.

#### Option B: Service Account Key

1. Create a service account in GCP Console
2. Download the JSON key file
3. Set the environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

### 3. Set Environment Variables

Create a `.env` file in the project root or set environment variables:

```bash
export GCP_PROJECT="your-gcp-project-id"
export GCP_LOCATION="us-central1"  # Optional, defaults to us-central1
```

Or create a `.env` file:

```bash
# .env
GCP_PROJECT=your-gcp-project-id
GCP_LOCATION=us-central1
```

### 4. Enable Vertex AI API

Make sure the Vertex AI API is enabled in your GCP project:

```bash
gcloud services enable aiplatform.googleapis.com --project=your-gcp-project-id
```

## Running Tests

### Unit Tests (No API Calls)

These tests don't require GCP credentials:

```bash
# From project root
uv run pytest plugins/vertex_ai/tests/test_vertex_ai_llm.py::TestVertexAILLM::test_message -v
uv run pytest plugins/vertex_ai/tests/test_vertex_ai_llm.py::TestVertexAILLM::test_advanced_message -v
```

### Integration Tests (Requires GCP Credentials)

These tests make real API calls to Vertex AI:

```bash
# Run all integration tests for Vertex AI
uv run pytest plugins/vertex_ai/tests/test_vertex_ai_llm.py -m integration -v

# Run a specific integration test
uv run pytest plugins/vertex_ai/tests/test_vertex_ai_llm.py::TestVertexAILLM::test_simple -m integration -v

# Run all tests (unit + integration)
uv run pytest plugins/vertex_ai/tests/test_vertex_ai_llm.py -v
```

### Skip Integration Tests

To run only unit tests and skip integration tests:

```bash
uv run pytest plugins/vertex_ai/tests/test_vertex_ai_llm.py -m "not integration" -v
```

## Quick Test Script

You can also create a simple test script:

```python
# test_vertex_ai_simple.py
import asyncio
import os
from vision_agents.plugins import vertex_ai

async def main():
    # Get credentials from environment
    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")
    
    if not project:
        print("Error: GCP_PROJECT environment variable not set")
        return
    
    # Initialize LLM
    llm = vertex_ai.LLM(
        model="gemini-1.5-flash",  # Use flash for faster/cheaper testing
        project=project,
        location=location
    )
    
    # Simple test
    print("Testing Vertex AI LLM...")
    response = await llm.simple_response("Say hello in one sentence!")
    print(f"Response: {response.text}")
    
    # Stream test
    print("\nTesting streaming...")
    chunk_count = 0
    
    @llm.events.subscribe
    async def on_chunk(event):
        nonlocal chunk_count
        chunk_count += 1
        print(f"Chunk {chunk_count}: {event.delta[:50]}...", end="", flush=True)
    
    await llm.simple_response("Count to 5 in a creative way")
    await llm.events.wait()
    
    print(f"\n\nReceived {chunk_count} chunks")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it with:

```bash
uv run python test_vertex_ai_simple.py
```

## Troubleshooting

### "ImportError: google-cloud-aiplatform is required"

Install the dependency:

```bash
cd plugins/vertex_ai
uv sync
```

### "GCP_PROJECT environment variable not set"

Make sure you've set the `GCP_PROJECT` environment variable or added it to your `.env` file.

### "Permission denied" or authentication errors

1. Verify your GCP credentials:
   ```bash
   gcloud auth application-default print-access-token
   ```

2. Make sure your account has the "Vertex AI User" role or equivalent permissions.

3. Check that Vertex AI API is enabled:
   ```bash
   gcloud services list --enabled --project=your-project-id | grep aiplatform
   ```

### Tests are slow or timing out

Integration tests make real API calls and may take 10-30 seconds each. If you're experiencing timeouts:

1. Check your internet connection
2. Verify GCP API quotas haven't been exceeded
3. Try using a faster/lighter model like `gemini-1.5-flash` instead of `gemini-1.5-pro`

## Model Availability

Make sure the model you're using is available in your region:

- `gemini-1.5-flash` - Fast, cost-effective, good for testing
- `gemini-1.5-pro` - More capable, slower
- `gemini-2.0-flash-exp` - Experimental, check availability

Check [Vertex AI model documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/overview) for the latest available models.
