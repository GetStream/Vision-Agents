## Vertex AI Plugin

Google Vertex AI LLM plugin for Vision Agents. Provides integration with Google's Vertex AI platform for language model interactions.

### Installation

```bash
pip install vision-agents-plugins-vertex-ai
```

### Requirements

- **Python**: 3.10+
- **Dependencies**: `vision-agents`, `google-cloud-aiplatform`
- **Authentication**: Google Cloud credentials configured (Application Default Credentials or service account)

### Quick Start

```python
from vision_agents.plugins import vertex_ai

# Initialize with Vertex AI
llm = vertex_ai.LLM(
    model="gemini-1.5-pro",
    project="your-gcp-project",
    location="us-central1"
)

# Simple response
response = await llm.simple_response("Explain quantum computing in 1 paragraph")
print(response.text)
```

### Configuration

The plugin requires Google Cloud authentication. You can:

1. Use Application Default Credentials (ADC):
   ```bash
   gcloud auth application-default login
   ```

2. Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to a service account key file

3. Pass credentials explicitly:
   ```python
   from google.oauth2 import service_account
   credentials = service_account.Credentials.from_service_account_file("path/to/key.json")
   llm = vertex_ai.LLM(model="gemini-1.5-pro", project="project", location="us-central1", credentials=credentials)
   ```

### Environment Variables

- **`GOOGLE_APPLICATION_CREDENTIALS`**: Path to service account key file (optional if using ADC)
- **`GCP_PROJECT`**: Default GCP project ID
- **`GCP_LOCATION`**: Default GCP location/region

### Features

- Streaming responses
- Function calling support
- Conversation history management
- Instruction following

### Testing

See [TESTING.md](TESTING.md) for detailed instructions on testing locally, including:
- Setting up GCP credentials
- Running unit and integration tests
- Example scripts
- Troubleshooting

Quick test:

```bash
# Set up credentials
gcloud auth application-default login
export GCP_PROJECT="your-project-id"

# Run tests
uv run pytest plugins/vertex_ai/tests/test_vertex_ai_llm.py -v

# Or run example
uv run python plugins/vertex_ai/example/vertex_ai_example.py
```
