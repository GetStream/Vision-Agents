# Prometheus Metrics Example

Export real metrics from Stream Agents to Prometheus using OpenTelemetry.

## Overview

This example demonstrates how to:
1. Configure OpenTelemetry with a Prometheus exporter
2. Attach `MetricsCollector` to an agent for opt-in metrics collection
3. Scrape metrics from the `/metrics` endpoint during a live video call

## Running the Example

```bash
cd examples/03_prometheus_metrics_example
uv sync
uv run python prometheus_metrics_example.py --call-type default --call-id test-metrics
```

Then open http://localhost:9464/metrics in your browser to see real-time metrics as you talk to the agent.

## Metrics Available

### LLM Metrics
- `llm_latency_ms` - Total response latency (histogram)
- `llm_time_to_first_token_ms` - Time to first token for streaming (histogram)
- `llm_tokens_input` - Input/prompt tokens consumed (counter)
- `llm_tokens_output` - Output/completion tokens generated (counter)
- `llm_errors` - LLM errors (counter)
- `llm_tool_calls` - Tool/function calls executed (counter)
- `llm_tool_latency_ms` - Tool execution latency (histogram)

### STT Metrics
- `stt_latency_ms` - STT processing latency (histogram)
- `stt_audio_duration_ms` - Duration of audio processed (histogram)
- `stt_errors` - STT errors (counter)

### TTS Metrics
- `tts_latency_ms` - TTS synthesis latency (histogram)
- `tts_audio_duration_ms` - Duration of synthesized audio (histogram)
- `tts_characters` - Characters synthesized (counter)
- `tts_errors` - TTS errors (counter)

### Turn Detection Metrics
- `turn_duration_ms` - Duration of detected turns (histogram)
- `turn_trailing_silence_ms` - Trailing silence duration (histogram)

### Realtime LLM Metrics
- `realtime_sessions` - Realtime sessions started (counter)
- `realtime_session_duration_ms` - Session duration (histogram)
- `realtime_audio_input_bytes` - Audio bytes sent (counter)
- `realtime_audio_output_bytes` - Audio bytes received (counter)
- `realtime_responses` - Responses received (counter)
- `realtime_errors` - Realtime errors (counter)

## Prometheus Configuration

Add this to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'stream-agents'
    static_configs:
      - targets: ['localhost:9464']
    scrape_interval: 15s
```

## Grafana Dashboard

Example PromQL queries:

```promql
# Average LLM latency over time
rate(llm_latency_ms_sum[5m]) / rate(llm_latency_ms_count[5m])

# Token usage rate
rate(llm_tokens_input[5m]) + rate(llm_tokens_output[5m])

# Error rate
rate(llm_errors[5m])
```

## Code Structure

The key pattern is:

```python
# 1. Configure OpenTelemetry BEFORE importing vision_agents
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from prometheus_client import start_http_server

start_http_server(9464)
reader = PrometheusMetricReader()
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)

# 2. Now import and create your agent
from vision_agents.core import Agent
from vision_agents.core.observability import MetricsCollector

agent = Agent(...)

# 3. Attach MetricsCollector to opt-in to metrics
collector = MetricsCollector(agent)
```

## Environment Variables

Set these in your `.env` file:

```
GOOGLE_API_KEY=your_key
DEEPGRAM_API_KEY=your_key
ELEVENLABS_API_KEY=your_key
STREAM_API_KEY=your_key
STREAM_API_SECRET=your_secret
```
