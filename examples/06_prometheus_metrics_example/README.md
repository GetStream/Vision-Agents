# Prometheus Metrics Example

Export real metrics from Stream Agents to Prometheus using OpenTelemetry, with optional Grafana visualization.

## Overview

This example demonstrates how to:

1. Configure OpenTelemetry with a Prometheus exporter
2. Scrape metrics from the `/metrics` endpoint during a live video call
3. Visualize metrics in Grafana with pre-built dashboards

## Quick Start

### Option 1: Metrics endpoint only

```bash
cd examples/06_prometheus_metrics_example
uv sync
uv run python prometheus_metrics_example.py run --call-type default --call-id test-metrics
```

Then open http://localhost:9464/metrics in your browser to see raw metrics as you talk to the agent.

### Option 2: Full observability stack (Prometheus + Grafana)

1. Start the observability stack:

```bash
cd examples/06_prometheus_metrics_example
docker compose up -d
```

2. Run the agent:

```bash
uv sync
uv run python prometheus_metrics_example.py run --call-type default --call-id test-metrics
```

3. Open Grafana at http://localhost:3000 (no login required - anonymous access enabled)

The pre-configured dashboard shows:

- LLM latency (p50, p95, p99)
- STT latency (p50, p95, p99)
- TTS latency (p50, p95, p99)
- Turn duration
- Turn trailing silence
- Error rates

4. Stop the stack when done:

```bash
docker compose down
```

## Architecture

```
┌─────────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Stream Agent      │────▶│   Prometheus    │────▶│    Grafana      │
│  (port 9464)        │     │   (port 9090)   │     │   (port 3000)   │
│                     │     │                 │     │                 │
│  Metrics endpoint:  │     │  Scrapes every  │     │  Pre-built      │
│  /metrics           │     │  5 seconds      │     │  dashboards     │
└─────────────────────┘     └─────────────────┘     └─────────────────┘
```

## Metrics Available

### LLM Metrics

- `llm_latency_ms_milliseconds` - Total response latency (histogram)
- `llm_time_to_first_token_ms_milliseconds` - Time to first token for streaming (histogram)
- `llm_tokens_input_total` - Input/prompt tokens consumed (counter)
- `llm_tokens_output_total` - Output/completion tokens generated (counter)
- `llm_errors_total` - LLM errors (counter)
- `llm_tool_calls_total` - Tool/function calls executed (counter)
- `llm_tool_latency_ms_milliseconds` - Tool execution latency (histogram)

### STT Metrics

- `stt_latency_ms_milliseconds` - STT processing latency (histogram)
- `stt_audio_duration_ms_milliseconds` - Duration of audio processed (histogram)
- `stt_errors_total` - STT errors (counter)

### TTS Metrics

- `tts_latency_ms_milliseconds` - TTS synthesis latency (histogram)
- `tts_audio_duration_ms_milliseconds` - Duration of synthesized audio (histogram)
- `tts_characters_total` - Characters synthesized (counter)
- `tts_errors_total` - TTS errors (counter)

### Turn Detection Metrics

- `turn_duration_ms_milliseconds` - Duration of detected turns (histogram)
- `turn_trailing_silence_ms_milliseconds` - Trailing silence duration (histogram)

### Realtime LLM Metrics

- `realtime_sessions_total` - Realtime sessions started (counter)
- `realtime_session_duration_ms_milliseconds` - Session duration (histogram)
- `realtime_audio_input_bytes_bytes` - Audio bytes sent (counter)
- `realtime_audio_output_bytes_bytes` - Audio bytes received (counter)
- `realtime_responses_total` - Responses received (counter)
- `realtime_errors_total` - Realtime errors (counter)

## Prometheus Configuration

Add this to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'stream-agents'
    static_configs:
      - targets: [ 'localhost:9464' ]
    scrape_interval: 15s
```

## Grafana Dashboard

The included dashboard (`observability/grafana/dashboards/stream-agents.json`) provides real-time visualization of all key metrics with:

- Time-series graphs for latency percentiles
- Error rate monitoring
- Auto-refresh every 5 seconds

### Example PromQL Queries

```promql
# Average LLM latency over time
rate(llm_latency_ms_milliseconds_sum[5m]) / rate(llm_latency_ms_milliseconds_count[5m])

# Token usage rate
rate(llm_tokens_input_total[5m]) + rate(llm_tokens_output_total[5m])

# Error rate
rate(llm_errors_total[5m])

# 95th percentile latency
histogram_quantile(0.95, sum(rate(llm_latency_ms_milliseconds_bucket[5m])) by (le))
```

## Code Structure

The key pattern is:

```python
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import start_http_server

from vision_agents.core import Agent

# Start Prometheus exporter on localhost:9464
start_http_server(9464)

# Configure OpenTelemetry to use Prometheus
reader = PrometheusMetricReader()
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)

# Setup and run the agent
agent = Agent(...)
```

## Files

```
06_prometheus_metrics_example/
├── prometheus_metrics_example.py   # Main example code
├── docker-compose.yml              # Prometheus + Grafana stack
├── observability/
│   ├── prometheus/
│   │   └── prometheus.yml          # Prometheus config
│   └── grafana/
│       ├── dashboards/
│       │   └── stream-agents.json  # Pre-built dashboard
│       ├── provisioning/
│       │   ├── dashboards/
│       │   │   └── default.yml     # Dashboard provisioning
│       │   └── datasources/
│       │       └── prometheus.yml  # Datasource config
│       └── init-home-dashboard.sh  # Sets home dashboard
└── README.md
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

## Troubleshooting

### Prometheus can't scrape metrics

- Make sure the agent is running before Prometheus starts, or restart Prometheus after starting the agent
- On macOS, `host.docker.internal` should work. On Linux, you may need to use `--network="host"` or configure the target differently

### Grafana shows no data

- Wait a few seconds for metrics to be scraped
- Check Prometheus targets at http://localhost:9090/targets
- Ensure the agent is actively processing (make a call and talk to it)
