# Vision Agents Observability Stack

This directory contains the complete observability setup for Vision Agents, including:
- **Prometheus** for metrics collection
- **Jaeger** for distributed tracing
- **Grafana** for visualization with pre-configured dashboards

## Quick Start

### 1. Start the Observability Stack

From the root of the Vision Agents repository:

```bash
docker-compose up -d
```

This will start:
- **Jaeger UI**: http://localhost:16686
- **Prometheus UI**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 2. Run Your Vision Agents Application

The example in `examples/01_simple_agent_example/simple_agent_example.py` already includes the `setup_telemetry()` function that:
- Exports traces to Jaeger (OTLP on port 4317)
- Exposes Prometheus metrics on port 9464

Run the example:

```bash
cd examples/01_simple_agent_example
uv run python simple_agent_example.py
```

### 3. View Metrics in Grafana

1. Open Grafana: http://localhost:3000
2. Login with `admin` / `admin`
3. Navigate to **Dashboards** → **Vision Agents - Performance Metrics**

The dashboard automatically displays:
- **STT Latency** (p50, p95, p99) by implementation
- **STT Errors** rate by provider and error type
- **TTS Latency** (p50, p95, p99) by implementation
- **TTS Errors** rate by provider and error type
- **Turn Detection Latency** (p50, p95, p99) by implementation

### 4. View Traces in Jaeger

1. Open Jaeger: http://localhost:16686
2. Select service: `agents`
3. Click **Find Traces** to see distributed traces

## Architecture

### Metrics Flow

```
Vision Agents App (port 9464)
    ↓ (scrape every 5s)
Prometheus (port 9090)
    ↓ (datasource)
Grafana (port 3000)
```

### Traces Flow

```
Vision Agents App
    ↓ (OTLP gRPC on port 4317)
Jaeger Collector
    ↓
Jaeger UI (port 16686)
```

## Available Metrics

### STT Metrics
- `stt_latency_ms` - Histogram of STT processing latency
  - Labels: `stt_class`, `provider`, `sample_rate`, `channels`, `samples`, `duration_ms`
- `stt_errors` - Counter of STT errors
  - Labels: `provider`, `error_type`

### TTS Metrics
- `tts_latency_ms` - Histogram of TTS synthesis latency
  - Labels: `tts_class`
- `tts_errors` - Counter of TTS errors
  - Labels: `provider`, `error_type`

### Turn Detection Metrics
- `turn_detection_latency_ms` - Histogram of turn detection latency
  - Labels: `class`

## Configuration

### Prometheus

Edit `prometheus/prometheus.yml` to:
- Change scrape interval
- Add additional scrape targets
- Configure alerting rules

### Grafana

#### Add Custom Dashboards

1. Place JSON dashboard files in `grafana/dashboards/`
2. They will be automatically loaded on startup

#### Modify Datasources

Edit `grafana/provisioning/datasources/prometheus.yml`

### Jaeger

Jaeger is configured with default settings. To customize, modify the `jaeger` service in `docker-compose.yml`.

## Troubleshooting

### Prometheus Can't Scrape Metrics

**Issue**: Prometheus shows target as "down"

**Solution**: Ensure `host.docker.internal` resolves correctly:
- **Linux**: Add `--add-host=host.docker.internal:host-gateway` to the prometheus service in docker-compose.yml
- **Mac/Windows**: Should work by default

### No Data in Grafana

1. Check Prometheus is scraping: http://localhost:9090/targets
2. Verify metrics are exposed: http://localhost:9464/metrics
3. Ensure your Vision Agents app is running with telemetry enabled

### Jaeger Shows No Traces

1. Verify OTLP receiver is running: `docker logs vision-agents-jaeger`
2. Check your app's trace exporter configuration
3. Ensure `endpoint="localhost:4317"` in your app

## Stopping the Stack

```bash
docker-compose down
```

To remove all data (metrics, dashboards, etc.):

```bash
docker-compose down -v
```

## Production Considerations

This setup is designed for development. For production:

1. **Security**:
   - Change default Grafana password
   - Add authentication to Prometheus
   - Use TLS for all connections

2. **Persistence**:
   - Configure external volumes for data persistence
   - Set up regular backups

3. **Scalability**:
   - Use Prometheus remote write for long-term storage
   - Consider Jaeger production deployment with Elasticsearch/Cassandra
   - Deploy Grafana with a proper database backend

4. **Monitoring**:
   - Set up alerts in Prometheus/Grafana
   - Configure notification channels (Slack, PagerDuty, etc.)
