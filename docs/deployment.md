# Deployment Guide

This guide covers deploying gunn in various environments with proper configuration management, health checks, and monitoring.

## Quick Start

### Docker Compose (Recommended for Development)

```bash
# Development environment
docker compose -f compose.dev.yml up -d

# Production environment
docker compose up -d
```

### Using Deployment Script

```bash
# Deploy to production
./deploy/deploy.sh production

# Deploy to staging with custom registry
./deploy/deploy.sh staging --registry=myregistry.com --push

# Deploy without building image
./deploy/deploy.sh production --no-build
```

## Configuration Management

### Environment Variables

gunn supports configuration via environment variables with the `GUNN_` prefix:

```bash
# Core settings
export GUNN_ENVIRONMENT=production
export GUNN_DEBUG=false
export GUNN_LOG_LEVEL=INFO
export GUNN_LOG_FORMAT=json

# Feature flags (comma-separated)
export GUNN_FEATURES=latency,backpressure,telemetry,metrics,logging,pii,memory,compaction,caching,ratelimit

# Database
export GUNN_DATABASE_URL=sqlite:///data/gunn.db

# Orchestrator settings
export GUNN_MAX_AGENTS=1000
export GUNN_STALENESS_THRESHOLD=0
export GUNN_BACKPRESSURE_POLICY=defer
export GUNN_DEBOUNCE_MS=100.0
export GUNN_DEADLINE_MS=5000.0

# Metrics
export GUNN_METRICS_PORT=8000
export GUNN_METRICS_PATH=/metrics
```

### Configuration Files

Create environment-specific configuration files:

```bash
# Initialize default configuration
gunn config init --output=config/production.yaml

# Validate configuration
gunn config validate config/production.yaml

# Show current configuration
gunn config show --format=json
```

Example configuration file (`config/production.yaml`):

```yaml
environment: production
debug: false

features:
  latency_simulation: true
  backpressure_management: true
  telemetry: true
  metrics_export: true
  structured_logging: true
  pii_redaction: true
  memory_management: true
  log_compaction: true
  view_caching: true
  rate_limiting: true
  authentication: false
  authorization: false

logging:
  level: INFO
  format: json
  enable_pii_redaction: true
  log_file: /app/logs/gunn.log
  max_file_size_mb: 100
  backup_count: 5

metrics:
  enabled: true
  port: 8000
  path: /metrics
  export_interval_seconds: 15.0
  include_feature_flags: true

orchestrator:
  max_agents: 1000
  staleness_threshold: 0
  backpressure_policy: defer
  max_queue_depth: 100
  quota_intents_per_minute: 60
```

### Feature Flags

Control functionality via the `GUNN_FEATURES` environment variable:

```bash
# Enable specific features
export GUNN_FEATURES=latency,backpressure,telemetry,metrics

# Available features:
# - latency: Latency simulation
# - backpressure: Backpressure management
# - staleness: Staleness detection
# - cancellation: Cancellation tokens
# - telemetry: Telemetry collection
# - metrics: Metrics export
# - logging: Structured logging
# - pii: PII redaction
# - memory: Memory management
# - compaction: Log compaction
# - caching: View caching
# - auth: Authentication
# - authz: Authorization
# - ratelimit: Rate limiting
# - distributed: Distributed mode (experimental)
# - gpu: GPU acceleration (experimental)
```

## Docker Deployment

### Building Images

```bash
# Build production image
docker build -t gunn:latest .

# Build with custom tag
docker build -t myregistry.com/gunn:v1.0.0 .

# Push to registry
docker push myregistry.com/gunn:v1.0.0
```

### Docker Compose

The repository includes multiple compose files:

- `compose.yml`: Production deployment with monitoring
- `compose.dev.yml`: Development environment
- `compose.staging.yml`: Staging environment (if created)

```bash
# Production with monitoring stack
docker compose up -d

# Development with hot reload
docker compose -f compose.dev.yml up -d

# Scale the service
docker compose up -d --scale gunn=3
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.19+)
- kubectl configured
- Persistent storage class available

### Deploy to Kubernetes

```bash
# Create namespace and resources
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/pvc.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml

# Optional: Create ingress
kubectl apply -f deploy/k8s/ingress.yaml
```

### Customize for Your Environment

1. **Update ConfigMap** (`deploy/k8s/configmap.yaml`):
   - Modify configuration values for your environment
   - Update database URL, resource limits, etc.

2. **Update Ingress** (`deploy/k8s/ingress.yaml`):
   - Change hostname from `gunn.example.com`
   - Update TLS certificate configuration

3. **Update Deployment** (`deploy/k8s/deployment.yaml`):
   - Change image repository and tag
   - Adjust resource requests/limits
   - Modify environment variables

### Monitoring in Kubernetes

The deployment includes Prometheus annotations for automatic scraping:

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"
```

## Health Checks

gunn provides comprehensive health check endpoints:

### Endpoints

- `GET /health`: Comprehensive health check with detailed status
- `GET /ready`: Readiness check for load balancer
- `GET /live`: Liveness check for container orchestration

### Health Check Script

Use the provided health check script:

```bash
# Check local deployment
./deploy/health-check.sh

# Check remote deployment
./deploy/health-check.sh myserver.com 8080

# With custom timeout and retries
TIMEOUT=5 MAX_RETRIES=5 ./deploy/health-check.sh
```

### Health Check Configuration

Configure health checks via environment or config file:

```yaml
health:
  enabled: true
  timeout_seconds: 5.0
  check_interval_seconds: 30.0
  check_database: true
  check_event_log: true
  check_orchestrator: true
  check_memory_usage: true
  max_memory_usage_percent: 90.0
  max_response_time_ms: 1000.0
```

## Monitoring and Observability

### Metrics

gunn exports Prometheus metrics on port 8000:

```bash
# View metrics
curl http://localhost:8000/metrics

# Key metrics include:
# - gunn_feature_flag_enabled: Feature flag status
# - gunn_config_info: Configuration information
# - gunn_startup_time_seconds: Service startup time
# - gunn_config_reloads_total: Configuration reload count
```

### Logging

Structured logging with configurable format:

```bash
# JSON format (production)
export GUNN_LOG_FORMAT=json

# Text format (development)
export GUNN_LOG_FORMAT=text

# Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
export GUNN_LOG_LEVEL=INFO
```

### Grafana Dashboard

The docker compose setup includes Grafana with pre-configured dashboards:

- URL: http://localhost:3000
- Username: admin
- Password: admin

## Security Considerations

### Production Checklist

- [ ] Disable debug mode (`GUNN_DEBUG=false`)
- [ ] Use INFO or WARNING log level
- [ ] Enable PII redaction (`pii` feature flag)
- [ ] Use HTTPS/TLS for external access
- [ ] Configure proper authentication if needed
- [ ] Set resource limits in container orchestration
- [ ] Use non-root user in containers
- [ ] Enable security scanning in CI/CD

### Network Security

```bash
# Restrict network access
# - Web API: Port 8080 (internal/load balancer only)
# - Metrics: Port 8000 (monitoring systems only)
# - Health checks: Same as web API
```

### Secrets Management

```bash
# Use secrets for sensitive configuration
kubectl create secret generic gunn-secrets \
  --from-literal=database-url="postgresql://user:pass@host/db" \
  --from-literal=auth-token="secret-token"
```

## Troubleshooting

### Common Issues

1. **Service won't start**:
   ```bash
   # Check configuration
   gunn config validate

   # Check logs
   docker compose logs gunn
   ```

2. **Health checks failing**:
   ```bash
   # Run health check script
   ./deploy/health-check.sh

   # Check specific endpoint
   curl -v http://localhost:8080/health
   ```

3. **High memory usage**:
   ```bash
   # Enable memory management features
   export GUNN_FEATURES=memory,compaction

   # Adjust limits
   export GUNN_MAX_LOG_ENTRIES=5000
   export GUNN_VIEW_CACHE_SIZE=500
   ```

4. **Performance issues**:
   ```bash
   # Check metrics
   curl http://localhost:8000/metrics | grep gunn_

   # Enable performance features
   export GUNN_FEATURES=latency,backpressure,caching
   ```

### Debugging

```bash
# Enable debug logging
export GUNN_DEBUG=true
export GUNN_LOG_LEVEL=DEBUG

# Run with verbose output
gunn web --log-level=DEBUG

# Check configuration
gunn config show --format=json
```

### Support

For deployment issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review logs with debug level enabled
3. Validate configuration with `gunn config validate`
4. Run health checks with `./deploy/health-check.sh`
5. Check metrics at `/metrics` endpoint

## Performance Tuning

### Resource Allocation

```yaml
# Kubernetes resource limits
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

### Configuration Tuning

```bash
# High-throughput configuration
export GUNN_MAX_AGENTS=2000
export GUNN_MAX_QUEUE_DEPTH=200
export GUNN_QUOTA_INTENTS_PER_MINUTE=120

# Memory optimization
export GUNN_MAX_LOG_ENTRIES=5000
export GUNN_VIEW_CACHE_SIZE=500
export GUNN_COMPACTION_THRESHOLD=2500

# Performance features
export GUNN_FEATURES=latency,backpressure,memory,compaction,caching
```
