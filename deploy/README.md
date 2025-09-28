# Deployment Utilities

This directory contains deployment utilities, configuration files, and scripts for deploying gunn in various environments.

## Files Overview

### Scripts
- `deploy.sh` - Main deployment script for Docker/Docker Compose
- `health-check.sh` - Health check script for verifying deployment status

### Configuration Files
- `prometheus.yml` - Prometheus monitoring configuration
- `k8s/` - Kubernetes deployment manifests

### Docker Files (in project root)
- `Dockerfile` - Multi-stage Docker build for production
- `compose.yml` - Production deployment with monitoring
- `compose.dev.yml` - Development environment

## Quick Start

### Local Development
```bash
# Start development environment
docker compose -f compose.dev.yml up -d

# Check health
./deploy/health-check.sh
```

### Production Deployment
```bash
# Deploy to production
./deploy/deploy.sh production

# Check deployment status
./deploy/health-check.sh
```

### Kubernetes Deployment
```bash
# Deploy all manifests
kubectl apply -f deploy/k8s/

# Check status
kubectl get pods -n gunn
```

## Configuration Management

### Environment Variables
Set configuration via environment variables with `GUNN_` prefix:

```bash
export GUNN_ENVIRONMENT=production
export GUNN_FEATURES=latency,backpressure,telemetry,metrics
export GUNN_LOG_LEVEL=INFO
export GUNN_MAX_AGENTS=1000
```

### Configuration Files
Create environment-specific YAML configuration files:

```bash
# Generate default config
gunn config init --output=config/production.yaml

# Validate configuration
gunn config validate config/production.yaml

# Show current config
gunn config show --format=json
```

### Feature Flags
Control functionality via the `GUNN_FEATURES` environment variable:

Available features:
- `latency` - Latency simulation
- `backpressure` - Backpressure management
- `telemetry` - Telemetry collection
- `metrics` - Metrics export
- `logging` - Structured logging
- `pii` - PII redaction
- `memory` - Memory management
- `compaction` - Log compaction
- `caching` - View caching
- `auth` - Authentication
- `ratelimit` - Rate limiting

## Health Checks

The deployment includes comprehensive health checks:

### Endpoints
- `GET /health` - Comprehensive health check
- `GET /ready` - Readiness check
- `GET /live` - Liveness check

### Health Check Script
```bash
# Check local deployment
./deploy/health-check.sh

# Check remote deployment
./deploy/health-check.sh myserver.com 8080
```

## Monitoring

### Metrics
Prometheus metrics are exported on port 8000:
- Feature flag status
- Configuration information
- System performance metrics
- Application-specific metrics

### Logging
Structured logging with configurable format:
- JSON format for production
- Text format for development
- PII redaction for sensitive data

### Grafana Dashboard
Included in docker compose setup:
- URL: http://localhost:3000
- Username: admin
- Password: admin

## Security

### Production Checklist
- [ ] Disable debug mode
- [ ] Use INFO or WARNING log level
- [ ] Enable PII redaction
- [ ] Use HTTPS/TLS for external access
- [ ] Set resource limits
- [ ] Use non-root user in containers

### Network Security
- Web API: Port 8080 (internal/load balancer only)
- Metrics: Port 8000 (monitoring systems only)

## Troubleshooting

### Common Issues

1. **Service won't start**:
   ```bash
   gunn config validate
   docker compose logs gunn
   ```

2. **Health checks failing**:
   ```bash
   ./deploy/health-check.sh
   curl -v http://localhost:8080/health
   ```

3. **High memory usage**:
   ```bash
   export GUNN_FEATURES=memory,compaction
   export GUNN_MAX_LOG_ENTRIES=5000
   ```

### Debug Mode
```bash
export GUNN_DEBUG=true
export GUNN_LOG_LEVEL=DEBUG
gunn web --log-level=DEBUG
```

## Performance Tuning

### Resource Allocation
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

### High-Throughput Configuration
```bash
export GUNN_MAX_AGENTS=2000
export GUNN_MAX_QUEUE_DEPTH=200
export GUNN_QUOTA_INTENTS_PER_MINUTE=120
export GUNN_FEATURES=latency,backpressure,memory,compaction,caching
```

## Support

For deployment issues:
1. Check logs with debug level enabled
2. Validate configuration with `gunn config validate`
3. Run health checks with `./deploy/health-check.sh`
4. Check metrics at `/metrics` endpoint
5. Review the deployment documentation in `docs/deployment.md`
