#!/bin/bash
set -e

# Deployment script for gunn
# Usage: ./deploy.sh [environment] [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
ENVIRONMENT="${1:-production}"
BUILD_IMAGE="${BUILD_IMAGE:-true}"
PUSH_IMAGE="${PUSH_IMAGE:-false}"
REGISTRY="${REGISTRY:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
COMPOSE_FILE="compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 1 ]]; do
    case $2 in
        --no-build)
            BUILD_IMAGE=false
            shift
            ;;
        --push)
            PUSH_IMAGE=true
            shift
            ;;
        --registry=*)
            REGISTRY="${2#*=}"
            shift
            ;;
        --tag=*)
            IMAGE_TAG="${2#*=}"
            shift
            ;;
        *)
            warn "Unknown option: $2"
            shift
            ;;
    esac
done

# Set environment-specific compose file
case $ENVIRONMENT in
    development|dev)
        COMPOSE_FILE="compose.dev.yml"
        ;;
    staging)
        COMPOSE_FILE="compose.staging.yml"
        ;;
    production|prod)
        COMPOSE_FILE="compose.yml"
        ;;
    *)
        error "Unknown environment: $ENVIRONMENT. Use development, staging, or production."
        ;;
esac

log "Deploying gunn to $ENVIRONMENT environment"
log "Using compose file: $COMPOSE_FILE"

cd "$PROJECT_ROOT"

# Validate configuration
log "Validating configuration..."
if ! python -m gunn config validate; then
    error "Configuration validation failed"
fi

# Build image if requested
if [ "$BUILD_IMAGE" = "true" ]; then
    log "Building Docker image..."

    if [ -n "$REGISTRY" ]; then
        IMAGE_NAME="$REGISTRY/gunn:$IMAGE_TAG"
    else
        IMAGE_NAME="gunn:$IMAGE_TAG"
    fi

    docker build -t "$IMAGE_NAME" .

    if [ "$PUSH_IMAGE" = "true" ]; then
        log "Pushing image to registry..."
        docker push "$IMAGE_NAME"
    fi
fi

# Create necessary directories
log "Creating deployment directories..."
mkdir -p deploy/config deploy/logs deploy/data

# Generate environment-specific configuration
log "Generating configuration for $ENVIRONMENT..."
python -m gunn config init --output="deploy/config/gunn.$ENVIRONMENT.yaml" --force

# Deploy with docker compose
log "Deploying services..."
if [ -f "$COMPOSE_FILE" ]; then
    docker compose -f "$COMPOSE_FILE" down --remove-orphans
    docker compose -f "$COMPOSE_FILE" up -d

    # Wait for services to be healthy
    log "Waiting for services to be healthy..."
    sleep 10

    # Check health
    if docker compose -f "$COMPOSE_FILE" ps | grep -q "unhealthy"; then
        warn "Some services are unhealthy"
        docker compose -f "$COMPOSE_FILE" ps
    else
        log "All services are healthy"
    fi

    log "Deployment completed successfully!"
    log "Services:"
    docker compose -f "$COMPOSE_FILE" ps

else
    error "Compose file not found: $COMPOSE_FILE"
fi

# Show deployment information
log "Deployment Information:"
echo "  Environment: $ENVIRONMENT"
echo "  Compose file: $COMPOSE_FILE"
echo "  Web API: http://localhost:8080"
echo "  Metrics: http://localhost:8000/metrics"
echo "  Health check: http://localhost:8080/health"

if [ -f "compose.yml" ] && grep -q "prometheus" "compose.yml"; then
    echo "  Prometheus: http://localhost:9090"
fi

if [ -f "compose.yml" ] && grep -q "grafana" "compose.yml"; then
    echo "  Grafana: http://localhost:3000 (admin/admin)"
fi
