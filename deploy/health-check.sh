#!/bin/bash
set -e

# Health check script for gunn deployment
# Usage: ./health-check.sh [host] [port]

HOST="${1:-localhost}"
PORT="${2:-8080}"
TIMEOUT="${TIMEOUT:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"

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
}

check_endpoint() {
    local endpoint="$1"
    local description="$2"
    local expected_status="${3:-200}"

    log "Checking $description at $endpoint"

    local retry=0
    while [ $retry -lt $MAX_RETRIES ]; do
        if response=$(curl -s -w "%{http_code}" -o /tmp/health_response --max-time $TIMEOUT "$endpoint" 2>/dev/null); then
            status_code="${response: -3}"

            if [ "$status_code" = "$expected_status" ]; then
                log "✓ $description is healthy (HTTP $status_code)"
                if [ -f /tmp/health_response ]; then
                    cat /tmp/health_response | jq . 2>/dev/null || cat /tmp/health_response
                fi
                return 0
            else
                warn "✗ $description returned HTTP $status_code (expected $expected_status)"
                if [ -f /tmp/health_response ]; then
                    cat /tmp/health_response
                fi
            fi
        else
            warn "✗ Failed to connect to $description"
        fi

        retry=$((retry + 1))
        if [ $retry -lt $MAX_RETRIES ]; then
            log "Retrying in 2 seconds... ($retry/$MAX_RETRIES)"
            sleep 2
        fi
    done

    error "✗ $description failed after $MAX_RETRIES attempts"
    return 1
}

log "Starting health check for gunn deployment"
log "Target: $HOST:$PORT"

# Check main health endpoint
check_endpoint "http://$HOST:$PORT/health" "Health Check"

# Check readiness endpoint
check_endpoint "http://$HOST:$PORT/ready" "Readiness Check"

# Check liveness endpoint
check_endpoint "http://$HOST:$PORT/live" "Liveness Check"

# Check metrics endpoint (if enabled)
if check_endpoint "http://$HOST:8000/metrics" "Metrics" "200" 2>/dev/null; then
    log "✓ Metrics endpoint is available"
else
    warn "Metrics endpoint not available (this may be expected)"
fi

# Check API endpoints
log "Checking API endpoints..."

# Test basic API functionality
if curl -s -f "http://$HOST:$PORT/api/v1/status" >/dev/null 2>&1; then
    log "✓ API status endpoint is responding"
else
    warn "API status endpoint not available"
fi

log "Health check completed successfully!"

# Cleanup
rm -f /tmp/health_response
