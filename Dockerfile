# Multi-stage Docker build for gunn
FROM python:3.13-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.13-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r gunn && useradd -r -g gunn gunn

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/
COPY schemas/ ./schemas/
COPY README.md LICENSE ./

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R gunn:gunn /app

# Switch to non-root user
USER gunn

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV GUNN_ENVIRONMENT=production
ENV GUNN_LOG_FORMAT=json
ENV GUNN_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8000

# Default command
CMD ["python", "-m", "gunn", "web", "--host", "0.0.0.0", "--port", "8080"]
