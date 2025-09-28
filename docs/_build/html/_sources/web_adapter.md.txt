# Web Adapter

The Web adapter provides FastAPI-based REST and WebSocket endpoints for external system integration with the multi-agent simulation core.

## Features

- **REST API**: Submit intents and retrieve observations via HTTP endpoints
- **WebSocket Streaming**: Real-time observation streaming for responsive applications
- **Authentication & Authorization**: Token-based authentication with per-agent/world permissions
- **Rate Limiting**: Configurable rate limiting per tenant and agent
- **Error Handling**: Structured error responses with detailed error information

## Quick Start

### 1. Generate Authentication Token

```bash
uv run python -m gunn web generate-token --agent-id alice --world-id demo_world
```

This will output a token and configuration details:

```
Generated authentication token:
Token: bOQFLmqjEJyeyfoz7yyCqQf8Xu8W8xbHFgfOsRx6nu4
World ID: demo_world
Agent ID: alice
Permissions: submit_intent,get_observation,stream_observations
```

### 2. Start the Web Server

```bash
uv run python -m gunn web server \
  --host 0.0.0.0 \
  --port 8000 \
  --auth-token "bOQFLmqjEJyeyfoz7yyCqQf8Xu8W8xbHFgfOsRx6nu4:demo_world:alice:submit_intent,get_observation,stream_observations"
```

### 3. Submit an Intent

```bash
curl -X POST "http://localhost:8000/worlds/demo_world/agents/alice/intents" \
  -H "Authorization: Bearer bOQFLmqjEJyeyfoz7yyCqQf8Xu8W8xbHFgfOsRx6nu4" \
  -H "Content-Type: application/json" \
  -d '{
    "kind": "Speak",
    "payload": {"text": "Hello, world!"},
    "priority": 1,
    "context_seq": 0
  }'
```

### 4. Get Observations

```bash
curl "http://localhost:8000/worlds/demo_world/agents/alice/observations?timeout=5" \
  -H "Authorization: Bearer bOQFLmqjEJyeyfoz7yyCqQf8Xu8W8xbHFgfOsRx6nu4"
```

## API Reference

### REST Endpoints

#### Health Check
- **GET** `/health`
- Returns server health status

#### Submit Intent
- **POST** `/worlds/{world_id}/agents/{agent_id}/intents`
- Headers: `Authorization: Bearer <token>`
- Body: Intent JSON object
- Returns: Request ID and status

#### Get Observation
- **GET** `/worlds/{world_id}/agents/{agent_id}/observations`
- Headers: `Authorization: Bearer <token>`
- Query: `timeout` (optional, default 30s)
- Returns: Observation delta with patches

### WebSocket Endpoints

#### Observation Streaming
- **WebSocket** `/worlds/{world_id}/agents/{agent_id}/observations/stream?token=<token>`
- Streams real-time observation deltas as JSON messages

## Authentication

The Web adapter uses Bearer token authentication. Each token is associated with:

- **World ID**: Which simulation world the token can access
- **Agent ID**: Which agent the token represents
- **Permissions**: List of allowed operations
- **Expiration**: Optional expiration timestamp

### Permissions

- `submit_intent`: Submit intents for processing
- `get_observation`: Retrieve observations via REST
- `stream_observations`: Stream observations via WebSocket

## Configuration

### Command Line Options

```bash
uv run python -m gunn web server --help
```

### JSON Configuration File

```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "workers": 1,
  "log_level": "info",
  "rate_limit_requests": 100,
  "rate_limit_window": 60,
  "auth_tokens": [
    {
      "token": "your-token-here",
      "world_id": "demo_world",
      "agent_id": "alice",
      "permissions": ["submit_intent", "get_observation", "stream_observations"],
      "expires_at": null
    }
  ]
}
```

Use with: `uv run python -m gunn web server --config config.json`

## Rate Limiting

Rate limiting is applied per `world_id:agent_id` combination:

- **Default**: 100 requests per 60-second window
- **Configurable**: Via `--rate-limit-requests` and `--rate-limit-window`
- **Response**: HTTP 429 when limit exceeded

## Error Handling

The API returns structured error responses:

```json
{
  "error": "STALE_CONTEXT",
  "message": "Context is stale: expected_seq=5, actual_seq=10, threshold=0",
  "details": {
    "expected_seq": 5,
    "actual_seq": 10,
    "threshold": 0
  }
}
```

### Error Types

- `STALE_CONTEXT`: Intent context is outdated
- `QUOTA_EXCEEDED`: Agent quota limits exceeded
- `BACKPRESSURE`: System under load, request deferred
- `VALIDATION_ERROR`: Invalid request format
- `INTERNAL_ERROR`: Server error

## Security Considerations

1. **Token Management**: Store tokens securely, rotate regularly
2. **HTTPS**: Use HTTPS in production environments
3. **CORS**: Configure CORS appropriately for your use case
4. **Rate Limiting**: Adjust limits based on expected load
5. **Logging**: Monitor authentication failures and rate limit violations

## Integration Examples

### Python Client

```python
import asyncio
import httpx
import websockets
import json

async def submit_intent():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/worlds/demo_world/agents/alice/intents",
            headers={"Authorization": "Bearer your-token-here"},
            json={
                "kind": "Speak",
                "payload": {"text": "Hello!"},
                "priority": 1,
                "context_seq": 0
            }
        )
        return response.json()

async def stream_observations():
    uri = "ws://localhost:8000/worlds/demo_world/agents/alice/observations/stream?token=your-token-here"
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            observation = json.loads(message)
            print(f"Received observation: {observation}")

# Run examples
asyncio.run(submit_intent())
asyncio.run(stream_observations())
```

### JavaScript Client

```javascript
// Submit intent
async function submitIntent() {
  const response = await fetch('http://localhost:8000/worlds/demo_world/agents/alice/intents', {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer your-token-here',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      kind: 'Speak',
      payload: { text: 'Hello!' },
      priority: 1,
      context_seq: 0
    })
  });
  return await response.json();
}

// Stream observations
function streamObservations() {
  const ws = new WebSocket('ws://localhost:8000/worlds/demo_world/agents/alice/observations/stream?token=your-token-here');

  ws.onmessage = (event) => {
    const observation = JSON.parse(event.data);
    console.log('Received observation:', observation);
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
}
```

## Monitoring

The Web adapter integrates with the core telemetry system:

- **Structured Logs**: All requests logged with context
- **Metrics**: Request counts, latencies, error rates
- **Tracing**: OpenTelemetry integration (when configured)

Monitor these metrics for operational health:

- Request rate and latency
- Authentication failure rate
- Rate limiting violations
- WebSocket connection counts
- Error rates by type

## Troubleshooting

### Common Issues

1. **403 Forbidden**: Check token validity and permissions
2. **429 Too Many Requests**: Rate limit exceeded, implement backoff
3. **408 Request Timeout**: No observations available, check agent registration
4. **WebSocket Connection Failed**: Verify token and network connectivity

### Debug Mode

Enable debug logging:

```bash
uv run python -m gunn web server --log-level debug
```

This provides detailed request/response logging for troubleshooting.
