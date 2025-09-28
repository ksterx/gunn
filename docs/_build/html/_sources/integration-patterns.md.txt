# External Integration Patterns

This document describes common patterns for integrating external systems with the Gunn multi-agent simulation core.

## Overview

The Gunn simulation core provides multiple integration points through standardized APIs:

- **REST API**: HTTP endpoints for web applications and services
- **WebSocket**: Real-time streaming for low-latency interactions
- **gRPC**: High-performance binary protocol for game engines
- **Message-oriented**: Event-driven patterns for loose coupling

## Integration Patterns

### 1. Game Engine Integration (Unity/Unreal)

Game engines typically need bidirectional, low-latency communication with the simulation core.

#### Unity Integration Pattern

```csharp
// Unity C# client using gRPC
using Gunn.Simulation.V1;
using Grpc.Core;

public class UnitySimulationClient : MonoBehaviour
{
    private UnityAdapter.UnityAdapterClient client;
    private AsyncServerStreamingCall<ObservationDelta> observationStream;

    async void Start()
    {
        // Connect to simulation core
        var channel = new Channel("localhost:50051", ChannelCredentials.Insecure);
        client = new UnityAdapter.UnityAdapterClient(channel);

        // Start observation stream
        var request = new ObservationRequest
        {
            WorldId = "unity_world_001",
            AgentId = "player_agent"
        };

        observationStream = client.StreamObservations(request);

        // Process observations in background
        _ = ProcessObservations();
    }

    private async Task ProcessObservations()
    {
        await foreach (var delta in observationStream.ResponseStream.ReadAllAsync())
        {
            // Apply JSON patches to local game state
            ApplyObservationDelta(delta);
        }
    }

    public async void SubmitPlayerAction(string action, Vector3 position)
    {
        var intent = new Intent
        {
            Kind = Intent.Types.Kind.KindMove,
            AgentId = "player_agent",
            ReqId = System.Guid.NewGuid().ToString(),
            ContextSeq = currentViewSeq,
            Priority = 1,
            SchemaVersion = "1.0.0"
        };

        // Set payload with Unity-specific data
        var movePayload = new MovePayload
        {
            TargetPosition = new Vector3 { X = position.x, Y = position.y, Z = position.z },
            Speed = 5.0f,
            Run = true
        };

        intent.Payload = Any.Pack(movePayload);

        try
        {
            var response = await client.SubmitIntentAsync(intent);
            Debug.Log($"Intent accepted: {response.Status}");
        }
        catch (RpcException ex)
        {
            Debug.LogError($"Intent failed: {ex.Status}");
        }
    }

    // Handle Unity physics events
    void OnCollisionEnter(Collision collision)
    {
        var collisionEvent = new CollisionEvent
        {
            Entity1 = gameObject.name,
            Entity2 = collision.gameObject.name,
            ImpactForce = collision.impulse.magnitude,
            Position = new Vector3
            {
                X = collision.contacts[0].point.x,
                Y = collision.contacts[0].point.y,
                Z = collision.contacts[0].point.z
            }
        };

        var effectDraft = new EffectDraft
        {
            Kind = "Collision",
            SourceId = "unity_adapter",
            SchemaVersion = "1.0.0",
            Payload = Any.Pack(collisionEvent)
        };

        _ = client.EmitEffectAsync(effectDraft);
    }
}
```

#### Key Unity Integration Points

1. **Time Synchronization**: Unity sends TimeTick events to maintain simulation time
2. **Physics Events**: Collisions, triggers, and other physics events become Effects
3. **Agent Actions**: Player inputs and AI decisions become Intents
4. **Visual Updates**: ObservationDeltas drive visual state changes
5. **Spatial Queries**: Unity's spatial systems inform observation policies

### 2. Web Application Integration

Web applications typically use REST APIs with WebSocket streaming for real-time updates.

#### JavaScript/TypeScript Web Client

```typescript
// Web client using REST API and WebSocket
class SimulationClient {
    private baseUrl: string;
    private authToken: string;
    private websocket: WebSocket | null = null;

    constructor(baseUrl: string, authToken: string) {
        this.baseUrl = baseUrl;
        this.authToken = authToken;
    }

    // Get current observation via REST
    async getObservation(worldId: string, agentId: string): Promise<View | ObservationDelta> {
        const response = await fetch(
            `${this.baseUrl}/worlds/${worldId}/agents/${agentId}/observe`,
            {
                headers: {
                    'Authorization': `Bearer ${this.authToken}`,
                    'Content-Type': 'application/json'
                }
            }
        );

        if (!response.ok) {
            const error = await response.json();
            throw new SimulationError(error.error);
        }

        return response.json();
    }

    // Submit intent via REST
    async submitIntent(worldId: string, agentId: string, intent: IntentRequest): Promise<IntentResponse> {
        const response = await fetch(
            `${this.baseUrl}/worlds/${worldId}/agents/${agentId}/intents`,
            {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.authToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(intent)
            }
        );

        if (!response.ok) {
            const error = await response.json();
            throw new SimulationError(error.error);
        }

        return response.json();
    }

    // Stream observations via WebSocket
    streamObservations(worldId: string, agentId: string, onDelta: (delta: ObservationDelta) => void): void {
        const wsUrl = `${this.baseUrl.replace('http', 'ws')}/worlds/${worldId}/agents/${agentId}/stream`;

        this.websocket = new WebSocket(wsUrl, [], {
            headers: {
                'Authorization': `Bearer ${this.authToken}`
            }
        });

        this.websocket.onmessage = (event) => {
            const delta: ObservationDelta = JSON.parse(event.data);
            onDelta(delta);
        };

        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.websocket.onclose = (event) => {
            if (event.code !== 1000) {
                // Reconnect on unexpected close
                setTimeout(() => this.streamObservations(worldId, agentId, onDelta), 1000);
            }
        };
    }

    // Apply JSON Patch to local state
    applyObservationDelta(currentView: any, delta: ObservationDelta): any {
        const jsonpatch = require('jsonpatch');
        return jsonpatch.apply_patch(currentView, delta.patches);
    }
}

// Error handling
class SimulationError extends Error {
    constructor(public errorInfo: ErrorResponse) {
        super(errorInfo.message);
        this.name = 'SimulationError';
    }

    get recoveryAction(): string {
        return this.errorInfo.recovery_action;
    }

    get errorCode(): number {
        return this.errorInfo.code;
    }
}

// Usage example
const client = new SimulationClient('https://api.gunn.example.com/v1', 'your-jwt-token');

// Get initial observation
const view = await client.getObservation('world_001', 'agent_001');

// Submit speaking intent
await client.submitIntent('world_001', 'agent_001', {
    kind: 'Speak',
    payload: { text: 'Hello, world!', target_agent: 'agent_002' },
    context_seq: view.view_seq,
    req_id: crypto.randomUUID(),
    priority: 1,
    schema_version: '1.0.0'
});

// Stream real-time updates
client.streamObservations('world_001', 'agent_001', (delta) => {
    const updatedView = client.applyObservationDelta(currentView, delta);
    updateUI(updatedView);
});
```

#### Key Web Integration Points

1. **Authentication**: JWT tokens or OAuth for API access
2. **Real-time Updates**: WebSocket streaming for low-latency observation deltas
3. **Error Handling**: Structured error responses with recovery actions
4. **State Management**: JSON Patch application for efficient UI updates
5. **Offline Support**: Local state caching and sync on reconnection

### 3. LLM Service Integration

LLM services need streaming capabilities with cancellation support for intelligent interruption.

#### Python LLM Adapter

```python
import asyncio
from typing import AsyncIterator, Optional
import openai
from gunn.schemas.types import CancelToken, Intent, Effect

class LLMAdapter:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate_response(
        self,
        prompt: str,
        context: dict,
        cancel_token: CancelToken,
        max_tokens: int = 150
    ) -> AsyncIterator[str]:
        """Generate streaming response with cancellation support."""

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an AI agent in a simulation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                stream=True
            )

            async for chunk in stream:
                # Check for cancellation every token
                if cancel_token.cancelled:
                    await stream.aclose()
                    return

                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

                # Yield control every 20-30ms for responsive cancellation
                await asyncio.sleep(0.025)

        except Exception as e:
            if not cancel_token.cancelled:
                raise

    async def process_intent(
        self,
        intent: Intent,
        world_context: dict,
        cancel_token: CancelToken
    ) -> Effect | None:
        """Process an intent and generate appropriate response."""

        if intent["kind"] == "Speak":
            # Generate speaking response
            prompt = f"Respond to: {intent['payload'].get('text', '')}"

            response_text = ""
            async for token in self.generate_response(prompt, world_context, cancel_token):
                response_text += token

            if cancel_token.cancelled:
                return None  # Generation was cancelled

            # Create effect for the response
            effect: Effect = {
                "uuid": str(uuid.uuid4()),
                "kind": "SpeakResponse",
                "payload": {
                    "text": response_text,
                    "agent_id": intent["agent_id"],
                    "in_response_to": intent["req_id"]
                },
                "global_seq": 0,  # Will be filled by orchestrator
                "sim_time": 0.0,  # Will be filled by orchestrator
                "source_id": "llm_adapter",
                "schema_version": "1.0.0"
            }

            return effect

        return None

# Usage with orchestrator
async def handle_agent_intent(orchestrator, intent: Intent):
    # Issue cancel token for this generation
    cancel_token = orchestrator.issue_cancel_token(intent["agent_id"], intent["req_id"])

    # Start LLM generation
    llm_adapter = LLMAdapter(api_key="your-api-key")
    world_context = orchestrator.get_world_context(intent["agent_id"])

    # Generate response with cancellation support
    effect = await llm_adapter.process_intent(intent, world_context, cancel_token)

    if effect and not cancel_token.cancelled:
        # Submit effect to orchestrator
        await orchestrator.broadcast_event(effect)
    elif cancel_token.cancelled:
        print(f"Generation cancelled: {cancel_token.reason}")
```

#### Key LLM Integration Points

1. **Streaming Generation**: Token-by-token generation with regular cancellation checks
2. **Context Management**: World state and conversation history as context
3. **Cancellation Handling**: Immediate halt on cancel token trigger
4. **Error Recovery**: Graceful handling of API failures and timeouts
5. **Token Budgets**: Respect token limits and quotas

### 4. Multi-Tenant SaaS Integration

SaaS applications need tenant isolation, authentication, and rate limiting.

#### Multi-Tenant Architecture

```python
# Tenant-aware client wrapper
class TenantSimulationClient:
    def __init__(self, base_client, tenant_id: str, api_key: str):
        self.base_client = base_client
        self.tenant_id = tenant_id
        self.api_key = api_key

    async def create_world(self, world_config: dict) -> str:
        """Create a new simulation world for this tenant."""
        world_id = f"{self.tenant_id}_{uuid.uuid4().hex[:8]}"

        # Initialize world with tenant-specific configuration
        await self.base_client.initialize_world(
            world_id=world_id,
            config=world_config,
            tenant_id=self.tenant_id
        )

        return world_id

    async def register_agent(self, world_id: str, agent_config: dict) -> str:
        """Register an agent in a tenant's world."""
        # Verify world belongs to this tenant
        if not world_id.startswith(self.tenant_id):
            raise PermissionError("Cannot access world from different tenant")

        agent_id = f"{self.tenant_id}_{agent_config['name']}_{uuid.uuid4().hex[:8]}"

        # Apply tenant-specific observation policies
        observation_policy = self._create_tenant_policy(agent_config)

        await self.base_client.register_agent(
            world_id=world_id,
            agent_id=agent_id,
            policy=observation_policy
        )

        return agent_id

    def _create_tenant_policy(self, agent_config: dict):
        """Create observation policy with tenant-specific constraints."""
        # Apply tenant limits (e.g., observation range, relationship depth)
        max_distance = min(agent_config.get('observation_range', 100),
                          self._get_tenant_limit('max_observation_range'))

        return ObservationPolicy(
            distance_limit=max_distance,
            relationship_filter=agent_config.get('relationships', []),
            tenant_id=self.tenant_id
        )

# Rate limiting and quotas
class TenantRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client

    async def check_rate_limit(self, tenant_id: str, operation: str) -> bool:
        """Check if tenant is within rate limits for operation."""
        key = f"rate_limit:{tenant_id}:{operation}"
        current = await self.redis.get(key)

        limits = {
            'intents_per_minute': 1000,
            'observations_per_minute': 5000,
            'worlds_per_day': 10
        }

        limit = limits.get(operation, 100)

        if current and int(current) >= limit:
            return False

        # Increment counter with expiration
        await self.redis.incr(key)
        await self.redis.expire(key, 60)  # 1 minute window

        return True
```

#### Key Multi-Tenant Integration Points

1. **Tenant Isolation**: World and agent IDs prefixed with tenant identifier
2. **Resource Quotas**: Per-tenant limits on worlds, agents, and operations
3. **Rate Limiting**: Per-tenant rate limits with Redis-based tracking
4. **Data Segregation**: Tenant-specific observation policies and access controls
5. **Billing Integration**: Usage tracking for metered billing

## Error Handling Patterns

### Structured Error Responses

All integrations should handle errors consistently using the structured format:

```typescript
interface ErrorResponse {
    error: {
        code: number;           // From docs/errors.md
        name: string;           // Error constant name
        message: string;        // Human-readable message
        details: any;           // Structured error details
        recovery_action: string; // Suggested recovery action
        timestamp: string;      // ISO 8601 timestamp
        trace_id: string;       // For distributed tracing
    };
}

// Error handling with recovery actions
async function handleSimulationError(error: ErrorResponse) {
    switch (error.error.recovery_action) {
        case 'RETRY':
            await delay(1000);
            return retryOperation();

        case 'RETRY_WITH_DELAY':
            const backoff = Math.min(1000 * Math.pow(2, retryCount), 30000);
            await delay(backoff);
            return retryOperation();

        case 'REGENERATE':
            // Cancel current generation and start fresh
            await cancelCurrentGeneration();
            return startNewGeneration();

        case 'MODIFY_INTENT':
            // Prompt user to modify their intent
            return promptForIntentModification(error.error.details);

        case 'DEFER':
            // Queue for later processing
            return queueForLater();

        case 'ABORT':
            // Permanent failure, notify user
            throw new Error(error.error.message);
    }
}
```

### Circuit Breaker Pattern

For resilient integrations, implement circuit breaker patterns:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise

# Usage
circuit_breaker = CircuitBreaker()

async def submit_intent_with_circuit_breaker(intent):
    return await circuit_breaker.call(simulation_client.submit_intent, intent)
```

## Performance Considerations

### Connection Pooling

For high-throughput integrations, use connection pooling:

```python
# gRPC connection pooling
class GrpcConnectionPool:
    def __init__(self, target: str, pool_size: int = 10):
        self.target = target
        self.pool = asyncio.Queue(maxsize=pool_size)

        # Pre-populate pool
        for _ in range(pool_size):
            channel = grpc.aio.insecure_channel(target)
            client = UnityAdapterStub(channel)
            self.pool.put_nowait(client)

    async def get_client(self):
        return await self.pool.get()

    async def return_client(self, client):
        await self.pool.put(client)

    @asynccontextmanager
    async def client(self):
        client = await self.get_client()
        try:
            yield client
        finally:
            await self.return_client(client)

# Usage
pool = GrpcConnectionPool("localhost:50051")

async with pool.client() as client:
    response = await client.SubmitIntent(intent)
```

### Batch Operations

For efficiency, batch multiple operations:

```python
async def batch_submit_intents(intents: List[Intent]) -> List[IntentResponse]:
    """Submit multiple intents in a single batch."""
    batch_request = BatchIntentRequest(intents=intents)
    response = await client.BatchSubmitIntents(batch_request)
    return response.responses

# Observation delta batching
class ObservationBatcher:
    def __init__(self, batch_size: int = 10, flush_interval: float = 0.1):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.pending_deltas = []
        self.last_flush = time.time()

    async def add_delta(self, delta: ObservationDelta):
        self.pending_deltas.append(delta)

        if (len(self.pending_deltas) >= self.batch_size or
            time.time() - self.last_flush > self.flush_interval):
            await self.flush()

    async def flush(self):
        if self.pending_deltas:
            await self.send_batch(self.pending_deltas)
            self.pending_deltas.clear()
            self.last_flush = time.time()
```

## Security Best Practices

### Authentication and Authorization

```python
# JWT token validation
import jwt
from functools import wraps

def require_auth(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            request.user = payload
        except jwt.InvalidTokenError:
            return {'error': {'code': 5001, 'name': 'UNAUTHORIZED'}}, 401

        return await f(*args, **kwargs)
    return decorated_function

# mTLS certificate validation
def validate_client_certificate(cert):
    """Validate client certificate for adapter authentication."""
    # Verify certificate chain
    # Check certificate subject matches expected adapter
    # Validate certificate hasn't expired
    pass
```

### Input Validation

```python
from pydantic import BaseModel, validator

class IntentRequest(BaseModel):
    kind: str
    payload: dict
    context_seq: int
    req_id: str
    priority: int
    schema_version: str

    @validator('kind')
    def validate_kind(cls, v):
        allowed_kinds = ['Speak', 'Move', 'Interact', 'Custom']
        if v not in allowed_kinds:
            raise ValueError(f'Invalid intent kind: {v}')
        return v

    @validator('req_id')
    def validate_req_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Invalid req_id format')
        return v

    @validator('priority')
    def validate_priority(cls, v):
        if not -100 <= v <= 100:
            raise ValueError('Priority must be between -100 and 100')
        return v
```

This comprehensive integration guide provides patterns for the most common external system integrations with the Gunn simulation core, emphasizing proper error handling, performance optimization, and security best practices.
