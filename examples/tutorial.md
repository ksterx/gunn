# Gunn Multi-Agent Simulation Core - Complete Tutorial

This tutorial provides a comprehensive guide to using the gunn multi-agent simulation core, with practical examples and best practices.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Performance Optimization](#performance-optimization)
6. [Integration Patterns](#integration-patterns)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

```bash
# Install gunn from source
git clone <repository-url>
cd gunn
uv sync
```

### Quick Start

```python
import asyncio
from gunn import Orchestrator, OrchestratorConfig
from gunn.facades import RLFacade
from gunn.policies.observation import DefaultObservationPolicy, PolicyConfig

async def quick_start():
    # Create orchestrator
    config = OrchestratorConfig(max_agents=3)
    orchestrator = Orchestrator(config, world_id="tutorial")

    # Create facade
    facade = RLFacade(orchestrator=orchestrator)
    await facade.initialize()

    # Register agent
    policy_config = PolicyConfig(distance_limit=50.0)
    policy = DefaultObservationPolicy(policy_config)
    await facade.register_agent("agent_1", policy)

    # Submit intent
    intent = {
        "kind": "Move",
        "payload": {"x": 10, "y": 20},
        "context_seq": 0,
        "req_id": "move_001",
        "agent_id": "agent_1",
        "priority": 1,
        "schema_version": "1.0.0",
    }

    effect, observation = await facade.step("agent_1", intent)
    print(f"Effect: {effect}")
    print(f"Observation: {observation}")

    await facade.shutdown()

asyncio.run(quick_start())
```

## Core Concepts

### 1. Event-Driven Architecture

The gunn system is built around an event-driven architecture where all changes to the world state are represented as **Effects** that are appended to an immutable **Event Log**.

```python
# Effects are the fundamental unit of change
effect = {
    "uuid": "unique-id",
    "kind": "Move",
    "payload": {"from": [0, 0], "to": [10, 10]},
    "global_seq": 42,
    "sim_time": 123.456,
    "source_id": "agent_1",
    "schema_version": "1.0.0"
}
```

### 2. Partial Observation

Agents don't see the complete world state. Instead, they receive filtered **Views** based on their **ObservationPolicy**.

```python
from gunn.policies.observation import PolicyConfig, DefaultObservationPolicy

# Configure what agents can observe
policy_config = PolicyConfig(
    distance_limit=100.0,  # Only see entities within 100 units
    relationship_filter=["friend", "ally"],  # Only see friends/allies
    field_visibility={"health": True, "inventory": False},  # Field-level filtering
    max_patch_ops=50,  # Fallback to full snapshot if too many changes
)

policy = DefaultObservationPolicy(policy_config)
```

### 3. Two-Phase Intent Processing

Agent actions go through a two-phase process:
1. **Intent Submission**: Agent expresses desire to act
2. **Effect Creation**: System validates and creates the actual effect

```python
# Phase 1: Agent submits intent
intent = {
    "kind": "Speak",
    "payload": {"text": "Hello world!"},
    "context_seq": current_view_seq,
    "req_id": "speak_001",
    "agent_id": "agent_1",
    "priority": 1,
    "schema_version": "1.0.0",
}

# Phase 2: System processes intent and creates effect
req_id = await orchestrator.submit_intent(intent)
```

### 4. Intelligent Interruption

Long-running operations (like LLM generation) can be intelligently interrupted when context becomes stale.

```python
# Issue cancel token for generation
cancel_token = orchestrator.issue_cancel_token("agent_1", "req_001")

# Check for staleness and cancel if needed
was_cancelled = await orchestrator.cancel_if_stale("agent_1", "req_001", new_view_seq)

if cancel_token.cancelled:
    print(f"Generation cancelled: {cancel_token.reason}")
```

## Basic Usage

### Setting Up an Orchestrator

```python
from gunn import Orchestrator, OrchestratorConfig

# Configure the orchestrator
config = OrchestratorConfig(
    max_agents=10,
    staleness_threshold=0,  # Immediate staleness detection
    debounce_ms=100.0,  # Debounce interruptions
    deadline_ms=5000.0,  # 5 second deadline for intents
    token_budget=1000,  # Token budget for generation
    backpressure_policy="defer",  # How to handle overload
    default_priority=1,  # Default intent priority
    use_in_memory_dedup=True,  # Use in-memory dedup for testing
)

orchestrator = Orchestrator(config, world_id="my_simulation")
await orchestrator.initialize()
```

### Using the RL Facade

The RL facade provides a familiar reinforcement learning interface:

```python
from gunn.facades import RLFacade

facade = RLFacade(orchestrator=orchestrator)
await facade.initialize()

# Register agent
await facade.register_agent("agent_1", observation_policy)

# Observe environment
observation = await facade.observe("agent_1")

# Take action
intent = {...}  # Intent dictionary
effect, new_observation = await facade.step("agent_1", intent)
```

### Using the Message Facade

The message facade provides an event-driven interface:

```python
from gunn.facades import MessageFacade

facade = MessageFacade(orchestrator=orchestrator)
await facade.initialize()

# Register agent
await facade.register_agent("agent_1", observation_policy)

# Subscribe to messages
subscription = await facade.subscribe(
    "agent_1",
    message_types={"Speak", "Move"},
    handler=my_message_handler
)

# Emit events
await facade.emit(
    "Speak",
    {"text": "Hello everyone!"},
    "agent_1"
)

# Get messages
messages = await facade.get_messages("agent_1", timeout=1.0)
```

### Creating Observation Policies

```python
from gunn.policies.observation import DefaultObservationPolicy, ConversationObservationPolicy, PolicyConfig

# Spatial observation policy
spatial_config = PolicyConfig(
    distance_limit=50.0,
    include_spatial_index=True,
    max_patch_ops=30
)
spatial_policy = DefaultObservationPolicy(spatial_config)

# Conversation observation policy
conversation_policy = ConversationObservationPolicy(PolicyConfig())

# Custom latency model
from gunn.policies.observation import DistanceLatencyModel
latency_model = DistanceLatencyModel(base_latency=0.01, distance_factor=0.001)
spatial_policy.set_latency_model(latency_model)
```

## Advanced Features

### 1. Deterministic Replay

All events are logged with hash chain integrity, enabling deterministic replay:

```python
# Get event log entries
entries = orchestrator.event_log.get_entries_since(0)

# Validate log integrity
is_valid = orchestrator.event_log.validate_integrity()

# Replay from CLI
# python -m gunn replay --from 0 --to 100 --world-id my_simulation
```

### 2. Memory Management

The system includes automatic memory management and compaction:

```python
# Configure memory management
config = OrchestratorConfig(
    max_log_entries=10000,
    view_cache_size=1000,
    compaction_threshold=5000,
    snapshot_interval=1000,
    auto_compaction_enabled=True,
)

# Manual memory operations
await orchestrator.memory_manager.create_snapshot(global_seq, world_state, sim_time)
await orchestrator.memory_manager.compact_log(orchestrator.event_log)
```

### 3. Backpressure Management

Handle system overload gracefully:

```python
# Set per-agent backpressure policy
orchestrator.set_agent_backpressure_policy("agent_1", "shed_oldest")

# Available policies: "defer", "shed_oldest", "drop_newest"

# Monitor backpressure events in logs
# Look for "backpressure_event" entries
```

### 4. Performance Monitoring

Built-in telemetry and metrics:

```python
from gunn.utils.telemetry import setup_logging, start_metrics_server

# Set up structured logging
setup_logging("INFO")

# Start Prometheus metrics server
start_metrics_server(port=8000)

# Metrics available at http://localhost:8000/metrics
```

### 5. Error Handling

Comprehensive error handling with recovery strategies:

```python
from gunn.utils.errors import StaleContextError, QuotaExceededError, BackpressureError

try:
    await orchestrator.submit_intent(intent)
except StaleContextError as e:
    print(f"Context is stale: expected {e.expected_seq}, got {e.actual_seq}")
    # Regenerate with updated context
except QuotaExceededError as e:
    print(f"Quota exceeded for {e.agent_id}: {e.quota_type}")
    # Wait or reduce load
except BackpressureError as e:
    print(f"Backpressure triggered: {e.policy_name}")
    # Apply backpressure policy
```

## Performance Optimization

### 1. SLO Requirements

The system is designed to meet specific Service Level Objectives:

- **Observation Delivery**: ≤20ms median latency
- **Cancellation Response**: ≤100ms cancel-to-halt
- **Intent Throughput**: ≥100 intents/sec per agent
- **Non-blocking Operations**: Per-agent isolation

### 2. Optimization Strategies

```python
# Optimize observation policies
policy_config = PolicyConfig(
    distance_limit=50.0,  # Smaller = faster
    max_patch_ops=20,  # Smaller = more full snapshots but faster patches
    include_spatial_index=True,  # Enable spatial optimizations
)

# Optimize orchestrator settings
config = OrchestratorConfig(
    max_queue_depth=100,  # Balance memory vs throughput
    staleness_threshold=1,  # Higher = less interruptions
    debounce_ms=50.0,  # Lower = more responsive
    use_in_memory_dedup=True,  # Faster for testing
)

# Use appropriate facade
# RLFacade: Better for step-by-step control
# MessageFacade: Better for event-driven scenarios
```

### 3. Memory Optimization

```python
# Configure memory limits
config = OrchestratorConfig(
    max_log_entries=5000,  # Trigger compaction sooner
    view_cache_size=500,  # Smaller cache
    compaction_threshold=2500,  # Compact more frequently
    snapshot_interval=500,  # More frequent snapshots
)

# Manual memory management
import gc
gc.collect()  # Force garbage collection

# Monitor memory usage
memory_stats = orchestrator.memory_manager.get_memory_stats()
```

## Integration Patterns

### 1. Web API Integration

```python
# See examples/web_adapter_demo.py for complete example
from gunn.adapters.web import WebAdapter

web_adapter = WebAdapter(orchestrator)
await web_adapter.start_server(host="0.0.0.0", port=8080)

# REST endpoints available:
# POST /agents/{agent_id}/intents
# GET /agents/{agent_id}/observations
# WebSocket /agents/{agent_id}/stream
```

### 2. LLM Integration

```python
# See examples/abc_conversation_demo.py for complete example
from gunn.adapters.llm import LLMAdapter

llm_adapter = LLMAdapter(orchestrator)

# Stream generation with cancellation
async for token in llm_adapter.generate_stream(prompt, cancel_token):
    if cancel_token.cancelled:
        break
    yield token
```

### 3. Unity Integration

```python
# See examples/unity_integration_demo.py for complete example
from gunn.adapters.unity import UnityAdapter

unity_adapter = UnityAdapter(orchestrator)
await unity_adapter.connect()

# Convert intents to Unity commands
await unity_adapter.handle_intent(move_intent)

# Convert Unity events to effects
await unity_adapter.handle_unity_event(collision_event)
```

## Troubleshooting

### Common Issues

1. **High Latency**
   ```python
   # Check observation policy settings
   policy_config.distance_limit = 30.0  # Reduce observation range
   policy_config.max_patch_ops = 10  # Reduce patch complexity

   # Check system load
   # Monitor CPU/memory usage
   # Reduce agent count or operation frequency
   ```

2. **Memory Growth**
   ```python
   # Enable auto-compaction
   config.auto_compaction_enabled = True
   config.compaction_threshold = 1000  # Compact more frequently

   # Check for memory leaks
   memory_stats = orchestrator.memory_manager.get_memory_stats()
   ```

3. **Intent Failures**
   ```python
   # Check staleness threshold
   config.staleness_threshold = 2  # Allow slightly stale context

   # Check quota limits
   config.quota_intents_per_minute = 120  # Increase quota

   # Check backpressure settings
   config.max_queue_depth = 200  # Increase queue capacity
   ```

4. **Cancellation Issues**
   ```python
   # Ensure token yielding in generation
   async def generate_with_cancellation(cancel_token):
       for i in range(num_tokens):
           if cancel_token.cancelled:
               break
           await asyncio.sleep(0.025)  # Yield every 25ms
           # Generate token...
   ```

### Debugging Tools

1. **Event Log Analysis**
   ```python
   # Check event sequence
   entries = orchestrator.event_log.get_entries_since(0)
   for entry in entries[-10:]:  # Last 10 events
       print(f"Seq {entry.global_seq}: {entry.effect['kind']}")

   # Validate integrity
   is_valid = orchestrator.event_log.validate_integrity()
   ```

2. **Performance Monitoring**
   ```python
   # Run performance benchmarks
   from examples.performance_benchmark import PerformanceBenchmark

   benchmark = PerformanceBenchmark(max_agents=5)
   await benchmark.setup()
   results = await benchmark.run_all_benchmarks()
   report = benchmark.generate_report()
   print(report)
   ```

3. **Logging Configuration**
   ```python
   from gunn.utils.telemetry import setup_logging

   # Enable debug logging
   setup_logging("DEBUG")

   # Look for specific log patterns:
   # - "intent_submitted" / "intent_processed"
   # - "observation_generated" / "observation_delivered"
   # - "cancellation_triggered" / "generation_cancelled"
   # - "backpressure_event" / "quota_exceeded"
   ```

### Performance Tuning

1. **Latency Optimization**
   ```python
   # Reduce observation complexity
   policy_config.distance_limit = 25.0
   policy_config.max_patch_ops = 15

   # Optimize debounce settings
   config.debounce_ms = 25.0  # Faster response

   # Use in-memory storage for testing
   config.use_in_memory_dedup = True
   ```

2. **Throughput Optimization**
   ```python
   # Increase queue capacities
   config.max_queue_depth = 500
   config.quota_intents_per_minute = 300

   # Optimize memory settings
   config.compaction_threshold = 10000  # Less frequent compaction
   config.view_cache_size = 2000  # Larger cache
   ```

3. **Memory Optimization**
   ```python
   # Aggressive compaction
   config.compaction_threshold = 1000
   config.snapshot_interval = 200
   config.max_snapshots = 5

   # Smaller caches
   config.view_cache_size = 100
   config.max_log_entries = 2000
   ```

## Best Practices

1. **Agent Design**
   - Keep intent payloads small
   - Use appropriate priorities (0=high, 1=normal, 2=low)
   - Handle cancellation gracefully
   - Implement proper error handling

2. **Observation Policies**
   - Use distance limits appropriate for your scenario
   - Configure field visibility to reduce data transfer
   - Set reasonable max_patch_ops limits
   - Consider using specialized policies (ConversationObservationPolicy)

3. **Performance**
   - Monitor SLO compliance regularly
   - Use performance benchmarks to validate changes
   - Configure memory management appropriately
   - Enable structured logging for debugging

4. **Integration**
   - Use appropriate facades for your use case
   - Implement proper error handling and retries
   - Monitor system health and performance
   - Test with realistic load patterns

This tutorial covers the essential aspects of using the gunn multi-agent simulation core. For more detailed examples, see the demo files in the `examples/` directory.
