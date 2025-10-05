# Priority Fairness and Quota Management

## Overview

This document describes the priority fairness and quota management system implemented in Task 30. The system provides:

1. **Token Bucket Rate Limiting**: Per-agent quota enforcement for intents and tokens
2. **Priority Aging**: Prevents starvation by increasing priority of waiting intents over time
3. **Weighted Round-Robin Scheduling**: Fair scheduling across agents with aging-adjusted priorities
4. **Configurable Policies**: Flexible configuration for different use cases

## Components

### 1. Token Bucket (`src/gunn/utils/quota.py`)

The `TokenBucket` class implements the classic token bucket algorithm for rate limiting:

```python
bucket = TokenBucket(
    capacity=10.0,        # Maximum burst size
    refill_rate=1.0,      # Tokens per refill interval
    refill_interval=1.0,  # Seconds between refills
)

# Try to consume tokens
if await bucket.consume(5.0):
    # Tokens available
    pass

# Wait for tokens to become available
if await bucket.wait_for_tokens(5.0, timeout=2.0):
    # Tokens acquired
    pass
```

**Features:**
- Automatic token refill based on elapsed time
- Burst handling with configurable capacity
- Async-safe with proper locking
- Timeout support for waiting

### 2. Quota Controller (`src/gunn/utils/quota.py`)

The `QuotaController` manages per-agent quotas using token buckets:

```python
policy = QuotaPolicy(
    intents_per_minute=60,
    tokens_per_minute=10000,
    burst_size=10,
    refill_interval_seconds=1.0,
)

controller = QuotaController(policy)

# Check intent quota
await controller.check_intent_quota("agent1")

# Check token quota
await controller.check_token_quota("agent1", tokens=100)

# Get available quota
available = await controller.get_available_intents("agent1")
```

**Features:**
- Separate quotas for intents and tokens
- Per-agent isolation
- Configurable burst sizes
- Wait or fail-fast modes
- Statistics tracking
- Agent removal and cleanup

**Error Handling:**
- Raises `QuotaExceededError` when quota is exhausted
- Includes agent_id, quota_type, and limit in error

### 3. Priority Aging (`src/gunn/utils/priority_aging.py`)

The `PriorityAging` system prevents starvation by increasing priority over time:

```python
policy = AgingPolicy(
    enabled=True,
    aging_rate=0.1,              # Priority increase per second
    max_priority_boost=5,         # Maximum boost
    min_wait_time_seconds=1.0,   # Start aging after 1 second
)

aging = PriorityAging(policy)

# Track intent for aging
aging.track_intent(intent, priority=5)

# Get aged priority (increases over time)
aged_priority = aging.get_aged_priority(req_id)

# Untrack when processed
aging.untrack_intent(req_id)
```

**Aging Formula:**
```
aged_priority = original_priority + min(
    (wait_time - min_wait_time) * aging_rate,
    max_priority_boost
)
```

**Features:**
- Configurable aging rate and maximum boost
- Minimum wait time before aging starts
- Can be disabled for strict priority ordering
- Statistics tracking (total aged, max boost applied, wait times)
- Per-intent tracking with automatic cleanup

### 4. Enhanced Weighted Round-Robin Scheduler (`src/gunn/utils/scheduling.py`)

The scheduler now integrates priority aging for fair processing:

```python
aging_policy = AgingPolicy(
    aging_rate=1.0,
    max_priority_boost=5,
)

scheduler = WeightedRoundRobinScheduler(
    default_weight=1,
    max_queue_depth=100,
    priority_levels=3,
    aging_policy=aging_policy,
)

# Set agent weights
scheduler.set_agent_weight("agent1", 2)  # Gets 2x processing time

# Enqueue with priority
scheduler.enqueue(intent, priority=0)

# Dequeue uses aged priorities
intent = scheduler.dequeue()
```

**Dequeue Logic:**
1. Calculate aged priority for all queued intents
2. Select intent with highest aged priority
3. Use queue priority as tiebreaker (lower queue number = higher priority)
4. Use FIFO within same aged priority and queue
5. Untrack intent from aging system

**Features:**
- Automatic priority aging integration
- Per-agent weight configuration
- Multiple priority levels
- Queue depth limits
- Comprehensive statistics including aging metrics

## Integration Example

Here's how the components work together:

```python
from gunn.utils.quota import QuotaController, QuotaPolicy
from gunn.utils.priority_aging import AgingPolicy
from gunn.utils.scheduling import WeightedRoundRobinScheduler

# Configure quota policy
quota_policy = QuotaPolicy(
    intents_per_minute=60,
    tokens_per_minute=10000,
)

# Configure aging policy
aging_policy = AgingPolicy(
    aging_rate=0.5,
    max_priority_boost=3,
    min_wait_time_seconds=2.0,
)

# Create components
quota_controller = QuotaController(quota_policy)
scheduler = WeightedRoundRobinScheduler(
    aging_policy=aging_policy,
    max_queue_depth=100,
)

# Process intent
async def process_intent(intent):
    # Check quota
    try:
        await quota_controller.check_intent_quota(intent["agent_id"])
    except QuotaExceededError:
        # Handle quota exceeded
        return
    
    # Enqueue for scheduling
    scheduler.enqueue(intent, priority=intent["priority"])
    
    # Dequeue and process (in background loop)
    next_intent = scheduler.dequeue()
    if next_intent:
        # Process the intent
        pass
```

## Configuration Guidelines

### Quota Configuration

**High-throughput scenarios:**
```python
QuotaPolicy(
    intents_per_minute=300,
    tokens_per_minute=50000,
    burst_size=50,
)
```

**Rate-limited scenarios:**
```python
QuotaPolicy(
    intents_per_minute=30,
    tokens_per_minute=5000,
    burst_size=5,
)
```

### Aging Configuration

**Aggressive anti-starvation:**
```python
AgingPolicy(
    aging_rate=1.0,           # Fast aging
    max_priority_boost=10,    # Large boost
    min_wait_time_seconds=0.5, # Quick start
)
```

**Conservative aging:**
```python
AgingPolicy(
    aging_rate=0.1,           # Slow aging
    max_priority_boost=2,     # Small boost
    min_wait_time_seconds=5.0, # Delayed start
)
```

**Disabled (strict priority):**
```python
AgingPolicy(enabled=False)
```

## Testing

Comprehensive test suites are provided:

- `src/gunn/utils/test_quota.py`: Token bucket and quota controller tests
- `src/gunn/utils/test_priority_aging.py`: Priority aging system tests
- `src/gunn/utils/test_scheduling_with_aging.py`: Scheduler integration tests

Run tests:
```bash
uv run pytest src/gunn/utils/test_quota.py -v
uv run pytest src/gunn/utils/test_priority_aging.py -v
uv run pytest src/gunn/utils/test_scheduling_with_aging.py -v
```

## Performance Considerations

### Token Bucket
- O(1) token consumption
- Minimal locking overhead
- Automatic cleanup on agent removal

### Priority Aging
- O(1) priority calculation
- O(n) for statistics (where n = tracked intents)
- Memory proportional to queued intents

### Scheduler
- O(n*m) dequeue complexity (n = agents, m = intents per agent)
- Can be optimized with priority heaps if needed
- Memory proportional to queued intents

## Future Enhancements

Potential improvements for future tasks:

1. **Adaptive Aging**: Adjust aging rate based on system load
2. **Priority Classes**: Group agents into priority classes
3. **Quota Sharing**: Allow quota sharing between related agents
4. **Predictive Scheduling**: Use ML to predict intent processing times
5. **Distributed Quotas**: Support for multi-node quota enforcement

## Requirements Addressed

This implementation addresses the following requirements:

- **18.1**: QuotaController with TokenBucket for per-agent rate limiting
- **18.2**: PriorityAging system to prevent starvation through wait-time priority increases
- **18.3**: Weighted round-robin scheduling with aging-adjusted priorities
- **18.4**: Configurable quota policies and backpressure responses
- **18.5**: Comprehensive unit tests for fairness, starvation prevention, and quota enforcement

## Related Documentation

- [Scheduling Utilities](../src/gunn/utils/scheduling.py)
- [Error Handling](./errors.md)
- [Performance & SLOs](./tech.md)
