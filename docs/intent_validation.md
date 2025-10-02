# Intent Payload Validation Guide

This document describes the validation rules for each Intent kind and their expected payload structures.

## Standard Intent Kinds

### Move Intent

**Kind**: `"Move"`

**Payload Structure**:
```python
{
    "to": [x: float, y: float, z: float],  # Target position (preferred)
    # OR
    "position": [x: float, y: float, z: float]  # Legacy format
}
```

**Validation Rules**:
- Position must be a list or tuple with 2 or 3 numeric elements
- 2D positions are automatically converted to 3D (z=0.0)
- Position values must be valid floats
- No validation on maximum distance (handled by game logic)

**Example**:
```python
intent = {
    "kind": "Move",
    "payload": {"to": [10.0, 20.0, 0.0]},
    "agent_id": "agent_1",
    "req_id": "move_123",
    "context_seq": 42,
    "priority": 10,
    "schema_version": "1.0.0"
}
```

---

### Speak Intent

**Kind**: `"Speak"`

**Payload Structure**:
```python
{
    "text": str,  # Message content
    "channel": str | None,  # Optional: "public" (default), "team", "private"
    "target_id": str | None  # Optional: recipient for private messages
}
```

**Validation Rules**:
- `text` field is required and must be non-empty string
- `channel` defaults to "public" if not specified
- `target_id` required only for private channel messages

**Example**:
```python
intent = {
    "kind": "Speak",
    "payload": {
        "text": "Hello, team!",
        "channel": "team"
    },
    "agent_id": "agent_1",
    "req_id": "speak_456",
    "context_seq": 42,
    "priority": 10,
    "schema_version": "1.0.0"
}
```

---

### Interact Intent

**Kind**: `"Interact"`

**Payload Structure**:
```python
{
    "target_id": str,  # Entity to interact with
    "interaction_type": str | None,  # Optional: "pickup", "use", "examine", etc.
    "data": dict | None  # Optional: interaction-specific data
}
```

**Validation Rules**:
- `target_id` is required
- Target entity must exist in world state
- Spatial proximity validation (if enabled in validator)
- Custom interaction types can be defined per-game

**Example**:
```python
intent = {
    "kind": "Interact",
    "payload": {
        "target_id": "object_42",
        "interaction_type": "pickup"
    },
    "agent_id": "agent_1",
    "req_id": "interact_789",
    "context_seq": 42,
    "priority": 10,
    "schema_version": "1.0.0"
}
```

---

### Custom Intent

**Kind**: `"Custom"`

**Payload Structure**:
```python
{
    "action": str,  # Custom action identifier
    # ... any additional fields
}
```

**Validation Rules**:
- Payload structure is application-specific
- Validation handled by custom EffectValidator
- Should include `action` field to identify custom action type

**Example**:
```python
intent = {
    "kind": "Custom",
    "payload": {
        "action": "attack",
        "target_id": "enemy_1",
        "damage": 10
    },
    "agent_id": "agent_1",
    "req_id": "custom_999",
    "context_seq": 42,
    "priority": 10,
    "schema_version": "1.0.0"
}
```

---

## Common Validation Rules

All intents must include these fields:

### Required Fields
- `kind`: One of `"Move"`, `"Speak"`, `"Interact"`, `"Custom"`
- `payload`: Dictionary with kind-specific data
- `agent_id`: ID of the submitting agent
- `req_id`: Unique request identifier for deduplication
- `context_seq`: Agent's view sequence number
- `priority`: Processing priority (0-100, higher = more urgent)
- `schema_version`: Schema version string (e.g., "1.0.0")

### System Validation Stages

1. **Structure Validation**: Required fields present and correct types
2. **Agent Validation**: Agent is registered and authorized
3. **Staleness Detection**: `context_seq` within threshold of current `global_seq`
4. **Quota Checking**: Agent within rate limits for intent kind
5. **Backpressure Checking**: System not overloaded
6. **Domain Validation**: Custom EffectValidator checks (cooldowns, permissions, etc.)

---

## Custom Validation

To implement custom validation rules, create a custom `EffectValidator`:

```python
from gunn.core.orchestrator import EffectValidator
from gunn.schemas.messages import WorldState
from gunn.schemas.types import Intent

class MyCustomValidator:
    def validate_intent(self, intent: Intent, world_state: WorldState) -> bool:
        kind = intent["kind"]
        payload = intent["payload"]

        if kind == "Custom":
            action = payload.get("action")
            if action == "attack":
                # Validate attack-specific rules
                target_id = payload.get("target_id")
                if not target_id:
                    return False
                # Check if target exists and is in range
                # ...
                return True

        # Fall back to default validation
        return True

    def set_agent_permissions(self, agent_id: str, permissions: set[str]) -> None:
        # Store agent permissions
        pass

# Use custom validator
orchestrator = Orchestrator(
    config=config,
    effect_validator=MyCustomValidator()
)
```

---

## Simultaneous Intent Submission

For submitting multiple intents at the same simulation time (e.g., speak + move):

```python
intents = [
    {
        "kind": "Speak",
        "payload": {"text": "Moving forward!"},
        "agent_id": "agent_1",
        "req_id": "speak_001",
        "context_seq": 42,
        "priority": 10,
        "schema_version": "1.0.0"
    },
    {
        "kind": "Move",
        "payload": {"to": [10.0, 20.0, 0.0]},
        "agent_id": "agent_1",
        "req_id": "move_001",
        "context_seq": 42,
        "priority": 10,
        "schema_version": "1.0.0"
    }
]

# Submit all intents for the same simulation time
req_ids = await orchestrator.submit_intents(intents, sim_time=current_time)
```

**Validation Notes**:
- All intents validated before any are enqueued (atomic batch)
- If any validation fails, none are processed
- Quota/cooldown checks are per intent kind (speak won't block move)
- Intents ordered by priority within same sim_time

---

## Error Handling

### Common Validation Errors

- **`ValueError`**: Invalid intent structure (missing required fields)
- **`StaleContextError`**: Intent's `context_seq` too old (> staleness threshold)
- **`QuotaExceededError`**: Agent exceeded rate limit for intent kind
- **`BackpressureError`**: System overloaded (queue depth or CPU threshold)
- **`ValidationError`**: Custom validator rejected intent

### Example Error Handling

```python
try:
    req_id = await orchestrator.submit_intent(intent)
except StaleContextError as e:
    # Agent view is outdated, refresh and retry
    print(f"Context stale: {e.current_seq} vs {e.context_seq}")
    await agent.refresh_view()
except QuotaExceededError as e:
    # Rate limited, wait and retry
    print(f"Quota exceeded for {e.agent_id}")
    await asyncio.sleep(1.0)
except ValidationError as e:
    # Intent rejected by validator
    print(f"Validation failed: {e}")
```

---

## Best Practices

1. **Use Specific Intent Kinds**: Prefer `"Move"`, `"Speak"`, `"Interact"` over `"Custom"` when possible
2. **Include Context**: Always set accurate `context_seq` from latest agent view
3. **Unique Request IDs**: Use UUIDs or timestamp-based IDs for `req_id`
4. **Appropriate Priority**: Reserve high priority (>50) for time-critical actions
5. **Batch Simultaneous Actions**: Use `submit_intents()` for speak+move combinations
6. **Handle Staleness**: Implement retry logic with view refresh on `StaleContextError`
7. **Respect Quotas**: Monitor rate limits and implement backoff strategies
8. **Test Custom Validation**: Thoroughly test custom validators with edge cases

---

## Schema Versioning

The `schema_version` field enables forward compatibility:

- Current version: `"1.0.0"`
- Orchestrator logs warnings for unknown versions
- Future versions may add optional fields
- Breaking changes increment major version

Always use the current schema version for new intents.
