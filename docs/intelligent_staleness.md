# Intelligent Staleness Detection

## Overview

Intelligent staleness detection is a feature of the `SpatialObservationPolicy` that prevents false positives by only triggering intent cancellation when relevant preconditions change. This significantly improves efficiency by avoiding unnecessary LLM generation cancellations.

## Problem Statement

In multi-agent simulations, agents generate intents (actions) based on their current observation of the world. When the world state changes during intent generation (e.g., LLM is generating a response), the system needs to decide whether to cancel the generation and regenerate with updated context.

**Naive approach**: Cancel whenever *any* world change occurs
- **Problem**: Causes many false positives
- **Impact**: Wastes LLM generation costs and increases latency
- **Example**: Agent A is generating a move intent when Agent B (far away) moves → unnecessary cancellation

**Intelligent approach**: Cancel only when *relevant* preconditions change
- **Benefit**: Prevents false positives
- **Impact**: Reduces LLM costs and improves responsiveness
- **Example**: Agent A's move intent only cancelled if A's position changes or obstacles appear near target

## Intent-Specific Staleness Logic

### Move Intents

A Move intent is considered stale if:

1. **Agent's position changed significantly** beyond the configured threshold
   - Threshold: `move_position_threshold` (default: 1.0 units)
   - Rationale: Agent's current position affects path planning

2. **New obstacles appeared near target** within the threshold distance
   - Rationale: Path to target may be blocked

**Not stale when**:
- Unrelated agents move far away
- Non-spatial world changes occur
- Changes are within the threshold

### Speak Intents

A Speak intent is considered stale if:

1. **Nearby agents changed** (someone joined or left conversation range)
   - Threshold: `speak_proximity_threshold` (default: 5.0 units)
   - Rationale: Audience for the message changed

**Not stale when**:
- Non-agent entities appear/disappear
- Agents move but conversation participants remain the same
- Changes occur outside conversation range

### Custom Intents

Custom intent types use configurable default behavior:
- `default_staleness_enabled=True`: Always stale (conservative)
- `default_staleness_enabled=False`: Never stale (optimistic)

## Configuration

### Basic Configuration

```python
from gunn.policies.observation import (
    PolicyConfig,
    SpatialObservationPolicy,
    StalenessConfig,
)

# Create policy with staleness configuration
policy = SpatialObservationPolicy(
    PolicyConfig(distance_limit=10.0),
    StalenessConfig(
        move_position_threshold=1.0,      # Position change threshold for Move
        speak_proximity_threshold=5.0,     # Proximity threshold for Speak
        default_staleness_enabled=True,    # Default for unknown intent types
    ),
)
```

### Agent-Specific Thresholds

Different agents can have different staleness thresholds:

```python
policy = SpatialObservationPolicy(
    PolicyConfig(distance_limit=10.0),
    StalenessConfig(
        move_position_threshold=1.0,  # Default threshold
        agent_specific_thresholds={
            "vip_agent": {
                "Move": 5.0,   # VIP agent gets higher threshold
                "Speak": 10.0,
            },
            "precise_agent": {
                "Move": 0.1,   # Precise agent gets lower threshold
            },
        },
    ),
)
```

## Usage

### Checking Intent Staleness

```python
from gunn.schemas.messages import WorldState
from gunn.schemas.types import Intent

# Create world states
old_state = WorldState(
    entities={"agent1": {"type": "agent"}},
    spatial_index={"agent1": (0.0, 0.0, 0.0)},
)

new_state = WorldState(
    entities={"agent1": {"type": "agent"}},
    spatial_index={"agent1": (2.0, 0.0, 0.0)},  # Agent moved
)

# Create intent
move_intent: Intent = {
    "kind": "Move",
    "payload": {"to": [10.0, 5.0, 0.0]},
    "context_seq": 1,
    "req_id": "req1",
    "agent_id": "agent1",
    "priority": 0,
    "schema_version": "1.0.0",
}

# Check staleness
is_stale = policy.is_intent_stale(move_intent, old_state, new_state)
if is_stale:
    # Cancel generation and regenerate with new context
    cancel_token.cancel("stale_context")
```

### Integration with Orchestrator

The `SpatialObservationPolicy` can be used with the Orchestrator for automatic staleness detection:

```python
from gunn.core.orchestrator import Orchestrator, OrchestratorConfig

# Create orchestrator with spatial policy
orchestrator = Orchestrator(
    config=OrchestratorConfig(staleness_threshold=0),
    world_id="simulation",
)

# Register agent with spatial observation policy
policy = SpatialObservationPolicy(
    PolicyConfig(distance_limit=10.0),
    StalenessConfig(move_position_threshold=1.0),
)

agent_handle = await orchestrator.register_agent(
    agent_id="agent1",
    policy=policy,
)
```

## Performance Benefits

### Efficiency Comparison

**Scenario**: 5 agents in a world, 4 agents move far from agent1

| Approach | Move Intent | Speak Intent | LLM Generations Saved |
|----------|-------------|--------------|----------------------|
| Naive    | ❌ Cancelled | ❌ Cancelled | 0% |
| Intelligent | ✅ Not Stale | ✅ Not Stale | 100% |

### Cost Savings

For a simulation with:
- 10 agents
- 100 intents/minute
- 30% unrelated world changes
- $0.01 per LLM generation

**Naive approach**: 30 unnecessary cancellations/minute = $18/hour wasted
**Intelligent approach**: ~0 unnecessary cancellations = $0/hour wasted

**Savings**: ~$18/hour or ~$432/day

## Best Practices

### 1. Tune Thresholds for Your Use Case

```python
# For fast-paced action games
StalenessConfig(
    move_position_threshold=0.5,   # Tight threshold
    speak_proximity_threshold=3.0,
)

# For strategic simulations
StalenessConfig(
    move_position_threshold=2.0,   # Loose threshold
    speak_proximity_threshold=10.0,
)
```

### 2. Use Agent-Specific Thresholds

```python
# Different agent types need different thresholds
StalenessConfig(
    agent_specific_thresholds={
        "sniper": {"Move": 0.1},      # Precise positioning
        "tank": {"Move": 5.0},         # Rough positioning
        "diplomat": {"Speak": 15.0},   # Large audience
        "spy": {"Speak": 2.0},         # Small audience
    },
)
```

### 3. Monitor False Positives and Negatives

```python
# Log staleness decisions for analysis
if is_stale:
    logger.info(
        "Intent marked stale",
        agent_id=intent["agent_id"],
        intent_kind=intent["kind"],
        position_change=position_delta,
        threshold=threshold,
    )
```

### 4. Consider Intent Complexity

For complex intents that take longer to generate:
- Use tighter thresholds (more sensitive to changes)
- Implement custom staleness logic if needed

For simple intents that generate quickly:
- Use looser thresholds (less sensitive to changes)
- May not need staleness detection at all

## Implementation Details

### Distance Calculation

Uses Euclidean distance in 3D space:

```python
def _calculate_distance(
    pos1: tuple[float, float, float],
    pos2: tuple[float, float, float],
) -> float:
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dz = pos1[2] - pos2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)
```

### Coordinate System Support

Supports both 2D and 3D coordinates:
- 2D coordinates `[x, y]` automatically converted to 3D `[x, y, 0.0]`
- 3D coordinates `[x, y, z]` used directly

### Entity Type Detection

Only agents (entities with `type="agent"`) are considered for Speak intent staleness:

```python
def _get_agents_near_position(
    world_state: WorldState,
    position: tuple[float, float, float],
    radius: float,
) -> set[str]:
    nearby = set()
    for entity_id, entity_position in world_state.spatial_index.items():
        entity_data = world_state.entities.get(entity_id)
        if entity_data and entity_data.get("type") == "agent":
            distance = self._calculate_distance(position, entity_position)
            if distance <= radius:
                nearby.add(entity_id)
    return nearby
```

## Testing

Comprehensive test suite covers:

1. **Move intent staleness**
   - Position unchanged → not stale
   - Position changed beyond threshold → stale
   - Position changed within threshold → not stale
   - Obstacle appeared near target → stale
   - 2D coordinate support

2. **Speak intent staleness**
   - Nearby agents unchanged → not stale
   - Agent joined conversation → stale
   - Agent left conversation → stale
   - Non-agent entity appeared → not stale

3. **Agent-specific thresholds**
   - Override default thresholds
   - Per-agent, per-intent configuration

4. **False positive prevention**
   - Unrelated changes don't trigger staleness
   - Spatial awareness prevents false positives

5. **Timing and accuracy**
   - Consistent results across multiple checks
   - Quick detection of relevant changes

Run tests:

```bash
uv run pytest src/gunn/policies/test_staleness.py -v
```

## Examples

See `examples/intelligent_staleness_demo.py` for a comprehensive demonstration of:
- Move intent staleness scenarios
- Speak intent staleness scenarios
- Agent-specific thresholds
- Efficiency comparison with naive approach

Run demo:

```bash
uv run python examples/intelligent_staleness_demo.py
```

## Future Enhancements

Potential improvements for future versions:

1. **Predictive staleness**: Predict when intent will become stale based on agent trajectories
2. **Intent-specific policies**: Allow custom staleness logic per intent type
3. **Learning-based thresholds**: Automatically tune thresholds based on historical data
4. **Partial regeneration**: Regenerate only affected parts of intent instead of full cancellation
5. **Staleness confidence**: Provide confidence score instead of binary stale/not-stale

## References

- Requirements: 17.1-17.6 in `.kiro/specs/multi-agent-simulation-core/requirements.md`
- Implementation: `src/gunn/policies/observation.py` - `SpatialObservationPolicy`
- Tests: `src/gunn/policies/test_staleness.py`
- Demo: `examples/intelligent_staleness_demo.py`
