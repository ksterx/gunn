# Temporal Authority and Duration Effects

## Overview

The temporal authority system provides sophisticated time management and duration-based effect handling for multi-agent simulations. It enables:

- **Temporal Authority Resolution**: Configurable priority-based time source selection (external adapters, simulation time, or wall-clock time)
- **Duration Effects**: Support for effects that span time intervals with start/end timestamps
- **Smooth Interpolation**: Automatic position interpolation for Move actions over their duration
- **Conflict Resolution**: Detection and resolution of overlapping duration-based effects

## Key Concepts

### Temporal Authority

Temporal authority determines which time source controls the simulation clock. The system supports three authority types with configurable priority:

1. **EXTERNAL** (highest priority): Time controlled by external adapters (Unity, Unreal)
2. **SIM_TIME** (medium priority): Simulation time from the orchestrator
3. **WALL_TIME** (lowest priority): Real wall-clock time

The authority resolution follows the configured priority order, falling back to the next available source if the preferred one is unavailable.

### Duration Effects

Duration effects are effects that persist over a time interval rather than being instantaneous. They include:

- **start_time**: When the effect begins (in the authority's time domain)
- **end_time**: When the effect completes (start_time + duration_ms)
- **authority**: Which temporal authority controls this effect's timing

Common use cases:
- Movement actions that take time to complete
- Continuous environmental effects
- Timed buffs/debuffs in game simulations

### Interpolation

For Move effects with duration, the system can automatically interpolate positions between start and end points. This enables:

- Smooth movement visualization
- Accurate collision detection during movement
- Realistic physics simulation

## Usage

### Basic Setup

```python
from gunn.utils.temporal import (
    TemporalAuthority,
    TemporalAuthorityManager,
    TemporalConfig,
)

# Create configuration
config = TemporalConfig(
    default_authority=TemporalAuthority.SIM_TIME,
    allow_overlapping_effects=True,
    max_effect_duration_ms=60000.0,
    interpolation_enabled=True,
)

# Initialize manager
manager = TemporalAuthorityManager(config)

# Register time sources
manager.register_time_source(TemporalAuthority.SIM_TIME, lambda: sim_time)
manager.register_time_source(TemporalAuthority.EXTERNAL, lambda: unity_time)
```

### Registering Duration Effects

```python
# Create an effect with duration
effect: Effect = {
    "uuid": "move-123",
    "kind": "Move",
    "payload": {
        "agent_id": "alice",
        "from": [0.0, 0.0, 0.0],
        "to": [10.0, 10.0, 0.0],
    },
    "duration_ms": 1000.0,  # 1 second duration
    "apply_at": None,  # Start immediately
    # ... other required fields
}

# Register the effect
duration_effect = manager.register_duration_effect(effect)
```

### Position Interpolation

```python
# Get interpolated position at current time
position = manager.interpolate_move_position(duration_effect)

# Get position at specific time
position_at_t = manager.interpolate_move_position(duration_effect, time=100.5)
```

### Conflict Detection and Resolution

```python
# Configure to prevent overlapping effects
config = TemporalConfig(allow_overlapping_effects=False)
manager = TemporalAuthorityManager(config)

# Register effects - conflicts will be automatically resolved by priority
effect1 = create_effect(priority=0)  # Low priority
effect2 = create_effect(priority=10)  # High priority

manager.register_duration_effect(effect1)
manager.register_duration_effect(effect2)  # Will cancel effect1 if overlapping
```

### Cleanup

```python
# Manually cleanup expired effects
cleaned_count = manager.cleanup_expired_effects(current_time)

# Or unregister specific effect
manager.unregister_duration_effect(effect_uuid)
```

## Configuration Options

### TemporalConfig

- **authority_priority**: List of authorities in priority order (default: [EXTERNAL, SIM_TIME, WALL_TIME])
- **default_authority**: Fallback authority if none available (default: SIM_TIME)
- **allow_overlapping_effects**: Whether to allow overlapping duration effects (default: True)
- **max_effect_duration_ms**: Maximum allowed effect duration in milliseconds (default: 60000.0)
- **interpolation_enabled**: Enable smooth interpolation for Move actions (default: True)

## Integration with Orchestrator

The temporal authority manager can be integrated with the Orchestrator to provide:

1. **Consistent Time Management**: Single source of truth for simulation time
2. **Duration-Based Effects**: Support for effects that span time intervals
3. **Smooth Movement**: Interpolated positions for agents during movement
4. **Conflict Resolution**: Automatic handling of overlapping effects

Example integration:

```python
class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        # ... existing initialization
        
        # Initialize temporal authority manager
        temporal_config = TemporalConfig(
            default_authority=TemporalAuthority.SIM_TIME,
        )
        self.temporal_manager = TemporalAuthorityManager(temporal_config)
        
        # Register time sources
        self.temporal_manager.register_time_source(
            TemporalAuthority.SIM_TIME,
            self._current_sim_time
        )
    
    async def broadcast_event(self, draft: EffectDraft) -> None:
        # Create effect
        effect = self._create_effect_from_draft(draft)
        
        # Register duration effect if applicable
        if effect.get("duration_ms"):
            self.temporal_manager.register_duration_effect(effect)
        
        # ... continue with normal broadcast
```

## Examples

See `examples/temporal_authority_demo.py` for comprehensive examples demonstrating:

- Basic temporal authority setup and resolution
- Duration effect registration and tracking
- Move action interpolation
- Conflict detection and resolution
- Manager statistics

## Requirements Addressed

This implementation addresses the following requirements:

- **19.1**: Effect schema extended with duration_ms and apply_at fields
- **19.2**: Temporal authority resolution with configurable priority (EXTERNAL > SIM_TIME > WALL_TIME)
- **19.3**: Move actions support start/end timestamps for smooth interpolation
- **19.4**: Conflict resolution for overlapping duration-based effects
- **19.5**: Comprehensive unit tests for temporal consistency and authority conflicts

## Performance Considerations

- **Memory**: Active effects are tracked in memory; use cleanup_expired_effects() periodically
- **Interpolation**: Position interpolation is O(1) and very fast
- **Conflict Detection**: O(n) where n is the number of active effects for the same entity
- **Time Source Calls**: Time sources are called on-demand; ensure they are fast

## Best Practices

1. **Choose Appropriate Authority**: Use EXTERNAL for game engine integration, SIM_TIME for deterministic simulations
2. **Set Reasonable Durations**: Keep effect durations within max_effect_duration_ms limits
3. **Cleanup Regularly**: Call cleanup_expired_effects() periodically to prevent memory growth
4. **Handle Conflicts**: Configure allow_overlapping_effects based on your simulation needs
5. **Use Priorities**: Assign meaningful priorities to effects for proper conflict resolution
