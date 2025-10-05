# Replay Invariance and Storage Consistency

## Overview

The replay invariance system ensures that incremental state updates produce identical results to full replay from the event log. This is critical for:

- **Storage Consistency**: Detecting state corruption or inconsistencies
- **Replay Correctness**: Ensuring deterministic replay behavior
- **Debugging**: Identifying when and where state diverges from expected
- **Auditability**: Validating that the system maintains integrity over time

## Core Components

### ReplayInvarianceValidator

The `ReplayInvarianceValidator` class provides comprehensive validation of state consistency:

```python
from gunn.utils.replay_invariance import ReplayInvarianceValidator
from gunn.core.event_log import EventLog
from gunn.schemas.messages import WorldState

# Create validator
validator = ReplayInvarianceValidator("my_world")

# Validate that incremental state matches full replay
report = await validator.validate_replay_invariance(
    event_log=log,
    incremental_state=current_state,
    from_seq=0,
    to_seq=None  # None = latest
)

if report.valid:
    print("State is consistent!")
else:
    print(f"Found {len(report.violations)} violations")
    for violation in report.violations:
        print(f"  {violation.field_path}: {violation.description}")
    
    print("Recovery options:")
    for option in report.recovery_options:
        print(f"  - {option}")
```

### Validation Report

The `ReplayInvarianceReport` contains:

- `valid`: Overall validation result
- `incremental_hash`: Hash of the incremental state
- `full_replay_hash`: Hash of the replayed state
- `violations`: List of detected inconsistencies
- `entries_checked`: Number of log entries validated
- `duration_seconds`: Time taken for validation
- `recovery_options`: Suggested recovery actions

### Consistency Violations

Each `ConsistencyViolation` includes:

- `global_seq`: Sequence number where violation was detected
- `field_path`: Path to the inconsistent field (e.g., "entities.agent_1")
- `incremental_value`: Value in incremental state
- `full_replay_value`: Value from full replay
- `description`: Human-readable description

## Usage Patterns

### Periodic Validation

For long-running simulations, use periodic validation:

```python
validator = ReplayInvarianceValidator("my_world")

# Validate every 100 entries
report = await validator.periodic_validation(
    event_log=log,
    incremental_state=current_state,
    interval_entries=100
)

if report is not None:
    # Validation was performed
    if not report.valid:
        # Handle violations
        pass
```

### State Snapshots

Create snapshots for faster validation and recovery:

```python
# Create snapshot at current sequence
snapshot = await validator.create_snapshot(
    state=current_state,
    global_seq=log.get_latest_seq()
)

# Later, retrieve snapshots
snapshots = validator.get_snapshots()
for snap in snapshots:
    print(f"Snapshot at seq {snap.global_seq}: hash={snap.state_hash[:8]}")
```

### EventLog Integration

The `EventLog` class includes built-in replay validation:

```python
# Async context - full replay checking available
result = await log.validate_integrity_async(
    include_replay_check=True,
    current_state=current_state
)

# Sync context - basic integrity only
result = log.validate_integrity(
    include_replay_check=False  # Replay check not available in sync
)
```

## Recovery Options

When violations are detected, the validator suggests recovery options:

### REBUILD_FROM_LOG
Rebuild the entire world state from full event log replay. This is the most comprehensive recovery option.

```python
if "REBUILD_FROM_LOG" in report.recovery_options:
    # Perform full replay
    entries = log.get_all_entries()
    new_state = await validator._full_replay(entries)
```

### SYNC_ENTITIES
Synchronize entity state from replay to incremental state.

### REBUILD_SPATIAL_INDEX
Rebuild the spatial index from entity positions.

### SYNC_METADATA
Synchronize metadata from replay state.

### PATCH_DIFFERENCES
Apply specific patches to fix identified violations (for small numbers of violations).

### FULL_RESET
Complete state reset recommended when too many violations exist (>10).

### CHECKPOINT_AND_CONTINUE
Create a checkpoint with the replay state and continue from there.

## Effect Application

The validator includes a simplified effect application system that handles common effect types:

- **AgentJoined**: Adds agent to entities and spatial index
- **AgentLeft**: Removes agent from entities and spatial index
- **Move**: Updates agent position in spatial index
- **MessageSent/Speak**: Stores messages in metadata
- **Interact**: Stores interactions in metadata

For production use, this should match the Orchestrator's effect application logic exactly.

## Performance Considerations

### Validation Frequency

- **Development**: Validate frequently (every 10-100 entries)
- **Production**: Validate less frequently (every 1000-10000 entries)
- **Critical Systems**: Validate after every state-modifying operation

### Optimization Strategies

1. **Incremental Validation**: Only validate new entries since last check
2. **Snapshot-Based**: Use snapshots to avoid full replay
3. **Parallel Validation**: Run validation in background thread
4. **Sampling**: Validate random samples for large logs

### Memory Management

```python
# Clear old snapshots to free memory
validator.clear_snapshots()

# Compact event log to reduce memory
removed = await log.compact(keep_entries=1000)
```

## Testing

Comprehensive tests are provided:

- **Unit Tests**: `src/gunn/utils/test_replay_invariance.py`
- **Integration Tests**: `tests/integration/test_replay_invariance_integration.py`

Run tests:
```bash
uv run pytest src/gunn/utils/test_replay_invariance.py -v
uv run pytest tests/integration/test_replay_invariance_integration.py -v
```

## Best Practices

1. **Always validate after major operations**: State changes, compaction, recovery
2. **Use periodic validation**: Catch issues early in long-running simulations
3. **Create snapshots regularly**: Enable faster recovery and validation
4. **Monitor validation metrics**: Track validation frequency and violation rates
5. **Test recovery procedures**: Ensure recovery options work as expected
6. **Log validation results**: Keep audit trail of validation checks

## Example: Complete Validation Workflow

```python
from gunn.core.event_log import EventLog
from gunn.schemas.messages import WorldState
from gunn.utils.replay_invariance import ReplayInvarianceValidator

async def validate_simulation_state(
    log: EventLog,
    current_state: WorldState,
    world_id: str
) -> bool:
    """Complete validation workflow with recovery."""
    
    validator = ReplayInvarianceValidator(world_id)
    
    # Perform validation
    report = await validator.validate_replay_invariance(
        event_log=log,
        incremental_state=current_state
    )
    
    if report.valid:
        # Create snapshot for future use
        await validator.create_snapshot(
            state=current_state,
            global_seq=log.get_latest_seq()
        )
        return True
    
    # Handle violations
    print(f"Validation failed with {len(report.violations)} violations")
    
    # Log violations
    for violation in report.violations:
        print(f"  {violation.field_path}: {violation.description}")
    
    # Attempt recovery
    if "REBUILD_FROM_LOG" in report.recovery_options:
        print("Rebuilding state from log...")
        entries = log.get_all_entries()
        recovered_state = await validator._full_replay(entries)
        
        # Verify recovery
        verify_report = await validator.validate_replay_invariance(
            event_log=log,
            incremental_state=recovered_state
        )
        
        if verify_report.valid:
            print("Recovery successful!")
            return True
    
    print("Recovery failed - manual intervention required")
    return False
```

## Requirements Addressed

This implementation addresses the following requirements:

- **20.1**: Create ReplayInvarianceValidator for comparing incremental vs full replay results
- **20.2**: Add periodic replay-invariance tests during long-running simulations
- **20.3**: Implement detailed diagnostics for consistency violations with recovery options
- **20.4**: Extend EventLog.validate_integrity() to include replay equivalence checks
- **20.5**: Write comprehensive tests for replay consistency, state integrity, and error recovery

## See Also

- [Event Log Documentation](./structure.md#event-log)
- [Determinism and Replay](./tech.md#determinism)
- [Testing Strategy](../tests/README.md)
