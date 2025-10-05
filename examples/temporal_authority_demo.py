"""Demo of temporal authority and duration-based effects.

This example demonstrates:
1. Temporal authority resolution with configurable priority
2. Duration effect registration and tracking
3. Smooth interpolation for Move actions
4. Conflict detection and resolution for overlapping effects
"""

import asyncio
import time

from gunn.schemas.types import Effect
from gunn.utils.temporal import (
    TemporalAuthority,
    TemporalAuthorityManager,
    TemporalConfig,
)


def create_move_effect(
    agent_id: str,
    from_pos: list[float],
    to_pos: list[float],
    duration_ms: float,
    priority: int = 0,
    apply_at: float | None = None,
) -> Effect:
    """Create a Move effect with duration."""
    return {
        "uuid": f"move-{agent_id}-{time.time()}",
        "kind": "Move",
        "payload": {
            "agent_id": agent_id,
            "from": from_pos,
            "to": to_pos,
        },
        "global_seq": 0,
        "sim_time": 0.0,
        "source_id": agent_id,
        "schema_version": "1.0.0",
        "req_id": f"req-{agent_id}-{time.time()}",
        "duration_ms": duration_ms,
        "apply_at": apply_at,
        "priority": priority,
    }


async def demo_basic_temporal_authority() -> None:
    """Demonstrate basic temporal authority setup and resolution."""
    print("\n=== Demo 1: Basic Temporal Authority ===\n")

    # Create configuration with default settings
    config = TemporalConfig(
        default_authority=TemporalAuthority.SIM_TIME,
        allow_overlapping_effects=True,
    )

    manager = TemporalAuthorityManager(config)

    # Register time sources
    sim_time = 100.0
    manager.register_time_source(TemporalAuthority.SIM_TIME, lambda: sim_time)
    manager.register_time_source(TemporalAuthority.WALL_TIME, time.time)
    manager.register_time_source(TemporalAuthority.EXTERNAL, lambda: 200.0)

    print(
        f"Registered authorities: {[a.value for a in manager._authority_sources.keys()]}"
    )
    print(f"Default authority: {manager.config.default_authority.value}")
    print(f"Current authority: {manager._current_authority.value}")

    # Resolve authority (should use EXTERNAL as highest priority)
    resolved = manager.resolve_authority()
    print(f"Resolved authority: {resolved.value}")

    # Get current time from different authorities
    print(
        f"\nTime from SIM_TIME: {manager.get_current_time(TemporalAuthority.SIM_TIME)}"
    )
    print(f"Time from EXTERNAL: {manager.get_current_time(TemporalAuthority.EXTERNAL)}")
    print(
        f"Time from WALL_TIME: {manager.get_current_time(TemporalAuthority.WALL_TIME):.2f}"
    )


async def demo_duration_effects() -> None:
    """Demonstrate duration effect registration and tracking."""
    print("\n=== Demo 2: Duration Effects ===\n")

    config = TemporalConfig()
    manager = TemporalAuthorityManager(config)

    # Register time source
    current_time = 100.0
    manager.register_time_source(TemporalAuthority.SIM_TIME, lambda: current_time)
    manager.register_time_source(TemporalAuthority.EXTERNAL, lambda: current_time)

    # Create and register a move effect with 2 second duration
    move_effect = create_move_effect(
        agent_id="alice",
        from_pos=[0.0, 0.0, 0.0],
        to_pos=[10.0, 10.0, 0.0],
        duration_ms=2000.0,
    )

    duration_effect = manager.register_duration_effect(move_effect)

    print(f"Registered effect: {duration_effect.effect['uuid']}")
    print(f"Start time: {duration_effect.start_time}")
    print(f"End time: {duration_effect.end_time}")
    print(f"Duration: {duration_effect.end_time - duration_effect.start_time} seconds")
    print(f"Authority: {duration_effect.authority.value}")

    # Check active effects
    active = manager.get_active_effects()
    print(f"\nActive effects: {len(active)}")

    # Cleanup expired effects (none should be expired yet)
    cleaned = manager.cleanup_expired_effects(current_time + 1.0)
    print(f"Cleaned up at t+1s: {cleaned} effects")

    # Cleanup after expiration
    cleaned = manager.cleanup_expired_effects(current_time + 3.0)
    print(f"Cleaned up at t+3s: {cleaned} effects")


async def demo_move_interpolation() -> None:
    """Demonstrate smooth position interpolation for Move effects."""
    print("\n=== Demo 3: Move Interpolation ===\n")

    config = TemporalConfig(interpolation_enabled=True)
    manager = TemporalAuthorityManager(config)

    # Register time source
    start_time = 100.0
    manager.register_time_source(TemporalAuthority.SIM_TIME, lambda: start_time)
    manager.register_time_source(TemporalAuthority.EXTERNAL, lambda: start_time)

    # Create a move effect from (0,0,0) to (10,10,0) over 1 second
    move_effect = create_move_effect(
        agent_id="alice",
        from_pos=[0.0, 0.0, 0.0],
        to_pos=[10.0, 10.0, 0.0],
        duration_ms=1000.0,
    )

    duration_effect = manager.register_duration_effect(move_effect)

    print(
        f"Move from {move_effect['payload']['from']} to {move_effect['payload']['to']}"
    )
    print(f"Duration: {move_effect['duration_ms']}ms\n")

    # Interpolate at different time points
    time_points = [
        (start_time, "Start"),
        (start_time + 0.25, "25%"),
        (start_time + 0.5, "50%"),
        (start_time + 0.75, "75%"),
        (start_time + 1.0, "End"),
    ]

    for t, label in time_points:
        pos = manager.interpolate_move_position(duration_effect, t)
        print(f"{label:6s} (t={t:6.2f}): position = {pos}")


async def demo_conflict_detection() -> None:
    """Demonstrate conflict detection for overlapping effects."""
    print("\n=== Demo 4: Conflict Detection ===\n")

    # Allow overlapping effects
    config = TemporalConfig(allow_overlapping_effects=True)
    manager = TemporalAuthorityManager(config)

    start_time = 100.0
    manager.register_time_source(TemporalAuthority.SIM_TIME, lambda: start_time)
    manager.register_time_source(TemporalAuthority.EXTERNAL, lambda: start_time)

    # Register first move effect for alice
    effect1 = create_move_effect(
        agent_id="alice",
        from_pos=[0.0, 0.0, 0.0],
        to_pos=[10.0, 10.0, 0.0],
        duration_ms=2000.0,
    )

    duration1 = manager.register_duration_effect(effect1)
    print(f"Effect 1: {duration1.start_time:.1f} - {duration1.end_time:.1f}")

    # Register overlapping move effect for alice
    effect2 = create_move_effect(
        agent_id="alice",
        from_pos=[5.0, 5.0, 0.0],
        to_pos=[15.0, 15.0, 0.0],
        duration_ms=2000.0,
        apply_at=start_time + 1.0,  # Starts 1 second later
    )

    duration2 = manager.register_duration_effect(effect2)
    print(f"Effect 2: {duration2.start_time:.1f} - {duration2.end_time:.1f}")

    # Check for conflicts
    conflicts = manager.find_conflicts(duration2)
    print(f"\nConflicts detected: {len(conflicts)}")
    for conflict in conflicts:
        print(
            f"  - {conflict.effect['uuid']}: {conflict.start_time:.1f} - {conflict.end_time:.1f}"
        )

    # Show all active effects
    active = manager.get_active_effects(entity_id="alice")
    print(f"\nActive effects for alice: {len(active)}")


async def demo_conflict_resolution() -> None:
    """Demonstrate conflict resolution based on priority."""
    print("\n=== Demo 5: Conflict Resolution ===\n")

    # Disable overlapping to trigger conflict resolution
    config = TemporalConfig(allow_overlapping_effects=False)
    manager = TemporalAuthorityManager(config)

    start_time = 100.0
    manager.register_time_source(TemporalAuthority.SIM_TIME, lambda: start_time)
    manager.register_time_source(TemporalAuthority.EXTERNAL, lambda: start_time)

    # Register low priority effect
    effect1 = create_move_effect(
        agent_id="alice",
        from_pos=[0.0, 0.0, 0.0],
        to_pos=[10.0, 10.0, 0.0],
        duration_ms=2000.0,
        priority=0,
    )

    duration1 = manager.register_duration_effect(effect1)
    print(f"Low priority effect registered: priority={effect1['priority']}")
    print(f"  UUID: {effect1['uuid']}")
    print(f"  Time: {duration1.start_time:.1f} - {duration1.end_time:.1f}")

    # Register high priority overlapping effect
    effect2 = create_move_effect(
        agent_id="alice",
        from_pos=[5.0, 5.0, 0.0],
        to_pos=[15.0, 15.0, 0.0],
        duration_ms=2000.0,
        priority=10,
        apply_at=start_time + 1.0,
    )

    duration2 = manager.register_duration_effect(effect2)
    print(f"\nHigh priority effect registered: priority={effect2['priority']}")
    print(f"  UUID: {effect2['uuid']}")
    print(f"  Time: {duration2.start_time:.1f} - {duration2.end_time:.1f}")

    # Check which effects are still active
    active = manager.get_active_effects(entity_id="alice")
    print(f"\nActive effects after conflict resolution: {len(active)}")
    for effect in active:
        print(
            f"  - {effect.effect['uuid']}: priority={effect.effect.get('priority', 0)}"
        )


async def demo_statistics() -> None:
    """Demonstrate manager statistics."""
    print("\n=== Demo 6: Manager Statistics ===\n")

    config = TemporalConfig()
    manager = TemporalAuthorityManager(config)

    # Register time sources
    manager.register_time_source(TemporalAuthority.SIM_TIME, lambda: 100.0)
    manager.register_time_source(TemporalAuthority.EXTERNAL, lambda: 200.0)

    # Register some effects
    for i in range(3):
        effect = create_move_effect(
            agent_id=f"agent_{i}",
            from_pos=[0.0, 0.0, 0.0],
            to_pos=[10.0, 10.0, 0.0],
            duration_ms=1000.0,
        )
        manager.register_duration_effect(effect)

    # Get statistics
    stats = manager.get_stats()

    print("Manager Statistics:")
    print(f"  Current authority: {stats['current_authority']}")
    print(f"  Registered authorities: {stats['registered_authorities']}")
    print(f"  Active effects: {stats['active_effects_count']}")
    print("\nConfiguration:")
    print(f"  Default authority: {stats['config']['default_authority']}")
    print(f"  Allow overlapping: {stats['config']['allow_overlapping_effects']}")
    print(f"  Max duration: {stats['config']['max_effect_duration_ms']}ms")
    print(f"  Interpolation: {stats['config']['interpolation_enabled']}")


async def main() -> None:
    """Run all demos."""
    print("=" * 60)
    print("Temporal Authority and Duration Effects Demo")
    print("=" * 60)

    await demo_basic_temporal_authority()
    await demo_duration_effects()
    await demo_move_interpolation()
    await demo_conflict_detection()
    await demo_conflict_resolution()
    await demo_statistics()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
