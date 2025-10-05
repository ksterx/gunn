"""Temporal authority and action duration management.

This module provides utilities for managing temporal authority resolution,
action duration tracking, and conflict resolution for overlapping duration-based effects.

Requirements addressed:
- 19.1: Extend Effect schema with duration_ms and apply_at for interval effects
- 19.2: Implement temporal authority resolution with configurable priority
- 19.3: Add support for Move actions with start/end timestamps for smooth interpolation
- 19.4: Create conflict resolution for overlapping duration-based effects
- 19.5: Write unit tests for temporal consistency, authority conflicts, and duration handling
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from gunn.schemas.types import Effect
from gunn.utils.telemetry import get_logger

logger = get_logger(__name__)


class TemporalAuthority(Enum):
    """Temporal authority sources for time synchronization.

    Priority order: EXTERNAL > SIM_TIME > WALL_TIME
    """

    EXTERNAL = "external"  # External adapter (Unity, Unreal) controls time
    SIM_TIME = "sim_time"  # Simulation time from orchestrator
    WALL_TIME = "wall_time"  # Real wall-clock time


@dataclass
class TemporalConfig:
    """Configuration for temporal authority and duration handling."""

    authority_priority: list[TemporalAuthority] = (
        None  # Priority order for authority resolution
    )
    default_authority: TemporalAuthority = TemporalAuthority.SIM_TIME
    allow_overlapping_effects: bool = (
        True  # Whether to allow overlapping duration effects
    )
    max_effect_duration_ms: float = (
        60000.0  # Maximum allowed effect duration (1 minute)
    )
    interpolation_enabled: bool = True  # Enable smooth interpolation for Move actions

    def __post_init__(self) -> None:
        """Initialize default authority priority if not provided."""
        if self.authority_priority is None:
            self.authority_priority = [
                TemporalAuthority.EXTERNAL,
                TemporalAuthority.SIM_TIME,
                TemporalAuthority.WALL_TIME,
            ]


@dataclass
class DurationEffect:
    """Represents an effect with duration and temporal information."""

    effect: Effect
    start_time: float  # When the effect starts (in authority's time domain)
    end_time: float  # When the effect ends (start_time + duration_ms)
    authority: TemporalAuthority  # Which temporal authority controls this effect

    def overlaps_with(self, other: "DurationEffect") -> bool:
        """Check if this effect overlaps with another effect.

        Args:
            other: Another duration effect to check against

        Returns:
            True if the effects overlap in time
        """
        return not (
            self.end_time <= other.start_time or other.end_time <= self.start_time
        )

    def affects_same_entity(self, other: "DurationEffect") -> bool:
        """Check if this effect affects the same entity as another effect.

        Args:
            other: Another duration effect to check against

        Returns:
            True if both effects affect the same entity
        """
        # Extract entity ID from effect payload
        self_entity = self.effect.get("payload", {}).get("agent_id") or self.effect.get(
            "source_id"
        )
        other_entity = other.effect.get("payload", {}).get(
            "agent_id"
        ) or other.effect.get("source_id")

        return self_entity == other_entity and self_entity is not None


class TemporalAuthorityManager:
    """Manages temporal authority resolution and duration-based effects.

    This class handles:
    - Temporal authority resolution based on configurable priority
    - Duration effect tracking and conflict detection
    - Smooth interpolation for Move actions with start/end timestamps
    - Conflict resolution for overlapping duration-based effects

    Examples
    --------
    >>> config = TemporalConfig(default_authority=TemporalAuthority.SIM_TIME)
    >>> manager = TemporalAuthorityManager(config)
    >>> effect = {
    ...     "uuid": "123",
    ...     "kind": "Move",
    ...     "payload": {"agent_id": "alice", "to": [10, 20, 0]},
    ...     "duration_ms": 1000.0,
    ...     "apply_at": None,
    ...     "sim_time": 100.0,
    ...     "global_seq": 1,
    ...     "source_id": "alice",
    ...     "schema_version": "1.0.0",
    ...     "req_id": "req_1"
    ... }
    >>> manager.register_duration_effect(effect, TemporalAuthority.SIM_TIME)
    """

    def __init__(self, config: TemporalConfig | None = None) -> None:
        """Initialize temporal authority manager.

        Args:
            config: Temporal configuration (uses defaults if None)
        """
        self.config = config or TemporalConfig()
        self._active_effects: dict[
            str, DurationEffect
        ] = {}  # effect_uuid -> DurationEffect
        self._authority_sources: dict[TemporalAuthority, Callable[[], float]] = {}
        self._current_authority: TemporalAuthority = self.config.default_authority

        logger.info(
            "temporal_authority_manager_initialized",
            default_authority=self.config.default_authority.value,
            authority_priority=[a.value for a in self.config.authority_priority],
        )

    def register_time_source(
        self,
        authority: TemporalAuthority,
        time_fn: Callable[[], float],
    ) -> None:
        """Register a time source for a specific authority.

        Args:
            authority: The temporal authority type
            time_fn: Function that returns current time for this authority
        """
        self._authority_sources[authority] = time_fn
        logger.debug(
            "time_source_registered",
            authority=authority.value,
        )

    def set_current_authority(self, authority: TemporalAuthority) -> None:
        """Set the current temporal authority.

        Args:
            authority: The temporal authority to use

        Raises:
            ValueError: If authority is not registered
        """
        if authority not in self._authority_sources:
            raise ValueError(
                f"Authority {authority.value} not registered. "
                f"Available: {[a.value for a in self._authority_sources.keys()]}"
            )

        old_authority = self._current_authority
        self._current_authority = authority

        logger.info(
            "temporal_authority_changed",
            old_authority=old_authority.value,
            new_authority=authority.value,
        )

    def resolve_authority(
        self,
        preferred_authority: TemporalAuthority | None = None,
    ) -> TemporalAuthority:
        """Resolve which temporal authority to use based on priority.

        Args:
            preferred_authority: Preferred authority (if available)

        Returns:
            The resolved temporal authority
        """
        # If preferred authority is available, use it
        if preferred_authority and preferred_authority in self._authority_sources:
            return preferred_authority

        # Otherwise, use priority order
        for authority in self.config.authority_priority:
            if authority in self._authority_sources:
                return authority

        # Fallback to default
        return self.config.default_authority

    def get_current_time(self, authority: TemporalAuthority | None = None) -> float:
        """Get current time from the specified or current authority.

        Args:
            authority: Temporal authority to query (uses current if None)

        Returns:
            Current time from the authority

        Raises:
            ValueError: If authority is not registered
        """
        auth = authority or self._current_authority

        if auth not in self._authority_sources:
            # Fallback to wall time
            logger.warning(
                "authority_not_registered_fallback_to_wall_time",
                requested_authority=auth.value,
            )
            return time.time()

        return self._authority_sources[auth]()

    def register_duration_effect(
        self,
        effect: Effect,
        authority: TemporalAuthority | None = None,
    ) -> DurationEffect | None:
        """Register an effect with duration for tracking.

        Args:
            effect: Effect to register
            authority: Temporal authority for this effect (uses current if None)

        Returns:
            DurationEffect if effect has duration, None otherwise

        Raises:
            ValueError: If duration exceeds maximum allowed
        """
        duration_ms = effect.get("duration_ms")
        if duration_ms is None:
            return None

        # Validate duration
        if duration_ms <= 0:
            raise ValueError(f"Effect duration must be positive, got {duration_ms}")

        if duration_ms > self.config.max_effect_duration_ms:
            raise ValueError(
                f"Effect duration {duration_ms}ms exceeds maximum "
                f"{self.config.max_effect_duration_ms}ms"
            )

        # Resolve authority
        auth = authority or self.resolve_authority()

        # Calculate start and end times
        apply_at = effect.get("apply_at")
        if apply_at is not None:
            start_time = apply_at
        else:
            start_time = self.get_current_time(auth)

        end_time = start_time + (duration_ms / 1000.0)  # Convert ms to seconds

        # Create duration effect
        duration_effect = DurationEffect(
            effect=effect,
            start_time=start_time,
            end_time=end_time,
            authority=auth,
        )

        # Check for conflicts if not allowing overlaps
        if not self.config.allow_overlapping_effects:
            conflicts = self.find_conflicts(duration_effect)
            if conflicts:
                logger.warning(
                    "duration_effect_conflicts_detected",
                    effect_uuid=effect["uuid"],
                    conflict_count=len(conflicts),
                    conflicts=[c.effect["uuid"] for c in conflicts],
                )
                # Resolve conflicts (for now, just log)
                self._resolve_conflicts(duration_effect, conflicts)

        # Register the effect
        self._active_effects[effect["uuid"]] = duration_effect

        logger.debug(
            "duration_effect_registered",
            effect_uuid=effect["uuid"],
            effect_kind=effect["kind"],
            duration_ms=duration_ms,
            start_time=start_time,
            end_time=end_time,
            authority=auth.value,
        )

        return duration_effect

    def find_conflicts(self, duration_effect: DurationEffect) -> list[DurationEffect]:
        """Find conflicting duration effects.

        Args:
            duration_effect: Effect to check for conflicts

        Returns:
            List of conflicting duration effects
        """
        conflicts = []

        for existing_effect in self._active_effects.values():
            # Check if effects overlap in time and affect same entity
            if duration_effect.overlaps_with(
                existing_effect
            ) and duration_effect.affects_same_entity(existing_effect):
                conflicts.append(existing_effect)

        return conflicts

    def _resolve_conflicts(
        self,
        new_effect: DurationEffect,
        conflicts: list[DurationEffect],
    ) -> None:
        """Resolve conflicts between duration effects.

        Default strategy: Higher priority effects override lower priority ones.

        Args:
            new_effect: New effect being registered
            conflicts: List of conflicting effects
        """
        new_priority = new_effect.effect.get("priority", 0)

        for conflict in conflicts:
            conflict_priority = conflict.effect.get("priority", 0)

            # If new effect has higher priority, cancel the conflicting effect
            if new_priority > conflict_priority:
                logger.info(
                    "conflict_resolved_cancelling_lower_priority",
                    cancelled_effect=conflict.effect["uuid"],
                    cancelled_priority=conflict_priority,
                    new_effect=new_effect.effect["uuid"],
                    new_priority=new_priority,
                )
                self.unregister_duration_effect(conflict.effect["uuid"])
            else:
                logger.info(
                    "conflict_detected_keeping_higher_priority",
                    kept_effect=conflict.effect["uuid"],
                    kept_priority=conflict_priority,
                    rejected_effect=new_effect.effect["uuid"],
                    rejected_priority=new_priority,
                )

    def unregister_duration_effect(self, effect_uuid: str) -> DurationEffect | None:
        """Unregister a duration effect.

        Args:
            effect_uuid: UUID of effect to unregister

        Returns:
            The unregistered DurationEffect, or None if not found
        """
        duration_effect = self._active_effects.pop(effect_uuid, None)

        if duration_effect:
            logger.debug(
                "duration_effect_unregistered",
                effect_uuid=effect_uuid,
            )

        return duration_effect

    def cleanup_expired_effects(self, current_time: float | None = None) -> int:
        """Remove expired duration effects.

        Args:
            current_time: Current time (uses current authority if None)

        Returns:
            Number of effects cleaned up
        """
        if current_time is None:
            current_time = self.get_current_time()

        expired_uuids = [
            uuid
            for uuid, effect in self._active_effects.items()
            if effect.end_time <= current_time
        ]

        for uuid in expired_uuids:
            self.unregister_duration_effect(uuid)

        if expired_uuids:
            logger.debug(
                "expired_effects_cleaned_up",
                count=len(expired_uuids),
                current_time=current_time,
            )

        return len(expired_uuids)

    def get_active_effects(
        self,
        entity_id: str | None = None,
        effect_kind: str | None = None,
    ) -> list[DurationEffect]:
        """Get currently active duration effects.

        Args:
            entity_id: Filter by entity ID (optional)
            effect_kind: Filter by effect kind (optional)

        Returns:
            List of active duration effects matching filters
        """
        effects = list(self._active_effects.values())

        if entity_id:
            effects = [
                e
                for e in effects
                if e.effect.get("payload", {}).get("agent_id") == entity_id
                or e.effect.get("source_id") == entity_id
            ]

        if effect_kind:
            effects = [e for e in effects if e.effect.get("kind") == effect_kind]

        return effects

    def interpolate_move_position(
        self,
        duration_effect: DurationEffect,
        current_time: float | None = None,
    ) -> tuple[float, float, float] | None:
        """Interpolate position for a Move effect with duration.

        Args:
            duration_effect: Move effect with duration
            current_time: Current time (uses current authority if None)

        Returns:
            Interpolated (x, y, z) position, or None if not applicable
        """
        if not self.config.interpolation_enabled:
            return None

        effect = duration_effect.effect
        if effect.get("kind") != "Move":
            return None

        payload = effect.get("payload", {})
        from_pos = payload.get("from")
        to_pos = payload.get("to")

        if not from_pos or not to_pos:
            return None

        # Ensure 3D coordinates
        if len(from_pos) == 2:
            from_pos = [from_pos[0], from_pos[1], 0.0]
        if len(to_pos) == 2:
            to_pos = [to_pos[0], to_pos[1], 0.0]

        if current_time is None:
            current_time = self.get_current_time(duration_effect.authority)

        # Calculate interpolation factor (0.0 to 1.0)
        total_duration = duration_effect.end_time - duration_effect.start_time
        elapsed = current_time - duration_effect.start_time

        if elapsed <= 0:
            # Not started yet
            return tuple(from_pos)
        elif elapsed >= total_duration:
            # Already completed
            return tuple(to_pos)
        else:
            # Interpolate
            t = elapsed / total_duration
            interpolated = tuple(
                from_pos[i] + t * (to_pos[i] - from_pos[i]) for i in range(3)
            )
            return interpolated

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about temporal authority manager.

        Returns:
            Dictionary with statistics
        """
        return {
            "current_authority": self._current_authority.value,
            "registered_authorities": [a.value for a in self._authority_sources.keys()],
            "active_effects_count": len(self._active_effects),
            "config": {
                "default_authority": self.config.default_authority.value,
                "allow_overlapping_effects": self.config.allow_overlapping_effects,
                "max_effect_duration_ms": self.config.max_effect_duration_ms,
                "interpolation_enabled": self.config.interpolation_enabled,
            },
        }
