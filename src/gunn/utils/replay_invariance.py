"""Replay invariance validation for storage consistency.

This module provides tools to validate that incremental state updates
produce identical results to full replay from the event log, ensuring
storage consistency and replay correctness.
"""

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any

from gunn.core.event_log import EventLog, EventLogEntry
from gunn.schemas.messages import WorldState
from gunn.schemas.types import Effect
from gunn.utils.hashing import canonical_json
from gunn.utils.telemetry import PerformanceTimer, get_logger


@dataclass
class StateSnapshot:
    """Snapshot of world state at a specific sequence number."""

    global_seq: int
    world_state: WorldState
    state_hash: str
    timestamp: float


@dataclass
class ConsistencyViolation:
    """Details of a consistency violation between incremental and full replay."""

    global_seq: int
    field_path: str
    incremental_value: Any
    full_replay_value: Any
    description: str


@dataclass
class ReplayInvarianceReport:
    """Report of replay invariance validation results."""

    valid: bool
    incremental_hash: str
    full_replay_hash: str
    violations: list[ConsistencyViolation]
    entries_checked: int
    duration_seconds: float
    recovery_options: list[str]


class ReplayInvarianceValidator:
    """Validator for comparing incremental vs full replay results.

    This validator ensures that incremental state updates produce identical
    results to full replay from the event log, which is critical for:
    - Storage consistency verification
    - Detecting state corruption
    - Validating effect application logic
    - Ensuring replay correctness

    The validator can:
    - Compare incremental state against full replay
    - Detect field-level differences
    - Generate detailed diagnostics
    - Suggest recovery options
    - Run periodic validation during long-running simulations
    """

    def __init__(self, world_id: str = "default"):
        """Initialize replay invariance validator.

        Args:
            world_id: Identifier for the world being validated
        """
        self.world_id = world_id
        self.logger = get_logger("gunn.replay_invariance", world_id=world_id)
        self._snapshots: list[StateSnapshot] = []
        self._last_validation_seq: int = 0

    async def validate_replay_invariance(
        self,
        event_log: EventLog,
        incremental_state: WorldState,
        from_seq: int = 0,
        to_seq: int | None = None,
    ) -> ReplayInvarianceReport:
        """Validate that incremental state matches full replay.

        Compares the current incremental state against a full replay
        from the event log to detect any inconsistencies.

        Args:
            event_log: Event log to replay from
            incremental_state: Current incremental world state
            from_seq: Starting sequence for replay (default: 0)
            to_seq: Ending sequence for replay (default: latest)

        Returns:
            ReplayInvarianceReport with validation results
        """
        with PerformanceTimer("replay_invariance_validation", record_metrics=True):
            start_time = asyncio.get_event_loop().time()

            self.logger.info(
                "Starting replay invariance validation",
                from_seq=from_seq,
                to_seq=to_seq,
            )

            # Determine replay range
            if to_seq is None:
                to_seq = event_log.get_latest_seq()

            # Get entries to replay
            if from_seq == 0:
                entries = [
                    e for e in event_log.get_all_entries() if e.global_seq <= to_seq
                ]
            else:
                entries = [
                    e
                    for e in event_log.get_entries_since(from_seq - 1)
                    if e.global_seq <= to_seq
                ]

            # Perform full replay
            replayed_state = await self._full_replay(entries)

            # Compare states
            violations = self._compare_states(incremental_state, replayed_state, to_seq)

            # Calculate state hashes
            incremental_hash = self._hash_state(incremental_state)
            full_replay_hash = self._hash_state(replayed_state)

            # Determine validity
            is_valid = len(violations) == 0 and incremental_hash == full_replay_hash

            # Generate recovery options
            recovery_options = self._generate_recovery_options(violations, is_valid)

            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            report = ReplayInvarianceReport(
                valid=is_valid,
                incremental_hash=incremental_hash,
                full_replay_hash=full_replay_hash,
                violations=violations,
                entries_checked=len(entries),
                duration_seconds=duration,
                recovery_options=recovery_options,
            )

            if is_valid:
                self.logger.info(
                    "Replay invariance validation passed",
                    entries_checked=len(entries),
                    duration_seconds=duration,
                )
            else:
                self.logger.warning(
                    "Replay invariance validation failed",
                    violations_count=len(violations),
                    entries_checked=len(entries),
                    duration_seconds=duration,
                )

            self._last_validation_seq = to_seq

            return report

    async def _full_replay(self, entries: list[EventLogEntry]) -> WorldState:
        """Perform full replay from entries to reconstruct world state.

        Args:
            entries: List of event log entries to replay

        Returns:
            Reconstructed world state
        """
        # Start with empty world state
        state = WorldState()

        # Apply each effect in order
        for entry in entries:
            effect = entry.effect
            state = self._apply_effect(state, effect)

        return state

    def _apply_effect(self, state: WorldState, effect: Effect) -> WorldState:
        """Apply an effect to world state.

        This is a simplified implementation that handles common effect types.
        In production, this should match the Orchestrator's effect application logic.

        Args:
            state: Current world state
            effect: Effect to apply

        Returns:
            Updated world state
        """
        # Create a copy to avoid mutating the input
        new_state = WorldState(
            entities=dict(state.entities),
            relationships=dict(state.relationships),
            spatial_index=dict(state.spatial_index),
            metadata=dict(state.metadata),
        )

        effect_kind = effect["kind"]
        payload = effect["payload"]

        if effect_kind == "AgentJoined":
            agent_id = payload.get("agent_id")
            if agent_id:
                new_state.entities[agent_id] = {
                    "type": "agent",
                    "position": payload.get("position", {}),
                    "status": "active",
                }
                if "position" in payload:
                    pos = payload["position"]
                    if isinstance(pos, dict):
                        new_state.spatial_index[agent_id] = (
                            pos.get("x", 0.0),
                            pos.get("y", 0.0),
                            pos.get("z", 0.0),
                        )

        elif effect_kind == "AgentLeft":
            agent_id = payload.get("agent_id")
            if agent_id:
                new_state.entities.pop(agent_id, None)
                new_state.spatial_index.pop(agent_id, None)

        elif effect_kind == "Move":
            agent_id = payload.get("agent_id")
            if agent_id and agent_id in new_state.entities:
                to_pos = payload.get("to_pos") or payload.get("to")
                if to_pos:
                    if isinstance(to_pos, dict):
                        new_state.spatial_index[agent_id] = (
                            to_pos.get("x", 0.0),
                            to_pos.get("y", 0.0),
                            to_pos.get("z", 0.0),
                        )
                    elif isinstance(to_pos, (list, tuple)):
                        if len(to_pos) == 2:
                            new_state.spatial_index[agent_id] = (
                                float(to_pos[0]),
                                float(to_pos[1]),
                                0.0,
                            )
                        elif len(to_pos) == 3:
                            new_state.spatial_index[agent_id] = (
                                float(to_pos[0]),
                                float(to_pos[1]),
                                float(to_pos[2]),
                            )

        elif effect_kind == "MessageSent" or effect_kind == "Speak":
            # Store message in metadata
            if "messages" not in new_state.metadata:
                new_state.metadata["messages"] = []
            new_state.metadata["messages"].append(
                {
                    "sender": payload.get("sender") or payload.get("agent_id"),
                    "text": payload.get("text"),
                    "timestamp": effect.get("sim_time", 0.0),
                }
            )

        elif effect_kind == "Interact":
            # Store interaction in metadata
            if "interactions" not in new_state.metadata:
                new_state.metadata["interactions"] = []
            new_state.metadata["interactions"].append(
                {
                    "agent_id": payload.get("agent_id"),
                    "target": payload.get("target"),
                    "action": payload.get("action"),
                    "timestamp": effect.get("sim_time", 0.0),
                }
            )

        return new_state

    def _compare_states(
        self, incremental: WorldState, replayed: WorldState, seq: int
    ) -> list[ConsistencyViolation]:
        """Compare two world states and identify violations.

        Args:
            incremental: Incrementally updated state
            replayed: State from full replay
            seq: Sequence number being validated

        Returns:
            List of consistency violations found
        """
        violations: list[ConsistencyViolation] = []

        # Compare entities
        inc_entities = set(incremental.entities.keys())
        rep_entities = set(replayed.entities.keys())

        # Check for missing entities
        missing_in_incremental = rep_entities - inc_entities
        for entity_id in missing_in_incremental:
            violations.append(
                ConsistencyViolation(
                    global_seq=seq,
                    field_path=f"entities.{entity_id}",
                    incremental_value=None,
                    full_replay_value=replayed.entities[entity_id],
                    description=f"Entity {entity_id} missing in incremental state",
                )
            )

        # Check for extra entities
        extra_in_incremental = inc_entities - rep_entities
        for entity_id in extra_in_incremental:
            violations.append(
                ConsistencyViolation(
                    global_seq=seq,
                    field_path=f"entities.{entity_id}",
                    incremental_value=incremental.entities[entity_id],
                    full_replay_value=None,
                    description=f"Entity {entity_id} exists in incremental but not in replay",
                )
            )

        # Compare common entities
        for entity_id in inc_entities & rep_entities:
            inc_entity = incremental.entities[entity_id]
            rep_entity = replayed.entities[entity_id]
            if inc_entity != rep_entity:
                violations.append(
                    ConsistencyViolation(
                        global_seq=seq,
                        field_path=f"entities.{entity_id}",
                        incremental_value=inc_entity,
                        full_replay_value=rep_entity,
                        description=f"Entity {entity_id} differs between states",
                    )
                )

        # Compare spatial index
        inc_spatial = set(incremental.spatial_index.keys())
        rep_spatial = set(replayed.spatial_index.keys())

        for entity_id in inc_spatial & rep_spatial:
            inc_pos = incremental.spatial_index[entity_id]
            rep_pos = replayed.spatial_index[entity_id]
            if inc_pos != rep_pos:
                violations.append(
                    ConsistencyViolation(
                        global_seq=seq,
                        field_path=f"spatial_index.{entity_id}",
                        incremental_value=inc_pos,
                        full_replay_value=rep_pos,
                        description=f"Position for {entity_id} differs",
                    )
                )

        # Compare metadata (simplified - just check keys)
        inc_meta_keys = set(incremental.metadata.keys())
        rep_meta_keys = set(replayed.metadata.keys())

        if inc_meta_keys != rep_meta_keys:
            violations.append(
                ConsistencyViolation(
                    global_seq=seq,
                    field_path="metadata",
                    incremental_value=list(inc_meta_keys),
                    full_replay_value=list(rep_meta_keys),
                    description="Metadata keys differ between states",
                )
            )

        return violations

    def _hash_state(self, state: WorldState) -> str:
        """Calculate hash of world state for comparison.

        Args:
            state: World state to hash

        Returns:
            SHA-256 hash of the state
        """
        # Convert state to canonical JSON for hashing
        state_dict = {
            "entities": state.entities,
            "relationships": state.relationships,
            "spatial_index": {k: list(v) for k, v in state.spatial_index.items()},
            "metadata": state.metadata,
        }

        state_bytes = canonical_json(state_dict)
        return hashlib.sha256(state_bytes).hexdigest()

    def _generate_recovery_options(
        self, violations: list[ConsistencyViolation], is_valid: bool
    ) -> list[str]:
        """Generate recovery options based on validation results.

        Args:
            violations: List of consistency violations
            is_valid: Whether validation passed

        Returns:
            List of suggested recovery actions
        """
        if is_valid:
            return ["No recovery needed - state is consistent"]

        options = []

        # Analyze violation patterns
        entity_violations = [v for v in violations if "entities" in v.field_path]
        spatial_violations = [v for v in violations if "spatial_index" in v.field_path]
        metadata_violations = [v for v in violations if "metadata" in v.field_path]

        if entity_violations:
            options.append(
                "REBUILD_FROM_LOG: Rebuild world state from full event log replay"
            )
            options.append(
                "SYNC_ENTITIES: Synchronize entity state from replay to incremental"
            )

        if spatial_violations:
            options.append(
                "REBUILD_SPATIAL_INDEX: Rebuild spatial index from entity positions"
            )

        if metadata_violations:
            options.append("SYNC_METADATA: Synchronize metadata from replay")

        if len(violations) > 10:
            options.append(
                "FULL_RESET: Too many violations - recommend full state reset from log"
            )
        else:
            options.append(
                "PATCH_DIFFERENCES: Apply specific patches to fix identified violations"
            )

        options.append(
            "CHECKPOINT_AND_CONTINUE: Create checkpoint and continue with replay state"
        )

        return options

    async def create_snapshot(
        self, state: WorldState, global_seq: int
    ) -> StateSnapshot:
        """Create a snapshot of current world state.

        Args:
            state: World state to snapshot
            global_seq: Current sequence number

        Returns:
            StateSnapshot with state and metadata
        """
        snapshot = StateSnapshot(
            global_seq=global_seq,
            world_state=WorldState(
                entities=dict(state.entities),
                relationships=dict(state.relationships),
                spatial_index=dict(state.spatial_index),
                metadata=dict(state.metadata),
            ),
            state_hash=self._hash_state(state),
            timestamp=asyncio.get_event_loop().time(),
        )

        self._snapshots.append(snapshot)

        self.logger.debug(
            "Created state snapshot",
            global_seq=global_seq,
            state_hash=snapshot.state_hash[:8],
        )

        return snapshot

    def get_snapshots(self) -> list[StateSnapshot]:
        """Get all stored snapshots.

        Returns:
            List of state snapshots
        """
        return self._snapshots.copy()

    def clear_snapshots(self) -> None:
        """Clear all stored snapshots."""
        self._snapshots.clear()
        self.logger.debug("Cleared all state snapshots")

    async def periodic_validation(
        self,
        event_log: EventLog,
        incremental_state: WorldState,
        interval_entries: int = 100,
    ) -> ReplayInvarianceReport | None:
        """Perform periodic validation during long-running simulations.

        Args:
            event_log: Event log to validate against
            incremental_state: Current incremental state
            interval_entries: Number of entries between validations

        Returns:
            ReplayInvarianceReport if validation was performed, None if skipped
        """
        latest_seq = event_log.get_latest_seq()

        # Check if we should validate
        # Always run on first validation (when _last_validation_seq is 0)
        if (
            self._last_validation_seq > 0
            and latest_seq - self._last_validation_seq < interval_entries
        ):
            return None

        self.logger.info(
            "Performing periodic replay invariance validation",
            latest_seq=latest_seq,
            last_validation_seq=self._last_validation_seq,
        )

        report = await self.validate_replay_invariance(
            event_log, incremental_state, from_seq=0, to_seq=latest_seq
        )

        return report
