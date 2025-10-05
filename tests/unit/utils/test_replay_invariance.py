"""Tests for replay invariance validation."""

import pytest

from gunn.core.event_log import EventLog
from gunn.schemas.messages import WorldState
from gunn.schemas.types import Effect
from gunn.utils.replay_invariance import (
    ConsistencyViolation,
    ReplayInvarianceValidator,
)


def create_effect(
    uuid: str,
    kind: str,
    payload: dict,
    sim_time: float = 0.0,
    source_id: str = "system",
    global_seq: int = 0,
) -> Effect:
    """Helper to create Effect with all required fields."""
    return Effect(
        uuid=uuid,
        kind=kind,
        payload=payload,
        global_seq=global_seq,
        sim_time=sim_time,
        source_id=source_id,
        schema_version="1.0.0",
        req_id=None,
        duration_ms=None,
        apply_at=None,
    )


class TestReplayInvarianceValidator:
    """Test replay invariance validation functionality."""

    @pytest.fixture
    def validator(self) -> ReplayInvarianceValidator:
        """Create validator instance."""
        return ReplayInvarianceValidator("test_world")

    @pytest.fixture
    async def sample_log(self) -> EventLog:
        """Create sample event log."""
        log = EventLog("test_world")

        # Add agent join
        await log.append(
            create_effect(
                uuid="effect-1",
                kind="AgentJoined",
                payload={"agent_id": "agent_1", "position": {"x": 10.0, "y": 20.0}},
                sim_time=0.0,
            )
        )

        # Add move
        await log.append(
            create_effect(
                uuid="effect-2",
                kind="Move",
                payload={"agent_id": "agent_1", "to_pos": {"x": 15.0, "y": 25.0}},
                sim_time=1.0,
                source_id="agent_1",
            )
        )

        # Add message
        await log.append(
            create_effect(
                uuid="effect-3",
                kind="MessageSent",
                payload={"sender": "agent_1", "text": "Hello"},
                sim_time=2.0,
                source_id="agent_1",
            )
        )

        return log

    @pytest.mark.asyncio
    async def test_full_replay_basic(self, validator: ReplayInvarianceValidator):
        """Test basic full replay functionality."""
        log = EventLog("test")

        # Add simple effect
        await log.append(
            create_effect(
                uuid="e1",
                kind="AgentJoined",
                payload={"agent_id": "a1", "position": {"x": 1.0, "y": 2.0}},
            )
        )

        entries = log.get_all_entries()
        state = await validator._full_replay(entries)

        assert "a1" in state.entities
        assert state.entities["a1"]["type"] == "agent"
        assert "a1" in state.spatial_index
        assert state.spatial_index["a1"] == (1.0, 2.0, 0.0)

    @pytest.mark.asyncio
    async def test_validate_consistent_states(
        self, validator: ReplayInvarianceValidator, sample_log: EventLog
    ):
        """Test validation with consistent incremental and replay states."""
        # Build incremental state by replaying
        entries = sample_log.get_all_entries()
        incremental_state = await validator._full_replay(entries)

        # Validate - should be consistent
        report = await validator.validate_replay_invariance(
            sample_log, incremental_state
        )

        assert report.valid
        assert len(report.violations) == 0
        assert report.incremental_hash == report.full_replay_hash
        assert report.entries_checked == 3

    @pytest.mark.asyncio
    async def test_detect_entity_mismatch(
        self, validator: ReplayInvarianceValidator, sample_log: EventLog
    ):
        """Test detection of entity mismatches."""
        # Build correct state
        entries = sample_log.get_all_entries()
        correct_state = await validator._full_replay(entries)

        # Create incorrect incremental state (missing entity)
        incorrect_state = WorldState(
            entities={},  # Missing agent_1
            relationships={},
            spatial_index={},
            metadata=correct_state.metadata,
        )

        # Validate - should detect mismatch
        report = await validator.validate_replay_invariance(sample_log, incorrect_state)

        assert not report.valid
        assert len(report.violations) > 0
        assert any("agent_1" in v.field_path for v in report.violations)

    @pytest.mark.asyncio
    async def test_detect_position_mismatch(
        self, validator: ReplayInvarianceValidator, sample_log: EventLog
    ):
        """Test detection of position mismatches."""
        # Build correct state
        entries = sample_log.get_all_entries()
        correct_state = await validator._full_replay(entries)

        # Create incorrect state with wrong position
        incorrect_state = WorldState(
            entities=dict(correct_state.entities),
            relationships=dict(correct_state.relationships),
            spatial_index={"agent_1": (999.0, 999.0, 0.0)},  # Wrong position
            metadata=dict(correct_state.metadata),
        )

        # Validate
        report = await validator.validate_replay_invariance(sample_log, incorrect_state)

        assert not report.valid
        assert any("spatial_index" in v.field_path for v in report.violations)

    @pytest.mark.asyncio
    async def test_recovery_options_generation(
        self, validator: ReplayInvarianceValidator
    ):
        """Test generation of recovery options."""
        # Test with no violations
        options = validator._generate_recovery_options([], True)
        assert len(options) == 1
        assert "No recovery needed" in options[0]

        # Test with entity violations
        violations = [
            ConsistencyViolation(
                global_seq=1,
                field_path="entities.agent_1",
                incremental_value=None,
                full_replay_value={"type": "agent"},
                description="Missing entity",
            )
        ]
        options = validator._generate_recovery_options(violations, False)
        assert any("REBUILD_FROM_LOG" in opt for opt in options)
        assert any("SYNC_ENTITIES" in opt for opt in options)

        # Test with spatial violations
        violations = [
            ConsistencyViolation(
                global_seq=1,
                field_path="spatial_index.agent_1",
                incremental_value=(0, 0, 0),
                full_replay_value=(1, 1, 0),
                description="Position mismatch",
            )
        ]
        options = validator._generate_recovery_options(violations, False)
        assert any("REBUILD_SPATIAL_INDEX" in opt for opt in options)

        # Test with many violations
        many_violations = [
            ConsistencyViolation(
                global_seq=i,
                field_path=f"entities.agent_{i}",
                incremental_value=None,
                full_replay_value={},
                description="Violation",
            )
            for i in range(15)
        ]
        options = validator._generate_recovery_options(many_violations, False)
        assert any("FULL_RESET" in opt for opt in options)

    @pytest.mark.asyncio
    async def test_state_hashing(self, validator: ReplayInvarianceValidator):
        """Test state hashing for comparison."""
        state1 = WorldState(
            entities={"a1": {"type": "agent"}},
            relationships={},
            spatial_index={"a1": (1.0, 2.0, 3.0)},
            metadata={},
        )

        state2 = WorldState(
            entities={"a1": {"type": "agent"}},
            relationships={},
            spatial_index={"a1": (1.0, 2.0, 3.0)},
            metadata={},
        )

        state3 = WorldState(
            entities={"a1": {"type": "agent"}},
            relationships={},
            spatial_index={"a1": (1.0, 2.0, 4.0)},  # Different z
            metadata={},
        )

        hash1 = validator._hash_state(state1)
        hash2 = validator._hash_state(state2)
        hash3 = validator._hash_state(state3)

        assert hash1 == hash2  # Identical states
        assert hash1 != hash3  # Different states

    @pytest.mark.asyncio
    async def test_snapshot_creation(self, validator: ReplayInvarianceValidator):
        """Test state snapshot creation."""
        state = WorldState(
            entities={"a1": {"type": "agent"}},
            relationships={},
            spatial_index={"a1": (1.0, 2.0, 3.0)},
            metadata={},
        )

        snapshot = await validator.create_snapshot(state, global_seq=5)

        assert snapshot.global_seq == 5
        assert snapshot.world_state.entities == state.entities
        assert len(snapshot.state_hash) == 64  # SHA-256 hex
        assert snapshot.timestamp > 0

        # Verify snapshot is stored
        snapshots = validator.get_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0].global_seq == 5

    @pytest.mark.asyncio
    async def test_snapshot_management(self, validator: ReplayInvarianceValidator):
        """Test snapshot storage and clearing."""
        state = WorldState()

        # Create multiple snapshots
        await validator.create_snapshot(state, 1)
        await validator.create_snapshot(state, 2)
        await validator.create_snapshot(state, 3)

        snapshots = validator.get_snapshots()
        assert len(snapshots) == 3

        # Clear snapshots
        validator.clear_snapshots()
        snapshots = validator.get_snapshots()
        assert len(snapshots) == 0

    @pytest.mark.asyncio
    async def test_periodic_validation_interval(
        self, validator: ReplayInvarianceValidator
    ):
        """Test periodic validation respects interval."""
        log = EventLog("test")

        # Add a few entries
        for i in range(5):
            await log.append(
                create_effect(
                    uuid=f"e{i}",
                    kind="AgentJoined",
                    payload={"agent_id": f"a{i}"},
                    sim_time=float(i),
                )
            )

        state = WorldState()

        # First validation should run (no previous validation)
        report1 = await validator.periodic_validation(log, state, interval_entries=100)
        assert report1 is not None

        # Second validation should be skipped (not enough new entries)
        report2 = await validator.periodic_validation(log, state, interval_entries=100)
        assert report2 is None

        # Add more entries to trigger validation
        for i in range(5, 105):
            await log.append(
                create_effect(
                    uuid=f"e{i}",
                    kind="AgentJoined",
                    payload={"agent_id": f"a{i}"},
                    sim_time=float(i),
                )
            )

        # Now validation should run again
        report3 = await validator.periodic_validation(log, state, interval_entries=100)
        assert report3 is not None

    @pytest.mark.asyncio
    async def test_apply_effect_agent_lifecycle(
        self, validator: ReplayInvarianceValidator
    ):
        """Test effect application for agent lifecycle."""
        state = WorldState()

        # Agent joins
        join_effect = create_effect(
            uuid="e1",
            kind="AgentJoined",
            payload={"agent_id": "a1", "position": {"x": 10.0, "y": 20.0, "z": 30.0}},
            global_seq=1,
        )
        state = validator._apply_effect(state, join_effect)

        assert "a1" in state.entities
        assert state.spatial_index["a1"] == (10.0, 20.0, 30.0)

        # Agent leaves
        leave_effect = create_effect(
            uuid="e2",
            kind="AgentLeft",
            payload={"agent_id": "a1", "reason": "disconnected"},
            global_seq=2,
            sim_time=1.0,
        )
        state = validator._apply_effect(state, leave_effect)

        assert "a1" not in state.entities
        assert "a1" not in state.spatial_index

    @pytest.mark.asyncio
    async def test_apply_effect_move_2d_coordinates(
        self, validator: ReplayInvarianceValidator
    ):
        """Test move effect with 2D coordinates."""
        state = WorldState(
            entities={"a1": {"type": "agent"}}, spatial_index={"a1": (0.0, 0.0, 0.0)}
        )

        # Move with 2D list
        move_effect = create_effect(
            uuid="e1",
            kind="Move",
            payload={"agent_id": "a1", "to": [5.0, 10.0]},
            global_seq=1,
            sim_time=1.0,
            source_id="a1",
        )
        state = validator._apply_effect(state, move_effect)

        assert state.spatial_index["a1"] == (5.0, 10.0, 0.0)

    @pytest.mark.asyncio
    async def test_apply_effect_move_3d_coordinates(
        self, validator: ReplayInvarianceValidator
    ):
        """Test move effect with 3D coordinates."""
        state = WorldState(
            entities={"a1": {"type": "agent"}}, spatial_index={"a1": (0.0, 0.0, 0.0)}
        )

        # Move with 3D list
        move_effect = create_effect(
            uuid="e1",
            kind="Move",
            payload={"agent_id": "a1", "to": [5.0, 10.0, 15.0]},
            global_seq=1,
            sim_time=1.0,
            source_id="a1",
        )
        state = validator._apply_effect(state, move_effect)

        assert state.spatial_index["a1"] == (5.0, 10.0, 15.0)

    @pytest.mark.asyncio
    async def test_apply_effect_messages(self, validator: ReplayInvarianceValidator):
        """Test message effect application."""
        state = WorldState()

        message_effect = create_effect(
            uuid="e1",
            kind="MessageSent",
            payload={"sender": "a1", "text": "Hello world"},
            global_seq=1,
            sim_time=1.0,
            source_id="a1",
        )
        state = validator._apply_effect(state, message_effect)

        assert "messages" in state.metadata
        assert len(state.metadata["messages"]) == 1
        assert state.metadata["messages"][0]["sender"] == "a1"
        assert state.metadata["messages"][0]["text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_apply_effect_interactions(
        self, validator: ReplayInvarianceValidator
    ):
        """Test interaction effect application."""
        state = WorldState()

        interact_effect = create_effect(
            uuid="e1",
            kind="Interact",
            payload={"agent_id": "a1", "target": "a2", "action": "greet"},
            global_seq=1,
            sim_time=1.0,
            source_id="a1",
        )
        state = validator._apply_effect(state, interact_effect)

        assert "interactions" in state.metadata
        assert len(state.metadata["interactions"]) == 1
        assert state.metadata["interactions"][0]["agent_id"] == "a1"
        assert state.metadata["interactions"][0]["target"] == "a2"

    @pytest.mark.asyncio
    async def test_validation_with_range(
        self, validator: ReplayInvarianceValidator, sample_log: EventLog
    ):
        """Test validation with specific sequence range."""
        # Build state up to seq 2
        entries = [e for e in sample_log.get_all_entries() if e.global_seq <= 2]
        state = await validator._full_replay(entries)

        # Validate only up to seq 2
        report = await validator.validate_replay_invariance(
            sample_log, state, from_seq=0, to_seq=2
        )

        assert report.valid
        assert report.entries_checked == 2

    @pytest.mark.asyncio
    async def test_compare_states_comprehensive(
        self, validator: ReplayInvarianceValidator
    ):
        """Test comprehensive state comparison."""
        state1 = WorldState(
            entities={"a1": {"type": "agent"}, "a2": {"type": "agent"}},
            relationships={"a1": ["a2"]},
            spatial_index={"a1": (1.0, 2.0, 3.0), "a2": (4.0, 5.0, 6.0)},
            metadata={"key1": "value1"},
        )

        # Identical state
        state2 = WorldState(
            entities={"a1": {"type": "agent"}, "a2": {"type": "agent"}},
            relationships={"a1": ["a2"]},
            spatial_index={"a1": (1.0, 2.0, 3.0), "a2": (4.0, 5.0, 6.0)},
            metadata={"key1": "value1"},
        )

        violations = validator._compare_states(state1, state2, 1)
        assert len(violations) == 0

        # State with differences
        state3 = WorldState(
            entities={"a1": {"type": "agent"}},  # Missing a2
            relationships={"a1": ["a2"]},
            spatial_index={"a1": (1.0, 2.0, 3.0)},
            metadata={"key2": "value2"},  # Different metadata
        )

        violations = validator._compare_states(state1, state3, 1)
        assert len(violations) > 0
        assert any("a2" in v.field_path for v in violations)
        assert any("metadata" in v.field_path for v in violations)


class TestEventLogReplayIntegration:
    """Test EventLog integration with replay invariance."""

    @pytest.mark.asyncio
    async def test_validate_integrity_with_replay_check(self):
        """Test EventLog.validate_integrity with replay check."""
        log = EventLog("test")

        # Add some effects
        await log.append(
            create_effect(uuid="e1", kind="AgentJoined", payload={"agent_id": "a1"})
        )

        # Build correct state
        validator = ReplayInvarianceValidator("test")
        entries = log.get_all_entries()
        correct_state = await validator._full_replay(entries)

        # Validate with replay check - should pass (use async version)
        result = await log.validate_integrity_async(
            include_replay_check=True, current_state=correct_state
        )

        assert result["valid"]
        assert result.get("replay_valid") is True

    @pytest.mark.asyncio
    async def test_validate_integrity_detects_replay_mismatch(self):
        """Test that validate_integrity detects replay mismatches."""
        log = EventLog("test")

        await log.append(
            create_effect(uuid="e1", kind="AgentJoined", payload={"agent_id": "a1"})
        )

        # Create incorrect state
        incorrect_state = WorldState()  # Empty state

        # Validate with replay check - should fail (use async version)
        result = await log.validate_integrity_async(
            include_replay_check=True, current_state=incorrect_state
        )

        assert not result["valid"]
        assert result.get("replay_valid") is False
        assert result.get("replay_violations", 0) > 0

    def test_validate_integrity_without_replay_check(self):
        """Test that validate_integrity works without replay check."""
        log = EventLog("test")

        # Validate without replay check
        result = log.validate_integrity(include_replay_check=False)

        assert result["valid"]
        assert "replay_valid" not in result
