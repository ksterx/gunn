"""Integration tests for replay invariance validation in real scenarios."""

import pytest

from gunn.core.event_log import EventLog
from gunn.schemas.messages import WorldState
from gunn.schemas.types import Effect
from gunn.utils.replay_invariance import ReplayInvarianceValidator


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


class TestReplayInvarianceIntegration:
    """Integration tests for replay invariance in realistic scenarios."""

    @pytest.mark.asyncio
    async def test_multi_agent_simulation_replay_consistency(self):
        """Test replay consistency in a multi-agent simulation scenario."""
        log = EventLog("multi_agent_sim")
        validator = ReplayInvarianceValidator("multi_agent_sim")

        # Simulate a multi-agent scenario
        # Agent 1 joins
        await log.append(
            create_effect(
                uuid="e1",
                kind="AgentJoined",
                payload={"agent_id": "agent_1", "position": {"x": 0.0, "y": 0.0}},
                sim_time=0.0,
            )
        )

        # Agent 2 joins
        await log.append(
            create_effect(
                uuid="e2",
                kind="AgentJoined",
                payload={"agent_id": "agent_2", "position": {"x": 10.0, "y": 10.0}},
                sim_time=1.0,
            )
        )

        # Agent 1 moves
        await log.append(
            create_effect(
                uuid="e3",
                kind="Move",
                payload={"agent_id": "agent_1", "to": [5.0, 5.0]},
                sim_time=2.0,
                source_id="agent_1",
            )
        )

        # Agent 1 sends message
        await log.append(
            create_effect(
                uuid="e4",
                kind="MessageSent",
                payload={"sender": "agent_1", "text": "Hello agent_2!"},
                sim_time=3.0,
                source_id="agent_1",
            )
        )

        # Agent 2 moves
        await log.append(
            create_effect(
                uuid="e5",
                kind="Move",
                payload={"agent_id": "agent_2", "to": [7.0, 7.0]},
                sim_time=4.0,
                source_id="agent_2",
            )
        )

        # Build incremental state by replaying
        entries = log.get_all_entries()
        incremental_state = await validator._full_replay(entries)

        # Validate replay invariance
        report = await validator.validate_replay_invariance(log, incremental_state)

        assert report.valid
        assert len(report.violations) == 0
        assert report.entries_checked == 5
        assert report.incremental_hash == report.full_replay_hash

        # Verify state is correct
        assert "agent_1" in incremental_state.entities
        assert "agent_2" in incremental_state.entities
        assert incremental_state.spatial_index["agent_1"] == (5.0, 5.0, 0.0)
        assert incremental_state.spatial_index["agent_2"] == (7.0, 7.0, 0.0)
        assert "messages" in incremental_state.metadata
        assert len(incremental_state.metadata["messages"]) == 1

    @pytest.mark.asyncio
    async def test_periodic_validation_during_long_simulation(self):
        """Test periodic validation during a long-running simulation."""
        log = EventLog("long_sim")
        validator = ReplayInvarianceValidator("long_sim")

        # Simulate a long-running simulation with periodic validation
        state = WorldState()

        # Add 50 events
        for i in range(50):
            await log.append(
                create_effect(
                    uuid=f"e{i}",
                    kind="AgentJoined",
                    payload={"agent_id": f"agent_{i}"},
                    sim_time=float(i),
                )
            )

            # Update state incrementally
            entries = log.get_entries_since(i)
            for entry in entries:
                state = validator._apply_effect(state, entry.effect)

        # First periodic validation (should run)
        report1 = await validator.periodic_validation(log, state, interval_entries=25)
        assert report1 is not None
        assert report1.valid

        # Add more events
        for i in range(50, 60):
            await log.append(
                create_effect(
                    uuid=f"e{i}",
                    kind="AgentJoined",
                    payload={"agent_id": f"agent_{i}"},
                    sim_time=float(i),
                )
            )

        # Second validation (should be skipped - not enough new entries)
        report2 = await validator.periodic_validation(log, state, interval_entries=25)
        assert report2 is None

        # Add more events to trigger validation
        for i in range(60, 90):
            await log.append(
                create_effect(
                    uuid=f"e{i}",
                    kind="AgentJoined",
                    payload={"agent_id": f"agent_{i}"},
                    sim_time=float(i),
                )
            )

        # Third validation (should run - enough new entries)
        report3 = await validator.periodic_validation(log, state, interval_entries=25)
        assert report3 is not None

    @pytest.mark.asyncio
    async def test_detect_state_corruption(self):
        """Test detection of state corruption through replay invariance."""
        log = EventLog("corruption_test")
        validator = ReplayInvarianceValidator("corruption_test")

        # Build correct state
        await log.append(
            create_effect(
                uuid="e1",
                kind="AgentJoined",
                payload={"agent_id": "agent_1", "position": {"x": 10.0, "y": 20.0}},
            )
        )

        await log.append(
            create_effect(
                uuid="e2",
                kind="Move",
                payload={"agent_id": "agent_1", "to": [30.0, 40.0]},
                sim_time=1.0,
                source_id="agent_1",
            )
        )

        # Build correct incremental state
        entries = log.get_all_entries()
        correct_state = await validator._full_replay(entries)

        # Validate - should pass
        report1 = await validator.validate_replay_invariance(log, correct_state)
        assert report1.valid

        # Simulate corruption by creating incorrect state
        corrupted_state = WorldState(
            entities={"agent_1": {"type": "agent"}},
            spatial_index={"agent_1": (999.0, 999.0, 0.0)},  # Wrong position!
            relationships={},
            metadata={},
        )

        # Validate - should detect corruption
        report2 = await validator.validate_replay_invariance(log, corrupted_state)
        assert not report2.valid
        assert len(report2.violations) > 0
        assert any("spatial_index" in v.field_path for v in report2.violations)

        # Check recovery options
        assert len(report2.recovery_options) > 0
        assert any("REBUILD" in opt for opt in report2.recovery_options)

    @pytest.mark.asyncio
    async def test_snapshot_based_validation(self):
        """Test validation using state snapshots."""
        log = EventLog("snapshot_test")
        validator = ReplayInvarianceValidator("snapshot_test")

        # Build state with snapshots at intervals
        state = WorldState()

        for i in range(10):
            await log.append(
                create_effect(
                    uuid=f"e{i}",
                    kind="AgentJoined",
                    payload={"agent_id": f"agent_{i}"},
                    sim_time=float(i),
                )
            )

            # Update state
            entries = log.get_entries_since(i)
            for entry in entries:
                state = validator._apply_effect(state, entry.effect)

            # Create snapshot every 3 entries
            if i % 3 == 0:
                snapshot = await validator.create_snapshot(state, i + 1)
                assert snapshot.global_seq == i + 1

        # Verify snapshots were created
        snapshots = validator.get_snapshots()
        assert len(snapshots) == 4  # At i=0, 3, 6, 9

        # Validate using final state
        report = await validator.validate_replay_invariance(log, state)
        assert report.valid

    @pytest.mark.asyncio
    async def test_eventlog_async_validation_integration(self):
        """Test EventLog.validate_integrity_async with replay checking."""
        log = EventLog("async_validation_test")

        # Add some effects
        for i in range(5):
            await log.append(
                create_effect(
                    uuid=f"e{i}",
                    kind="AgentJoined",
                    payload={"agent_id": f"agent_{i}"},
                    sim_time=float(i),
                )
            )

        # Build correct state
        validator = ReplayInvarianceValidator("async_validation_test")
        entries = log.get_all_entries()
        correct_state = await validator._full_replay(entries)

        # Validate with replay check using async method
        result = await log.validate_integrity_async(
            include_replay_check=True, current_state=correct_state
        )

        assert result["valid"]
        assert result["replay_valid"] is True
        assert result["replay_violations"] == 0
        assert result["replay_hash_match"] is True
        assert result["total_entries"] == 5

    @pytest.mark.asyncio
    async def test_recovery_from_detected_violations(self):
        """Test recovery options when violations are detected."""
        log = EventLog("recovery_test")
        validator = ReplayInvarianceValidator("recovery_test")

        # Create a scenario with multiple types of violations
        await log.append(
            create_effect(
                uuid="e1",
                kind="AgentJoined",
                payload={"agent_id": "agent_1", "position": {"x": 10.0, "y": 20.0}},
            )
        )

        await log.append(
            create_effect(
                uuid="e2",
                kind="AgentJoined",
                payload={"agent_id": "agent_2", "position": {"x": 30.0, "y": 40.0}},
                sim_time=1.0,
            )
        )

        # Create state with multiple issues
        incorrect_state = WorldState(
            entities={"agent_1": {"type": "agent"}},  # Missing agent_2
            spatial_index={"agent_1": (999.0, 999.0, 0.0)},  # Wrong position
            relationships={},
            metadata={"extra_key": "should_not_be_here"},  # Extra metadata
        )

        # Validate
        report = await validator.validate_replay_invariance(log, incorrect_state)

        assert not report.valid
        assert len(report.violations) > 0

        # Check that recovery options are comprehensive
        recovery_options = report.recovery_options
        assert len(recovery_options) > 0

        # Should suggest entity sync
        assert any(
            "SYNC_ENTITIES" in opt or "REBUILD" in opt for opt in recovery_options
        )

        # Should suggest spatial index rebuild
        assert any("SPATIAL" in opt or "REBUILD" in opt for opt in recovery_options)

        # Should suggest metadata sync
        assert any("METADATA" in opt or "REBUILD" in opt for opt in recovery_options)
