"""Unit tests for temporal authority and duration management.

Tests cover:
- Temporal authority resolution with configurable priority
- Duration effect registration and tracking
- Conflict detection and resolution for overlapping effects
- Move action interpolation with start/end timestamps
- Temporal consistency and authority conflicts
"""

import time
from unittest.mock import Mock

import pytest

from gunn.schemas.types import Effect
from gunn.utils.temporal import (
    DurationEffect,
    TemporalAuthority,
    TemporalAuthorityManager,
    TemporalConfig,
)


@pytest.fixture
def temporal_config() -> TemporalConfig:
    """Create default temporal configuration."""
    return TemporalConfig(
        default_authority=TemporalAuthority.SIM_TIME,
        allow_overlapping_effects=True,
        max_effect_duration_ms=60000.0,
        interpolation_enabled=True,
    )


@pytest.fixture
def temporal_manager(temporal_config: TemporalConfig) -> TemporalAuthorityManager:
    """Create temporal authority manager with test configuration."""
    manager = TemporalAuthorityManager(temporal_config)

    # Register time sources
    manager.register_time_source(TemporalAuthority.WALL_TIME, time.time)
    manager.register_time_source(TemporalAuthority.SIM_TIME, lambda: 100.0)
    manager.register_time_source(TemporalAuthority.EXTERNAL, lambda: 200.0)

    return manager


@pytest.fixture
def sample_effect() -> Effect:
    """Create a sample effect for testing."""
    return {
        "uuid": "test-uuid-123",
        "kind": "Move",
        "payload": {
            "agent_id": "alice",
            "from": [0.0, 0.0, 0.0],
            "to": [10.0, 10.0, 0.0],
        },
        "global_seq": 1,
        "sim_time": 100.0,
        "source_id": "alice",
        "schema_version": "1.0.0",
        "req_id": "req_1",
        "duration_ms": 1000.0,
        "apply_at": None,
    }


class TestTemporalConfig:
    """Test temporal configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TemporalConfig()

        assert config.default_authority == TemporalAuthority.SIM_TIME
        assert config.allow_overlapping_effects is True
        assert config.max_effect_duration_ms == 60000.0
        assert config.interpolation_enabled is True
        assert config.authority_priority == [
            TemporalAuthority.EXTERNAL,
            TemporalAuthority.SIM_TIME,
            TemporalAuthority.WALL_TIME,
        ]

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TemporalConfig(
            default_authority=TemporalAuthority.EXTERNAL,
            allow_overlapping_effects=False,
            max_effect_duration_ms=30000.0,
            interpolation_enabled=False,
            authority_priority=[
                TemporalAuthority.SIM_TIME,
                TemporalAuthority.WALL_TIME,
            ],
        )

        assert config.default_authority == TemporalAuthority.EXTERNAL
        assert config.allow_overlapping_effects is False
        assert config.max_effect_duration_ms == 30000.0
        assert config.interpolation_enabled is False
        assert len(config.authority_priority) == 2


class TestDurationEffect:
    """Test duration effect data class."""

    def test_overlaps_with(self, sample_effect: Effect) -> None:
        """Test overlap detection between duration effects."""
        effect1 = DurationEffect(
            effect=sample_effect,
            start_time=100.0,
            end_time=101.0,
            authority=TemporalAuthority.SIM_TIME,
        )

        # Overlapping effect
        effect2 = DurationEffect(
            effect=sample_effect,
            start_time=100.5,
            end_time=101.5,
            authority=TemporalAuthority.SIM_TIME,
        )

        # Non-overlapping effect (before)
        effect3 = DurationEffect(
            effect=sample_effect,
            start_time=99.0,
            end_time=100.0,
            authority=TemporalAuthority.SIM_TIME,
        )

        # Non-overlapping effect (after)
        effect4 = DurationEffect(
            effect=sample_effect,
            start_time=101.0,
            end_time=102.0,
            authority=TemporalAuthority.SIM_TIME,
        )

        assert effect1.overlaps_with(effect2)
        assert effect2.overlaps_with(effect1)
        assert not effect1.overlaps_with(effect3)
        assert not effect1.overlaps_with(effect4)

    def test_affects_same_entity(self, sample_effect: Effect) -> None:
        """Test entity affection detection."""
        effect1 = DurationEffect(
            effect=sample_effect,
            start_time=100.0,
            end_time=101.0,
            authority=TemporalAuthority.SIM_TIME,
        )

        # Same entity
        effect2_same = sample_effect.copy()
        effect2 = DurationEffect(
            effect=effect2_same,
            start_time=100.5,
            end_time=101.5,
            authority=TemporalAuthority.SIM_TIME,
        )

        # Different entity
        effect3_different = sample_effect.copy()
        effect3_different["payload"] = {"agent_id": "bob", "to": [5.0, 5.0, 0.0]}
        effect3 = DurationEffect(
            effect=effect3_different,
            start_time=100.5,
            end_time=101.5,
            authority=TemporalAuthority.SIM_TIME,
        )

        assert effect1.affects_same_entity(effect2)
        assert not effect1.affects_same_entity(effect3)


class TestTemporalAuthorityManager:
    """Test temporal authority manager."""

    def test_initialization(self, temporal_manager: TemporalAuthorityManager) -> None:
        """Test manager initialization."""
        assert temporal_manager.config.default_authority == TemporalAuthority.SIM_TIME
        assert len(temporal_manager._active_effects) == 0

    def test_register_time_source(self) -> None:
        """Test registering time sources."""
        manager = TemporalAuthorityManager()

        mock_time_fn = Mock(return_value=123.45)
        manager.register_time_source(TemporalAuthority.EXTERNAL, mock_time_fn)

        assert TemporalAuthority.EXTERNAL in manager._authority_sources

    def test_set_current_authority(
        self, temporal_manager: TemporalAuthorityManager
    ) -> None:
        """Test setting current authority."""
        temporal_manager.set_current_authority(TemporalAuthority.EXTERNAL)
        assert temporal_manager._current_authority == TemporalAuthority.EXTERNAL

        # Test invalid authority
        with pytest.raises(ValueError, match="Authority .* not registered"):
            unregistered_manager = TemporalAuthorityManager()
            unregistered_manager.set_current_authority(TemporalAuthority.EXTERNAL)

    def test_resolve_authority_with_preference(
        self, temporal_manager: TemporalAuthorityManager
    ) -> None:
        """Test authority resolution with preferred authority."""
        # Preferred authority available
        resolved = temporal_manager.resolve_authority(TemporalAuthority.EXTERNAL)
        assert resolved == TemporalAuthority.EXTERNAL

        # Preferred authority not available
        manager = TemporalAuthorityManager()
        manager.register_time_source(TemporalAuthority.SIM_TIME, lambda: 100.0)
        resolved = manager.resolve_authority(TemporalAuthority.EXTERNAL)
        assert resolved == TemporalAuthority.SIM_TIME

    def test_resolve_authority_priority_order(
        self, temporal_manager: TemporalAuthorityManager
    ) -> None:
        """Test authority resolution follows priority order."""
        # All authorities available - should use EXTERNAL (highest priority)
        resolved = temporal_manager.resolve_authority()
        assert resolved == TemporalAuthority.EXTERNAL

        # Remove EXTERNAL - should use SIM_TIME
        temporal_manager._authority_sources.pop(TemporalAuthority.EXTERNAL)
        resolved = temporal_manager.resolve_authority()
        assert resolved == TemporalAuthority.SIM_TIME

        # Remove SIM_TIME - should use WALL_TIME
        temporal_manager._authority_sources.pop(TemporalAuthority.SIM_TIME)
        resolved = temporal_manager.resolve_authority()
        assert resolved == TemporalAuthority.WALL_TIME

    def test_get_current_time(self, temporal_manager: TemporalAuthorityManager) -> None:
        """Test getting current time from authority."""
        # Default authority (SIM_TIME)
        current_time = temporal_manager.get_current_time()
        assert current_time == 100.0

        # Specific authority
        external_time = temporal_manager.get_current_time(TemporalAuthority.EXTERNAL)
        assert external_time == 200.0

        # Unregistered authority (should fallback to wall time)
        manager = TemporalAuthorityManager()
        fallback_time = manager.get_current_time(TemporalAuthority.EXTERNAL)
        assert isinstance(fallback_time, float)

    def test_register_duration_effect(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test registering duration effect."""
        duration_effect = temporal_manager.register_duration_effect(sample_effect)

        assert duration_effect is not None
        assert duration_effect.effect == sample_effect
        assert (
            duration_effect.start_time == 200.0
        )  # Current EXTERNAL time (highest priority)
        assert duration_effect.end_time == 201.0  # start + 1 second
        assert (
            duration_effect.authority == TemporalAuthority.EXTERNAL
        )  # Highest priority

        # Check it's tracked
        assert sample_effect["uuid"] in temporal_manager._active_effects

    def test_register_duration_effect_with_apply_at(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test registering duration effect with delayed application."""
        sample_effect["apply_at"] = 150.0

        duration_effect = temporal_manager.register_duration_effect(sample_effect)

        assert duration_effect is not None
        assert duration_effect.start_time == 150.0
        assert duration_effect.end_time == 151.0

    def test_register_effect_without_duration(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test registering effect without duration returns None."""
        sample_effect["duration_ms"] = None

        duration_effect = temporal_manager.register_duration_effect(sample_effect)

        assert duration_effect is None
        assert sample_effect["uuid"] not in temporal_manager._active_effects

    def test_register_effect_invalid_duration(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test registering effect with invalid duration raises error."""
        # Zero duration
        sample_effect["duration_ms"] = 0.0
        with pytest.raises(ValueError, match="duration must be positive"):
            temporal_manager.register_duration_effect(sample_effect)

        # Negative duration
        sample_effect["duration_ms"] = -100.0
        with pytest.raises(ValueError, match="duration must be positive"):
            temporal_manager.register_duration_effect(sample_effect)

        # Exceeds maximum
        sample_effect["duration_ms"] = 100000.0
        with pytest.raises(ValueError, match="exceeds maximum"):
            temporal_manager.register_duration_effect(sample_effect)

    def test_find_conflicts(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test finding conflicting duration effects."""
        # Register first effect (starts at 200.0, ends at 201.0 with EXTERNAL authority)
        effect1 = temporal_manager.register_duration_effect(sample_effect)

        # Create overlapping effect for same entity
        effect2_dict = sample_effect.copy()
        effect2_dict["uuid"] = "test-uuid-456"
        effect2_dict["apply_at"] = 200.5  # Overlaps with effect1 (200.0-201.0)

        duration_effect2 = DurationEffect(
            effect=effect2_dict,
            start_time=200.5,
            end_time=201.5,
            authority=TemporalAuthority.SIM_TIME,
        )

        conflicts = temporal_manager.find_conflicts(duration_effect2)

        assert len(conflicts) == 1
        assert conflicts[0].effect["uuid"] == sample_effect["uuid"]

    def test_conflict_resolution_by_priority(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test conflict resolution based on priority."""
        # Disable overlapping to trigger conflict resolution
        temporal_manager.config.allow_overlapping_effects = False

        # Register low priority effect (starts at 200.0, ends at 201.0)
        sample_effect["priority"] = 0
        effect1 = temporal_manager.register_duration_effect(sample_effect)
        assert effect1 is not None

        # Register high priority overlapping effect
        effect2_dict = sample_effect.copy()
        effect2_dict["uuid"] = "test-uuid-456"
        effect2_dict["priority"] = 10
        effect2_dict["apply_at"] = 200.5  # Overlaps with effect1

        effect2 = temporal_manager.register_duration_effect(effect2_dict)

        # Low priority effect should be cancelled
        assert sample_effect["uuid"] not in temporal_manager._active_effects
        assert effect2_dict["uuid"] in temporal_manager._active_effects

    def test_unregister_duration_effect(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test unregistering duration effect."""
        temporal_manager.register_duration_effect(sample_effect)

        unregistered = temporal_manager.unregister_duration_effect(
            sample_effect["uuid"]
        )

        assert unregistered is not None
        assert unregistered.effect["uuid"] == sample_effect["uuid"]
        assert sample_effect["uuid"] not in temporal_manager._active_effects

        # Unregistering again returns None
        unregistered_again = temporal_manager.unregister_duration_effect(
            sample_effect["uuid"]
        )
        assert unregistered_again is None

    def test_cleanup_expired_effects(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test cleaning up expired effects."""
        # Register effect that starts at 200.0 and ends at 201.0 (EXTERNAL authority)
        temporal_manager.register_duration_effect(sample_effect)

        # Cleanup at time 200.5 - effect still active
        cleaned = temporal_manager.cleanup_expired_effects(200.5)
        assert cleaned == 0
        assert sample_effect["uuid"] in temporal_manager._active_effects

        # Cleanup at time 201.5 - effect expired
        cleaned = temporal_manager.cleanup_expired_effects(201.5)
        assert cleaned == 1
        assert sample_effect["uuid"] not in temporal_manager._active_effects

    def test_get_active_effects(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test getting active effects with filters."""
        # Register alice's move effect
        temporal_manager.register_duration_effect(sample_effect)

        # Create bob's move effect
        effect2: Effect = {
            "uuid": "test-uuid-456",
            "kind": "Move",
            "payload": {
                "agent_id": "bob",
                "from": [0.0, 0.0, 0.0],
                "to": [5.0, 5.0, 0.0],
            },
            "global_seq": 2,
            "sim_time": 100.0,
            "source_id": "bob",
            "schema_version": "1.0.0",
            "req_id": "req_2",
            "duration_ms": 1000.0,
            "apply_at": None,
        }
        temporal_manager.register_duration_effect(effect2)

        # Create alice's speak effect
        effect3: Effect = {
            "uuid": "test-uuid-789",
            "kind": "Speak",
            "payload": {"agent_id": "alice", "text": "Hello"},
            "global_seq": 3,
            "sim_time": 100.0,
            "source_id": "alice",
            "schema_version": "1.0.0",
            "req_id": "req_3",
            "duration_ms": 1000.0,
            "apply_at": None,
        }
        temporal_manager.register_duration_effect(effect3)

        # Get all active effects
        all_effects = temporal_manager.get_active_effects()
        assert len(all_effects) == 3

        # Filter by entity
        alice_effects = temporal_manager.get_active_effects(entity_id="alice")
        assert len(alice_effects) == 2

        # Filter by kind
        move_effects = temporal_manager.get_active_effects(effect_kind="Move")
        assert len(move_effects) == 2

        # Filter by both
        alice_move = temporal_manager.get_active_effects(
            entity_id="alice", effect_kind="Move"
        )
        assert len(alice_move) == 1

    def test_interpolate_move_position(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test position interpolation for Move effects."""
        duration_effect = temporal_manager.register_duration_effect(sample_effect)

        # At start time (200.0 - EXTERNAL authority)
        pos_start = temporal_manager.interpolate_move_position(duration_effect, 200.0)
        assert pos_start == (0.0, 0.0, 0.0)

        # At midpoint (200.5)
        pos_mid = temporal_manager.interpolate_move_position(duration_effect, 200.5)
        assert pos_mid == (5.0, 5.0, 0.0)

        # At end time (201.0)
        pos_end = temporal_manager.interpolate_move_position(duration_effect, 201.0)
        assert pos_end == (10.0, 10.0, 0.0)

        # Before start
        pos_before = temporal_manager.interpolate_move_position(duration_effect, 199.0)
        assert pos_before == (0.0, 0.0, 0.0)

        # After end
        pos_after = temporal_manager.interpolate_move_position(duration_effect, 202.0)
        assert pos_after == (10.0, 10.0, 0.0)

    def test_interpolate_move_position_2d(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test position interpolation with 2D coordinates."""
        sample_effect["payload"]["from"] = [0.0, 0.0]
        sample_effect["payload"]["to"] = [10.0, 10.0]

        duration_effect = temporal_manager.register_duration_effect(sample_effect)

        # Should convert to 3D and interpolate (at midpoint 200.5)
        pos_mid = temporal_manager.interpolate_move_position(duration_effect, 200.5)
        assert pos_mid == (5.0, 5.0, 0.0)

    def test_interpolate_non_move_effect(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test interpolation returns None for non-Move effects."""
        sample_effect["kind"] = "Speak"
        duration_effect = temporal_manager.register_duration_effect(sample_effect)

        pos = temporal_manager.interpolate_move_position(duration_effect, 100.5)
        assert pos is None

    def test_interpolation_disabled(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test interpolation returns None when disabled."""
        temporal_manager.config.interpolation_enabled = False
        duration_effect = temporal_manager.register_duration_effect(sample_effect)

        pos = temporal_manager.interpolate_move_position(duration_effect, 100.5)
        assert pos is None

    def test_get_stats(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test getting manager statistics."""
        temporal_manager.register_duration_effect(sample_effect)

        stats = temporal_manager.get_stats()

        # Current authority is still SIM_TIME (default), not changed by registration
        assert stats["current_authority"] == "sim_time"
        assert len(stats["registered_authorities"]) == 3
        assert stats["active_effects_count"] == 1
        assert stats["config"]["default_authority"] == "sim_time"
        assert stats["config"]["allow_overlapping_effects"] is True
        assert stats["config"]["max_effect_duration_ms"] == 60000.0
        assert stats["config"]["interpolation_enabled"] is True


class TestTemporalConsistency:
    """Test temporal consistency across different scenarios."""

    def test_authority_switching_consistency(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test consistency when switching authorities."""
        # Register effect with SIM_TIME
        temporal_manager.set_current_authority(TemporalAuthority.SIM_TIME)
        effect1 = temporal_manager.register_duration_effect(sample_effect)

        # Switch to EXTERNAL
        temporal_manager.set_current_authority(TemporalAuthority.EXTERNAL)

        # Register another effect
        effect2_dict = sample_effect.copy()
        effect2_dict["uuid"] = "test-uuid-456"
        effect2 = temporal_manager.register_duration_effect(effect2_dict)

        # Both effects should be tracked
        assert len(temporal_manager._active_effects) == 2
        assert (
            effect1.authority == TemporalAuthority.EXTERNAL
        )  # Resolved to highest priority
        assert effect2.authority == TemporalAuthority.EXTERNAL

    def test_multiple_overlapping_effects_allowed(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test multiple overlapping effects when allowed."""
        temporal_manager.config.allow_overlapping_effects = True

        # Register multiple overlapping effects
        effect1 = temporal_manager.register_duration_effect(sample_effect)

        effect2_dict = sample_effect.copy()
        effect2_dict["uuid"] = "test-uuid-456"
        effect2_dict["apply_at"] = 100.5
        effect2 = temporal_manager.register_duration_effect(effect2_dict)

        # Both should be active
        assert len(temporal_manager._active_effects) == 2

    def test_deterministic_conflict_resolution(
        self,
        temporal_manager: TemporalAuthorityManager,
        sample_effect: Effect,
    ) -> None:
        """Test conflict resolution is deterministic."""
        temporal_manager.config.allow_overlapping_effects = False

        # Register effects with same priority
        sample_effect["priority"] = 5
        effect1 = temporal_manager.register_duration_effect(sample_effect)

        effect2_dict = sample_effect.copy()
        effect2_dict["uuid"] = "test-uuid-456"
        effect2_dict["priority"] = 5
        effect2_dict["apply_at"] = 100.5

        # Second effect should not cancel first (same priority)
        effect2 = temporal_manager.register_duration_effect(effect2_dict)

        # First effect should still be active (not cancelled by equal priority)
        assert sample_effect["uuid"] in temporal_manager._active_effects
