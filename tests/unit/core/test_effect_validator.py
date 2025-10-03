"""Unit tests for DefaultEffectValidator implementation.

Tests comprehensive validation logic including quota limits, cooldowns,
permissions, and world state constraints.
"""

import time
from typing import Literal
from unittest.mock import MagicMock

import pytest

from gunn.core.orchestrator import DefaultEffectValidator
from gunn.schemas.messages import WorldState
from gunn.schemas.types import Intent
from gunn.utils.errors import ValidationError


class TestDefaultEffectValidator:
    """Test DefaultEffectValidator implementation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = DefaultEffectValidator(
            max_intents_per_minute=10,
            max_tokens_per_minute=1000,
            default_cooldown_seconds=0.1,  # Shorter cooldown for testing
            max_payload_size_bytes=5000,
        )

        # Override intent-specific cooldowns for testing
        self.validator.set_intent_kind_cooldown("Speak", 0.1)
        self.validator.set_intent_kind_cooldown("Move", 0.1)
        self.validator.set_intent_kind_cooldown("Interact", 0.1)
        self.validator.set_intent_kind_cooldown("Custom", 0.1)

        self.world_state = WorldState(
            entities={
                "agent1": {"name": "Alice", "health": 100},
                "agent2": {"name": "Bob", "health": 80},
                "item1": {"type": "sword", "damage": 10},
            },
            spatial_index={
                "agent1": (10.0, 20.0, 0.0),
                "agent2": (15.0, 25.0, 0.0),
                "item1": (12.0, 22.0, 0.0),
            },
            relationships={
                "agent1": ["agent2"],
                "agent2": ["agent1"],
            },
        )

        # Set up basic permissions
        self.validator.set_agent_permissions(
            "agent1",
            {
                "submit_intent",
                "intent:speak",
                "intent:move",
                "intent:interact",
                "intent:custom",
            },
        )
        self.validator.set_agent_permissions(
            "agent2", {"submit_intent", "intent:speak", "intent:move"}
        )

    def create_valid_intent(
        self,
        agent_id: str = "agent1",
        kind: Literal["Speak", "Move", "Interact", "Custom"] = "Speak",
        payload: dict | None = None,
        req_id: str = "test_req_1",
    ) -> Intent:
        """Create a valid intent for testing."""
        if payload is None:
            payload = {"message": "Hello world"}
        return Intent(
            agent_id=agent_id,
            kind=kind,
            payload=payload,
            context_seq=0,
            req_id=req_id,
            priority=0,
            schema_version="1.0.0",
        )

    def test_validate_intent_success(self) -> None:
        """Test successful intent validation."""
        intent = self.create_valid_intent()

        result = self.validator.validate_intent(intent, self.world_state)

        assert result is True

    def test_validate_structure_missing_fields(self) -> None:
        """Test validation fails for missing required fields."""
        intent = Intent(
            agent_id="",  # Missing agent_id
            kind="Speak",
            payload={"message": "test"},
            context_seq=0,
            req_id="",  # Missing req_id
            priority=0,
            schema_version="1.0.0",
        )

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert "missing_required_field: agent_id" in failures
        assert "missing_required_field: req_id" in failures

    def test_validate_structure_invalid_agent_id(self) -> None:
        """Test validation fails for invalid agent_id format."""
        intent = self.create_valid_intent(agent_id="invalid@agent#id")

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert "invalid_agent_id_format" in failures

    def test_validate_structure_invalid_priority(self) -> None:
        """Test validation fails for invalid priority range."""
        intent = self.create_valid_intent()
        intent["priority"] = 150  # Out of range

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert "invalid_priority_range" in failures

    def test_validate_permissions_missing_submit_intent(self) -> None:
        """Test validation fails when agent lacks submit_intent permission."""
        # Create agent without submit_intent permission
        self.validator.set_agent_permissions("agent3", {"intent:speak"})
        intent = self.create_valid_intent(agent_id="agent3")

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert "missing_permission: submit_intent" in failures

    def test_validate_permissions_missing_intent_specific(self) -> None:
        """Test validation fails when agent lacks intent-specific permission."""
        intent = self.create_valid_intent(agent_id="agent2", kind="Interact")

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert "missing_permission: intent:interact" in failures

    def test_validate_intent_kind_invalid(self) -> None:
        """Test validation fails for invalid intent kind."""
        intent = self.create_valid_intent(kind="InvalidKind")

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert any("invalid_intent_kind" in failure for failure in failures)

    def test_validate_payload_size_too_large(self) -> None:
        """Test validation fails for oversized payload."""
        large_payload = {"data": "x" * 10000}  # Exceeds 5000 byte limit
        intent = self.create_valid_intent(payload=large_payload)

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert any("payload_too_large" in failure for failure in failures)

    def test_validate_payload_not_serializable(self) -> None:
        """Test validation fails for non-serializable payload."""
        # Create payload with non-serializable object
        intent = self.create_valid_intent(payload={"func": lambda x: x})

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert "payload_not_serializable" in failures

    def test_validate_quota_intent_limit_exceeded(self) -> None:
        """Test validation fails when intent quota is exceeded."""
        # Mock time to avoid cooldown issues
        original_time = time.time
        mock_time = MagicMock(return_value=1000.0)
        time.time = mock_time

        try:
            # Submit intents up to the limit
            for i in range(10):  # max_intents_per_minute = 10
                mock_time.return_value += 0.2  # Advance time to avoid cooldown
                test_intent = self.create_valid_intent(req_id=f"req_{i}")
                self.validator.validate_intent(test_intent, self.world_state)

            # Next intent should fail
            mock_time.return_value += 0.2
            over_limit_intent = self.create_valid_intent(req_id="over_limit")
            with pytest.raises(ValidationError) as exc_info:
                self.validator.validate_intent(over_limit_intent, self.world_state)

            failures = exc_info.value.validation_failures
            assert any("intent_quota_exceeded" in failure for failure in failures)
        finally:
            time.time = original_time

    def test_validate_quota_token_limit_exceeded(self) -> None:
        """Test validation fails when token quota is exceeded."""
        # Create intent with large payload to exceed token limit but not message length
        large_message = "x" * 800  # Should be ~200 tokens, under message limit
        intent = self.create_valid_intent(payload={"message": large_message})

        # Mock time to avoid cooldown issues
        original_time = time.time
        mock_time = MagicMock(return_value=1000.0)
        time.time = mock_time

        try:
            # First intent should succeed
            self.validator.validate_intent(intent, self.world_state)

            # Submit more intents to build up token usage until we hit the limit
            token_exceeded = False
            for i in range(10):  # Try up to 10 more intents
                mock_time.return_value += 0.2
                intent_i = self.create_valid_intent(
                    payload={"message": large_message}, req_id=f"req_{i}"
                )
                try:
                    self.validator.validate_intent(intent_i, self.world_state)
                except ValidationError as e:
                    # Check if this is the token quota error we expect
                    if any(
                        "token_quota_exceeded" in failure
                        for failure in e.validation_failures
                    ):
                        token_exceeded = True
                        break
                    else:
                        # Re-raise if it's a different validation error
                        raise

            # Assert that we did hit the token quota limit
            assert token_exceeded, "Expected token quota to be exceeded"
        finally:
            time.time = original_time

    def test_validate_cooldown_active(self) -> None:
        """Test validation fails when cooldown is active."""
        intent = self.create_valid_intent()

        # First intent should succeed
        self.validator.validate_intent(intent, self.world_state)

        # Immediate second intent should fail due to cooldown
        intent2 = self.create_valid_intent(req_id="req_2")
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent2, self.world_state)

        failures = exc_info.value.validation_failures
        assert any("cooldown_active" in failure for failure in failures)

    def test_validate_cooldown_expired(self) -> None:
        """Test validation succeeds when cooldown has expired."""
        intent = self.create_valid_intent()

        # First intent should succeed
        self.validator.validate_intent(intent, self.world_state)

        # Mock time to simulate cooldown expiration
        original_time = time.time
        time.time = MagicMock(return_value=original_time() + 2.0)

        try:
            # Second intent should succeed after cooldown
            intent2 = self.create_valid_intent(req_id="req_2")
            result = self.validator.validate_intent(intent2, self.world_state)
            assert result is True
        finally:
            time.time = original_time

    def test_validate_world_state_agent_not_in_world(self) -> None:
        """Test validation fails when agent is not in world state."""
        intent = self.create_valid_intent(agent_id="nonexistent_agent")

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert "agent_not_in_world: nonexistent_agent" in failures

    def test_validate_move_constraints_invalid_target(self) -> None:
        """Test validation fails for invalid move target."""
        intent = self.create_valid_intent(
            kind="Move",
            payload={"to": "invalid_position"},  # Should be [x, y, z]
        )

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert "invalid_target_position_format" in failures

    def test_validate_move_constraints_distance_too_large(self) -> None:
        """Test validation fails for move distance too large."""
        intent = self.create_valid_intent(
            kind="Move",
            payload={"to": [1000.0, 1000.0, 0.0]},  # Very far from current position
        )

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert any("move_distance_too_large" in failure for failure in failures)

    def test_validate_move_constraints_position_occupied(self) -> None:
        """Test validation fails when target position is occupied."""
        # Try to move to agent2's position
        agent2_pos = self.world_state.spatial_index["agent2"]
        intent = self.create_valid_intent(kind="Move", payload={"to": list(agent2_pos)})

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert any("target_position_occupied" in failure for failure in failures)

    def test_validate_interact_constraints_missing_target(self) -> None:
        """Test validation fails for interaction without target."""
        intent = self.create_valid_intent(
            kind="Interact",
            payload={"type": "examine"},  # Missing target
        )

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert "missing_interaction_target" in failures

    def test_validate_interact_constraints_target_not_found(self) -> None:
        """Test validation fails for interaction with nonexistent target."""
        intent = self.create_valid_intent(
            kind="Interact", payload={"target": "nonexistent_entity", "type": "examine"}
        )

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert "interaction_target_not_found: nonexistent_entity" in failures

    def test_validate_interact_constraints_target_too_far(self) -> None:
        """Test validation fails for interaction with distant target."""
        # Add a distant entity
        self.world_state.entities["distant_item"] = {"type": "treasure"}
        self.world_state.spatial_index["distant_item"] = (1000.0, 1000.0, 0.0)

        intent = self.create_valid_intent(
            kind="Interact", payload={"target": "distant_item", "type": "examine"}
        )

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert any("interaction_target_too_far" in failure for failure in failures)

    def test_validate_speak_constraints_missing_message(self) -> None:
        """Test validation fails for speak intent without message."""
        intent = self.create_valid_intent(
            kind="Speak",
            payload={},  # Missing message
        )

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert "missing_or_invalid_message" in failures

    def test_validate_speak_constraints_message_too_long(self) -> None:
        """Test validation fails for overly long message."""
        long_message = "x" * 1500  # Exceeds 1000 character limit
        intent = self.create_valid_intent(
            kind="Speak", payload={"message": long_message}
        )

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert any("message_too_long" in failure for failure in failures)

    def test_validate_speak_constraints_prohibited_content(self) -> None:
        """Test validation fails for prohibited content."""
        intent = self.create_valid_intent(
            kind="Speak", payload={"message": "This is spam content"}
        )

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert any("prohibited_content: spam" in failure for failure in failures)

    def test_permission_management(self) -> None:
        """Test permission management methods."""
        agent_id = "test_agent"

        # Set initial permissions
        self.validator.set_agent_permissions(agent_id, {"perm1", "perm2"})
        assert self.validator._agent_permissions[agent_id] == {"perm1", "perm2"}

        # Add permission
        self.validator.add_agent_permission(agent_id, "perm3")
        assert "perm3" in self.validator._agent_permissions[agent_id]

        # Remove permission
        self.validator.remove_agent_permission(agent_id, "perm1")
        assert "perm1" not in self.validator._agent_permissions[agent_id]
        assert "perm2" in self.validator._agent_permissions[agent_id]

    def test_intent_kind_cooldown_configuration(self) -> None:
        """Test intent kind cooldown configuration."""
        # Set custom cooldown
        self.validator.set_intent_kind_cooldown("Custom", 5.0)
        assert self.validator._intent_kind_cooldowns["Custom"] == 5.0

        # Test cooldown is applied
        intent = self.create_valid_intent(kind="Custom")
        self.validator.validate_intent(intent, self.world_state)

        # Immediate second intent should fail with longer cooldown
        intent2 = self.create_valid_intent(kind="Custom", req_id="req_2")
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent2, self.world_state)

        failures = exc_info.value.validation_failures
        assert any(
            "cooldown_active" in failure and "5.0" in failure for failure in failures
        )

    def test_get_agent_quota_status(self) -> None:
        """Test agent quota status reporting."""
        agent_id = "agent1"

        # Mock time to avoid cooldown issues
        original_time = time.time
        mock_time = MagicMock(return_value=1000.0)
        time.time = mock_time

        try:
            # Submit some intents to build up usage
            for i in range(3):
                mock_time.return_value += 0.2  # Advance time to avoid cooldown
                intent = self.create_valid_intent(req_id=f"req_{i}")
                self.validator.validate_intent(intent, self.world_state)

            status = self.validator.get_agent_quota_status(agent_id)

            assert status["agent_id"] == agent_id
            assert status["intents_used"] == 3
            assert status["intents_limit"] == 10
            assert status["intents_remaining"] == 7
            assert status["tokens_used"] > 0
            assert status["tokens_limit"] == 1000
            assert status["last_intent_time"] > 0
            assert "submit_intent" in status["permissions"]
        finally:
            time.time = original_time

    def test_quota_cleanup_old_entries(self) -> None:
        """Test that old quota entries are cleaned up."""
        agent_id = "agent1"

        # Mock time to simulate old entries
        original_time = time.time
        old_time = original_time() - 120.0  # 2 minutes ago
        time.time = MagicMock(return_value=old_time)

        try:
            # Submit intent with old timestamp
            intent = self.create_valid_intent()
            self.validator.validate_intent(intent, self.world_state)
        finally:
            time.time = original_time

        # Check that old entries are cleaned up when checking quota
        status = self.validator.get_agent_quota_status(agent_id)
        assert status["intents_used"] == 0  # Old entries should be cleaned up
        assert status["tokens_used"] == 0

    def test_estimate_token_usage(self) -> None:
        """Test token usage estimation."""
        # Small payload
        small_payload = {"message": "hi"}
        tokens = self.validator._estimate_token_usage(small_payload)
        assert tokens > 0
        assert tokens < 10

        # Large payload
        large_payload = {"message": "x" * 100}
        large_tokens = self.validator._estimate_token_usage(large_payload)
        assert large_tokens > tokens

        # Non-serializable payload
        bad_payload = {"func": lambda x: x}
        default_tokens = self.validator._estimate_token_usage(bad_payload)
        assert default_tokens == 10  # Default estimate

    def test_validation_error_structure(self) -> None:
        """Test that ValidationError contains proper structure."""
        intent = self.create_valid_intent(agent_id="")  # Invalid intent

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        error = exc_info.value
        assert error.intent == intent
        assert isinstance(error.validation_failures, list)
        assert len(error.validation_failures) > 0
        assert error.recovery_action.value == "abort"

    def test_multiple_validation_failures(self) -> None:
        """Test that multiple validation failures are collected."""
        intent = Intent(
            agent_id="",  # Missing
            kind="InvalidKind",  # Invalid
            payload={"message": "x" * 2000},  # Too long
            context_seq=0,
            req_id="",  # Missing
            priority=200,  # Invalid range
            schema_version="1.0.0",
        )

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_intent(intent, self.world_state)

        failures = exc_info.value.validation_failures
        assert len(failures) >= 4  # Should have multiple failures
        assert any("missing_required_field: agent_id" in f for f in failures)
        assert any("missing_required_field: req_id" in f for f in failures)
        assert any("invalid_intent_kind" in f for f in failures)
        assert any("invalid_priority_range" in f for f in failures)

    def test_successful_validation_records_usage(self) -> None:
        """Test that successful validation records usage for tracking."""
        agent_id = "agent1"
        intent = self.create_valid_intent()

        # Check initial state
        initial_status = self.validator.get_agent_quota_status(agent_id)
        assert initial_status["intents_used"] == 0

        # Validate intent
        self.validator.validate_intent(intent, self.world_state)

        # Check that usage was recorded
        final_status = self.validator.get_agent_quota_status(agent_id)
        assert final_status["intents_used"] == 1
        assert final_status["tokens_used"] > 0
        assert final_status["last_intent_time"] > initial_status["last_intent_time"]
