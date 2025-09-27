"""Unit tests for data model validation and serialization."""

import asyncio
import json
import time

import pytest
from pydantic import ValidationError

from .messages import EventLogEntry, View, WorldState
from .types import CancelToken, Effect, EffectDraft, Intent, ObservationDelta


class TestWorldState:
    """Test WorldState Pydantic model."""

    def test_default_initialization(self) -> None:
        """Test WorldState initializes with proper defaults."""
        world = WorldState()

        assert world.entities == {}
        assert world.relationships == {}
        assert world.spatial_index == {}
        assert world.metadata == {}

    def test_spatial_index_validation(self) -> None:
        """Test spatial index validation with proper 3D coordinates."""
        # Valid coordinates as tuple
        world = WorldState(spatial_index={"agent1": (1.0, 2.0, 3.0)})
        assert world.spatial_index["agent1"] == (1.0, 2.0, 3.0)

        # Valid coordinates as list (auto-converted to tuple)
        world = WorldState(
            spatial_index={"agent1": [1.0, 2.0, 3.0]}  # type: ignore[dict-item]
        )
        assert world.spatial_index["agent1"] == (1.0, 2.0, 3.0)

        # Invalid coordinates - wrong length (list) - Pydantic validates tuple length
        with pytest.raises(ValidationError, match="Field required"):
            WorldState(spatial_index={"agent1": [1.0, 2.0]})  # type: ignore[dict-item]

        # Invalid coordinates - wrong length (tuple) - Pydantic validates tuple length
        with pytest.raises(ValidationError, match="Field required"):
            WorldState(spatial_index={"agent1": (1.0, 2.0)})  # type: ignore[dict-item]

        # Invalid coordinates - non-numeric (Pydantic validates float type)
        with pytest.raises(ValidationError, match="Input should be a valid number"):
            WorldState(spatial_index={"agent1": ("x", "y", "z")})  # type: ignore[dict-item]

    def test_with_data(self) -> None:
        """Test WorldState with actual data."""
        world = WorldState(
            entities={"agent1": {"name": "Alice", "health": 100}},
            relationships={"agent1": ["agent2"]},
            spatial_index={"agent1": (10.0, 20.0, 0.0)},
            metadata={"world_name": "test_world"},
        )

        assert world.entities["agent1"]["name"] == "Alice"
        assert world.relationships["agent1"] == ["agent2"]
        assert world.spatial_index["agent1"] == (10.0, 20.0, 0.0)
        assert world.metadata["world_name"] == "test_world"

    def test_json_serialization(self) -> None:
        """Test WorldState JSON serialization."""
        world = WorldState(
            entities={"agent1": {"name": "Alice"}},
            spatial_index={"agent1": (1.0, 2.0, 3.0)},
        )

        json_str = world.model_dump_json()
        data = json.loads(json_str)

        assert data["entities"]["agent1"]["name"] == "Alice"
        assert data["spatial_index"]["agent1"] == [1.0, 2.0, 3.0]

    def test_validation_assignment(self) -> None:
        """Test that validation occurs on assignment."""
        world = WorldState()
        world.entities = {"test": {"value": 42}}

        assert world.entities["test"]["value"] == 42

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            WorldState(invalid_field="should_fail")  # type: ignore[call-arg]


class TestView:
    """Test View Pydantic model."""

    def test_required_fields(self) -> None:
        """Test View requires agent_id, view_seq, and context_digest."""
        view = View(agent_id="agent1", view_seq=42, context_digest="abc123")

        assert view.agent_id == "agent1"
        assert view.view_seq == 42
        assert view.context_digest == "abc123"
        assert view.visible_entities == {}
        assert view.visible_relationships == {}

    def test_view_seq_validation(self) -> None:
        """Test view_seq validation for non-negative values."""
        # Valid non-negative view_seq
        view = View(agent_id="agent1", view_seq=0, context_digest="abc123")
        assert view.view_seq == 0

        # Invalid negative view_seq
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            View(agent_id="agent1", view_seq=-1, context_digest="abc123")

    def test_agent_id_validation(self) -> None:
        """Test agent_id validation for non-empty strings."""
        # Valid agent_id
        view = View(agent_id="agent1", view_seq=1, context_digest="abc123")
        assert view.agent_id == "agent1"

        # Invalid empty agent_id
        with pytest.raises(
            ValidationError, match="String should have at least 1 character"
        ):
            View(agent_id="", view_seq=1, context_digest="abc123")

    def test_context_digest_validation(self) -> None:
        """Test context_digest validation for non-empty strings."""
        # Valid context_digest
        view = View(agent_id="agent1", view_seq=1, context_digest="sha256:abc123")
        assert view.context_digest == "sha256:abc123"

        # Invalid empty context_digest
        with pytest.raises(
            ValidationError, match="String should have at least 1 character"
        ):
            View(agent_id="agent1", view_seq=1, context_digest="")

    def test_with_visibility_data(self) -> None:
        """Test View with visibility data."""
        view = View(
            agent_id="agent1",
            view_seq=1,
            context_digest="hash123",
            visible_entities={"entity1": {"type": "player"}},
            visible_relationships={"entity1": ["entity2"]},
        )

        assert view.visible_entities["entity1"]["type"] == "player"
        assert view.visible_relationships["entity1"] == ["entity2"]

    def test_json_schema_generation(self) -> None:
        """Test that View can generate JSON schema."""
        schema = View.model_json_schema()

        assert "properties" in schema
        assert "agent_id" in schema["properties"]
        assert "view_seq" in schema["properties"]
        assert "context_digest" in schema["properties"]

    def test_missing_required_field(self) -> None:
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError, match="Field required"):
            View(agent_id="agent1")  # type: ignore[call-arg] # Missing view_seq and context_digest


class TestEventLogEntry:
    """Test EventLogEntry Pydantic model."""

    def test_required_fields(self) -> None:
        """Test EventLogEntry with required fields."""
        effect = {
            "kind": "test",
            "payload": {},
            "source_id": "agent1",
            "schema_version": "1.0.0",
        }
        entry = EventLogEntry(
            global_seq=1, sim_time=100.0, effect=effect, checksum="abc123"
        )

        assert entry.global_seq == 1
        assert entry.sim_time == 100.0
        assert entry.effect == effect
        assert entry.checksum == "abc123"
        assert entry.req_id == ""  # Default value
        assert entry.source_metadata == {}  # Default value
        assert isinstance(entry.wall_time, float)  # Auto-generated

    def test_global_seq_validation(self) -> None:
        """Test global_seq validation for non-negative values."""
        effect = {
            "kind": "test",
            "payload": {},
            "source_id": "agent1",
            "schema_version": "1.0.0",
        }

        # Valid non-negative global_seq
        entry = EventLogEntry(
            global_seq=0, sim_time=100.0, effect=effect, checksum="abc123"
        )
        assert entry.global_seq == 0

        # Invalid negative global_seq
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            EventLogEntry(
                global_seq=-1, sim_time=100.0, effect=effect, checksum="abc123"
            )

    def test_time_validation(self) -> None:
        """Test time validation for non-negative values."""
        effect = {
            "kind": "test",
            "payload": {},
            "source_id": "agent1",
            "schema_version": "1.0.0",
        }

        # Valid non-negative times
        entry = EventLogEntry(
            global_seq=1, sim_time=0.0, wall_time=0.0, effect=effect, checksum="abc123"
        )
        assert entry.sim_time == 0.0
        assert entry.wall_time == 0.0

        # Invalid negative sim_time
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            EventLogEntry(global_seq=1, sim_time=-1.0, effect=effect, checksum="abc123")

        # Invalid negative wall_time
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            EventLogEntry(
                global_seq=1,
                sim_time=1.0,
                wall_time=-1.0,
                effect=effect,
                checksum="abc123",
            )

    def test_effect_structure_validation(self) -> None:
        """Test effect structure validation for required fields."""
        # Valid effect with all required fields
        valid_effect = {
            "kind": "Move",
            "payload": {"x": 10, "y": 20},
            "source_id": "agent1",
            "schema_version": "1.0.0",
        }
        entry = EventLogEntry(
            global_seq=1, sim_time=1.0, effect=valid_effect, checksum="abc123"
        )
        assert entry.effect == valid_effect

        # Invalid effect missing 'kind'
        with pytest.raises(ValidationError, match="Effect must contain 'kind' field"):
            EventLogEntry(
                global_seq=1,
                sim_time=1.0,
                effect={
                    "payload": {},
                    "source_id": "agent1",
                    "schema_version": "1.0.0",
                },
                checksum="abc123",
            )

        # Invalid effect missing 'payload'
        with pytest.raises(
            ValidationError, match="Effect must contain 'payload' field"
        ):
            EventLogEntry(
                global_seq=1,
                sim_time=1.0,
                effect={
                    "kind": "test",
                    "source_id": "agent1",
                    "schema_version": "1.0.0",
                },
                checksum="abc123",
            )

        # Invalid effect missing 'source_id'
        with pytest.raises(
            ValidationError, match="Effect must contain 'source_id' field"
        ):
            EventLogEntry(
                global_seq=1,
                sim_time=1.0,
                effect={"kind": "test", "payload": {}, "schema_version": "1.0.0"},
                checksum="abc123",
            )

        # Invalid effect missing 'schema_version'
        with pytest.raises(
            ValidationError, match="Effect must contain 'schema_version' field"
        ):
            EventLogEntry(
                global_seq=1,
                sim_time=1.0,
                effect={"kind": "test", "payload": {}, "source_id": "agent1"},
                checksum="abc123",
            )

    def test_wall_time_default(self) -> None:
        """Test that wall_time is auto-generated."""
        before = time.time()
        effect = {
            "kind": "test",
            "payload": {},
            "source_id": "agent1",
            "schema_version": "1.0.0",
        }
        entry = EventLogEntry(
            global_seq=1, sim_time=100.0, effect=effect, checksum="abc123"
        )
        after = time.time()

        assert before <= entry.wall_time <= after

    def test_with_optional_fields(self) -> None:
        """Test EventLogEntry with all optional fields."""
        entry = EventLogEntry(
            global_seq=1,
            sim_time=100.0,
            wall_time=123.456,
            effect={
                "kind": "test",
                "payload": {"data": "value"},
                "source_id": "agent1",
                "schema_version": "1.0.0",
            },
            source_metadata={"adapter": "unity"},
            checksum="def456",
            req_id="req_123",
        )

        assert entry.wall_time == 123.456
        assert entry.source_metadata["adapter"] == "unity"
        assert entry.req_id == "req_123"

    def test_repr(self) -> None:
        """Test EventLogEntry string representation."""
        entry = EventLogEntry(
            global_seq=42,
            sim_time=100.5,
            effect={
                "kind": "Move",
                "payload": {},
                "source_id": "agent1",
                "schema_version": "1.0.0",
            },
            checksum="abc123",
        )

        repr_str = repr(entry)
        assert "global_seq=42" in repr_str
        assert "sim_time=100.5" in repr_str
        assert "effect_kind=Move" in repr_str


class TestTypedDictTypes:
    """Test TypedDict type definitions."""

    def test_intent_structure(self) -> None:
        """Test Intent TypedDict structure."""
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello world"},
            "context_seq": 10,
            "req_id": "req_123",
            "agent_id": "agent1",
            "priority": 5,
            "schema_version": "1.0.0",
        }

        assert intent["kind"] == "Speak"
        assert intent["payload"]["text"] == "Hello world"
        assert intent["context_seq"] == 10
        assert intent["priority"] == 5

    def test_effect_draft_structure(self) -> None:
        """Test EffectDraft TypedDict structure."""
        draft: EffectDraft = {
            "kind": "MessageEmitted",
            "payload": {"text": "Hello"},
            "source_id": "agent1",
            "schema_version": "1.0.0",
        }

        assert draft["kind"] == "MessageEmitted"
        assert draft["source_id"] == "agent1"

    def test_effect_structure(self) -> None:
        """Test Effect TypedDict structure."""
        effect: Effect = {
            "uuid": "uuid-123",
            "kind": "Move",
            "payload": {"x": 10, "y": 20},
            "global_seq": 42,
            "sim_time": 100.0,
            "source_id": "agent1",
            "schema_version": "1.0.0",
        }

        assert effect["uuid"] == "uuid-123"
        assert effect["global_seq"] == 42
        assert effect["sim_time"] == 100.0

    def test_observation_delta_structure(self) -> None:
        """Test ObservationDelta TypedDict structure."""
        delta: ObservationDelta = {
            "view_seq": 15,
            "patches": [
                {
                    "op": "add",
                    "path": "/entities/new_entity",
                    "value": {"type": "player"},
                }
            ],
            "context_digest": "hash456",
            "schema_version": "1.0.0",
        }

        assert delta["view_seq"] == 15
        assert len(delta["patches"]) == 1
        assert delta["patches"][0]["op"] == "add"


class TestCancelToken:
    """Test CancelToken class."""

    def test_initialization(self) -> None:
        """Test CancelToken initialization."""
        token = CancelToken("req_123", "agent1")

        assert token.req_id == "req_123"
        assert token.agent_id == "agent1"
        assert not token.cancelled
        assert token.reason is None

    def test_initialization_validation(self) -> None:
        """Test CancelToken initialization validation."""
        # Valid initialization
        token = CancelToken("req_123", "agent1")
        assert token.req_id == "req_123"
        assert token.agent_id == "agent1"

        # Invalid empty req_id
        with pytest.raises(ValueError, match="req_id cannot be empty"):
            CancelToken("", "agent1")

        # Invalid whitespace-only req_id
        with pytest.raises(ValueError, match="req_id cannot be empty"):
            CancelToken("   ", "agent1")

        # Invalid empty agent_id
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            CancelToken("req_123", "")

        # Invalid whitespace-only agent_id
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            CancelToken("req_123", "   ")

    def test_cancel_functionality(self) -> None:
        """Test cancellation functionality."""
        token = CancelToken("req_123", "agent1")

        # Initially not cancelled
        assert not token.cancelled

        # Cancel with reason
        token.cancel("stale_context")

        # Now cancelled
        assert token.cancelled
        assert token.reason == "stale_context"  # type: ignore[unreachable]

    def test_cancel_validation(self) -> None:
        """Test cancel method validation."""
        token = CancelToken("req_123", "agent1")

        # Valid cancellation
        token.cancel("valid_reason")
        assert token.reason == "valid_reason"

        # Invalid empty reason
        with pytest.raises(ValueError, match="Cancellation reason cannot be empty"):
            token.cancel("")

        # Invalid whitespace-only reason
        with pytest.raises(ValueError, match="Cancellation reason cannot be empty"):
            token.cancel("   ")

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_wait_cancelled(self) -> None:
        """Test async wait for cancellation."""
        token = CancelToken("req_123", "agent1")

        # Start waiting in background
        wait_task = asyncio.create_task(token.wait_cancelled())

        # Give it a moment to start waiting
        await asyncio.sleep(0.01)
        assert not wait_task.done()

        # Cancel the token
        token.cancel("test_reason")

        # Wait should complete
        await wait_task
        assert token.cancelled
        assert token.reason == "test_reason"

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_wait_cancelled_already_cancelled(self) -> None:
        """Test wait_cancelled when token is already cancelled."""
        token = CancelToken("req_123", "agent1")
        token.cancel("already_cancelled")

        # Should return immediately
        await token.wait_cancelled()
        assert token.cancelled

    def test_repr(self) -> None:
        """Test CancelToken string representation."""
        token = CancelToken("req_123", "agent1")
        repr_str = repr(token)

        assert "req_id=req_123" in repr_str
        assert "agent_id=agent1" in repr_str
        assert "status=active" in repr_str

        token.cancel("test_reason")
        repr_str = repr(token)
        assert "status=cancelled" in repr_str
        assert "reason=test_reason" in repr_str

    def test_equality_and_hashing(self) -> None:
        """Test CancelToken equality and hashing."""
        token1 = CancelToken("req_123", "agent1")
        token2 = CancelToken("req_123", "agent1")
        token3 = CancelToken("req_456", "agent1")
        token4 = CancelToken("req_123", "agent2")

        # Equality based on req_id and agent_id
        assert token1 == token2
        assert token1 != token3
        assert token1 != token4

        # Hashing works for sets/dicts
        token_set = {token1, token2, token3, token4}
        assert len(token_set) == 3  # token1 and token2 are equal

        # Can be used as dict keys
        token_dict = {token1: "value1", token3: "value3", token4: "value4"}
        assert len(token_dict) == 3
        assert token_dict[token2] == "value1"  # token2 equals token1


class TestSchemaVersioning:
    """Test schema versioning support."""

    def test_intent_schema_version(self) -> None:
        """Test Intent includes schema_version."""
        intent: Intent = {
            "kind": "Custom",
            "payload": {},
            "context_seq": 1,
            "req_id": "req_1",
            "agent_id": "agent1",
            "priority": 0,
            "schema_version": "2.1.0",
        }

        assert intent["schema_version"] == "2.1.0"

    def test_all_types_have_schema_version(self) -> None:
        """Test that all message types include schema_version."""
        # Intent
        intent: Intent = {
            "kind": "Speak",
            "payload": {},
            "context_seq": 1,
            "req_id": "r1",
            "agent_id": "a1",
            "priority": 0,
            "schema_version": "1.0.0",
        }
        assert "schema_version" in intent

        # EffectDraft
        draft: EffectDraft = {
            "kind": "test",
            "payload": {},
            "source_id": "s1",
            "schema_version": "1.0.0",
        }
        assert "schema_version" in draft

        # Effect
        effect: Effect = {
            "uuid": "u1",
            "kind": "test",
            "payload": {},
            "global_seq": 1,
            "sim_time": 1.0,
            "source_id": "s1",
            "schema_version": "1.0.0",
        }
        assert "schema_version" in effect

        # ObservationDelta
        delta: ObservationDelta = {
            "view_seq": 1,
            "patches": [],
            "context_digest": "h1",
            "schema_version": "1.0.0",
        }
        assert "schema_version" in delta


class TestDataValidation:
    """Test data validation and edge cases."""

    def test_world_state_empty_collections(self) -> None:
        """Test WorldState handles empty collections properly."""
        world = WorldState()

        # Should be able to add to empty collections
        world.entities["new_entity"] = {"type": "test"}
        world.relationships["entity1"] = []
        world.spatial_index["entity1"] = (0.0, 0.0, 0.0)

        assert len(world.entities) == 1
        assert len(world.relationships) == 1
        assert len(world.spatial_index) == 1

    def test_view_context_digest_required(self) -> None:
        """Test that View requires context_digest."""
        with pytest.raises(ValidationError, match="Field required"):
            View(agent_id="agent1", view_seq=1)  # type: ignore[call-arg] # Missing context_digest

    def test_event_log_entry_effect_validation(self) -> None:
        """Test EventLogEntry validates effect structure."""
        # Should accept valid effect structure
        valid_effect = {
            "kind": "custom",
            "payload": {"custom": "data"},
            "source_id": "agent1",
            "schema_version": "1.0.0",
        }
        entry = EventLogEntry(
            global_seq=1, sim_time=1.0, effect=valid_effect, checksum="abc"
        )

        assert entry.effect["payload"]["custom"] == "data"

    def test_cancel_token_multiple_cancels(self) -> None:
        """Test CancelToken handles multiple cancel calls."""
        token = CancelToken("req_1", "agent1")

        token.cancel("reason1")
        assert token.reason == "reason1"

        # Second cancel should update reason
        token.cancel("reason2")
        assert token.reason == "reason2"
        assert token.cancelled  # Still cancelled
