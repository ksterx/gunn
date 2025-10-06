"""Unit tests for Pydantic addon."""

import pytest
from pydantic import ValidationError

from gunn.addons.pydantic import (
    EffectDraftModel,
    EffectModel,
    IntentModel,
    ObservationDeltaModel,
    effect_draft_from_dict,
    effect_draft_to_dict,
    effect_from_dict,
    effect_to_dict,
    intent_from_dict,
    intent_to_dict,
    observation_delta_from_dict,
    observation_delta_to_dict,
)
from gunn.schemas.types import Effect, EffectDraft, Intent, ObservationDelta


class TestIntentModel:
    """Test IntentModel validation and conversion."""

    def test_valid_intent_creation(self) -> None:
        """Test creating valid intent with Pydantic model."""
        intent = IntentModel(
            kind="Move",
            payload={"to": [10.0, 20.0]},
            context_seq=0,
            req_id="move_1",
            agent_id="agent_a",
            priority=1,
            schema_version="1.0.0",
        )

        assert intent.kind == "Move"
        assert intent.payload == {"to": [10.0, 20.0]}
        assert intent.context_seq == 0
        assert intent.req_id == "move_1"
        assert intent.agent_id == "agent_a"
        assert intent.priority == 1
        assert intent.schema_version == "1.0.0"

    def test_intent_validation_invalid_kind(self) -> None:
        """Test that invalid kind raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IntentModel(
                kind="InvalidKind",  # type: ignore[arg-type]
                payload={},
                context_seq=0,
                req_id="test",
                agent_id="agent_a",
                priority=1,
                schema_version="1.0.0",
            )

        assert "kind" in str(exc_info.value)

    def test_intent_validation_negative_context_seq(self) -> None:
        """Test that negative context_seq raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IntentModel(
                kind="Move",
                payload={},
                context_seq=-1,  # Invalid
                req_id="test",
                agent_id="agent_a",
                priority=1,
                schema_version="1.0.0",
            )

        assert "context_seq" in str(exc_info.value)

    def test_intent_to_dict_conversion(self) -> None:
        """Test converting IntentModel to TypedDict."""
        intent = IntentModel(
            kind="Speak",
            payload={"message": "Hello"},
            context_seq=5,
            req_id="speak_1",
            agent_id="agent_b",
            priority=2,
            schema_version="1.0.0",
        )

        intent_dict = intent_to_dict(intent)

        assert isinstance(intent_dict, dict)
        assert intent_dict["kind"] == "Speak"
        assert intent_dict["payload"] == {"message": "Hello"}
        assert intent_dict["context_seq"] == 5

    def test_intent_from_dict_conversion(self) -> None:
        """Test converting TypedDict to IntentModel."""
        intent_dict: Intent = {
            "kind": "Move",
            "payload": {"to": [1.0, 2.0]},
            "context_seq": 0,
            "req_id": "move_2",
            "agent_id": "agent_c",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        intent = intent_from_dict(intent_dict)

        assert isinstance(intent, IntentModel)
        assert intent.kind == "Move"
        assert intent.payload == {"to": [1.0, 2.0]}

    def test_intent_roundtrip_conversion(self) -> None:
        """Test roundtrip conversion IntentModel -> dict -> IntentModel."""
        original = IntentModel(
            kind="Interact",
            payload={"target": "door"},
            context_seq=10,
            req_id="interact_1",
            agent_id="agent_d",
            priority=5,
            schema_version="1.0.0",
        )

        # Convert to dict and back
        intent_dict = intent_to_dict(original)
        restored = intent_from_dict(intent_dict)

        assert restored.kind == original.kind
        assert restored.payload == original.payload
        assert restored.context_seq == original.context_seq
        assert restored.req_id == original.req_id
        assert restored.agent_id == original.agent_id
        assert restored.priority == original.priority


class TestEffectDraftModel:
    """Test EffectDraftModel validation and conversion."""

    def test_valid_effect_draft_creation(self) -> None:
        """Test creating valid effect draft with Pydantic model."""
        draft = EffectDraftModel(
            kind="Move",
            payload={"to": [10.0, 20.0]},
            source_id="agent_a",
            schema_version="1.0.0",
        )

        assert draft.kind == "Move"
        assert draft.payload == {"to": [10.0, 20.0]}
        assert draft.source_id == "agent_a"
        assert draft.schema_version == "1.0.0"

    def test_effect_draft_conversion(self) -> None:
        """Test converting EffectDraftModel to TypedDict and back."""
        draft = EffectDraftModel(
            kind="Speak",
            payload={"message": "Test"},
            source_id="agent_b",
            schema_version="1.0.0",
        )

        # Convert to dict
        draft_dict = effect_draft_to_dict(draft)
        assert isinstance(draft_dict, dict)
        assert draft_dict["kind"] == "Speak"

        # Convert back
        restored = effect_draft_from_dict(draft_dict)
        assert restored.kind == draft.kind
        assert restored.payload == draft.payload


class TestEffectModel:
    """Test EffectModel validation and conversion."""

    def test_valid_effect_creation(self) -> None:
        """Test creating valid effect with Pydantic model."""
        effect = EffectModel(
            uuid="test-uuid",
            kind="Move",
            payload={"to": [10.0, 20.0]},
            global_seq=1,
            sim_time=0.0,
            source_id="agent_a",
            schema_version="1.0.0",
            req_id="move_1",
            duration_ms=None,
            apply_at=None,
        )

        assert effect.uuid == "test-uuid"
        assert effect.kind == "Move"
        assert effect.global_seq == 1
        assert effect.sim_time == 0.0

    def test_effect_validation_negative_global_seq(self) -> None:
        """Test that negative global_seq raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EffectModel(
                uuid="test-uuid",
                kind="Move",
                payload={},
                global_seq=-1,  # Invalid
                sim_time=0.0,
                source_id="agent_a",
                schema_version="1.0.0",
            )

        assert "global_seq" in str(exc_info.value)

    def test_effect_validation_negative_sim_time(self) -> None:
        """Test that negative sim_time raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EffectModel(
                uuid="test-uuid",
                kind="Move",
                payload={},
                global_seq=0,
                sim_time=-1.0,  # Invalid
                source_id="agent_a",
                schema_version="1.0.0",
            )

        assert "sim_time" in str(exc_info.value)

    def test_effect_conversion(self) -> None:
        """Test converting EffectModel to TypedDict and back."""
        effect = EffectModel(
            uuid="uuid-123",
            kind="Speak",
            payload={"message": "Hello"},
            global_seq=5,
            sim_time=10.5,
            source_id="agent_b",
            schema_version="1.0.0",
            req_id="speak_1",
            duration_ms=100.0,
            apply_at=15.0,
        )

        # Convert to dict
        effect_dict = effect_to_dict(effect)
        assert isinstance(effect_dict, dict)
        assert effect_dict["uuid"] == "uuid-123"
        assert effect_dict["global_seq"] == 5

        # Convert back
        restored = effect_from_dict(effect_dict)
        assert restored.uuid == effect.uuid
        assert restored.global_seq == effect.global_seq
        assert restored.sim_time == effect.sim_time


class TestObservationDeltaModel:
    """Test ObservationDeltaModel validation and conversion."""

    def test_valid_observation_delta_creation(self) -> None:
        """Test creating valid observation delta with Pydantic model."""
        delta = ObservationDeltaModel(
            view_seq=1,
            patches=[{"op": "add", "path": "/test", "value": 123}],
            context_digest="abc123",
            schema_version="1.0.0",
            delivery_id="delivery-1",
            redelivery=False,
        )

        assert delta.view_seq == 1
        assert len(delta.patches) == 1
        assert delta.context_digest == "abc123"
        assert delta.redelivery is False

    def test_observation_delta_validation_negative_view_seq(self) -> None:
        """Test that negative view_seq raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ObservationDeltaModel(
                view_seq=-1,  # Invalid
                patches=[],
                context_digest="test",
                schema_version="1.0.0",
                delivery_id="delivery-1",
                redelivery=False,
            )

        assert "view_seq" in str(exc_info.value)

    def test_observation_delta_conversion(self) -> None:
        """Test converting ObservationDeltaModel to TypedDict and back."""
        delta = ObservationDeltaModel(
            view_seq=10,
            patches=[{"op": "replace", "path": "/x", "value": 100}],
            context_digest="digest-xyz",
            schema_version="1.0.0",
            delivery_id="delivery-2",
            redelivery=True,
        )

        # Convert to dict
        delta_dict = observation_delta_to_dict(delta)
        assert isinstance(delta_dict, dict)
        assert delta_dict["view_seq"] == 10
        assert delta_dict["redelivery"] is True

        # Convert back
        restored = observation_delta_from_dict(delta_dict)
        assert restored.view_seq == delta.view_seq
        assert restored.patches == delta.patches
        assert restored.redelivery == delta.redelivery


class TestExtraFieldsValidation:
    """Test that extra fields are rejected."""

    def test_intent_extra_fields_rejected(self) -> None:
        """Test that extra fields in Intent raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IntentModel(
                kind="Move",
                payload={},
                context_seq=0,
                req_id="test",
                agent_id="agent_a",
                priority=1,
                schema_version="1.0.0",
                extra_field="should_fail",  # type: ignore[call-arg]
            )

        assert "extra_field" in str(exc_info.value)

    def test_effect_extra_fields_rejected(self) -> None:
        """Test that extra fields in Effect raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EffectModel(
                uuid="test",
                kind="Move",
                payload={},
                global_seq=0,
                sim_time=0.0,
                source_id="agent_a",
                schema_version="1.0.0",
                extra_field="should_fail",  # type: ignore[call-arg]
            )

        assert "extra_field" in str(exc_info.value)
