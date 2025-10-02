"""
Integration tests for OpenAI structured output schemas.

This module tests the integration between the Pydantic schemas and
OpenAI's structured output format, ensuring compatibility.
"""

import json

import pytest

from demo.shared.schemas import (
    AgentDecision,
    AttackAction,
    CommunicateAction,
    HealAction,
    MoveAction,
    RepairAction,
)


class TestSchemaOpenAIIntegration:
    """Test schema compatibility with OpenAI structured outputs."""

    def test_move_decision_json_schema(self):
        """Test that MoveAction generates valid JSON schema for OpenAI."""
        schema = AgentDecision.model_json_schema()

        # Verify the schema has the required structure
        assert "properties" in schema
        assert "primary_action" in schema["properties"]
        assert "communication" in schema["properties"]
        assert "confidence" in schema["properties"]
        assert "strategic_assessment" in schema["properties"]

        # Verify primary_action has discriminated union
        primary_action_schema = schema["properties"]["primary_action"]
        assert "anyOf" in primary_action_schema or "oneOf" in primary_action_schema

    def test_decision_serialization_deserialization(self):
        """Test full serialization/deserialization cycle."""
        original_decision = AgentDecision(
            primary_action=AttackAction(
                target_agent_id="team_b_agent_1", reason="Enemy is vulnerable"
            ),
            communication=CommunicateAction(
                message="Engaging enemy agent 1", urgency="high"
            ),
            confidence=0.85,
            strategic_assessment="High probability of success",
        )

        # Serialize to JSON (as OpenAI would return)
        json_data = original_decision.model_dump_json()

        # Deserialize back to object
        parsed_data = json.loads(json_data)
        reconstructed_decision = AgentDecision(**parsed_data)

        # Verify all fields match
        assert isinstance(reconstructed_decision.primary_action, AttackAction)
        assert reconstructed_decision.primary_action.target_agent_id == "team_b_agent_1"
        assert reconstructed_decision.primary_action.reason == "Enemy is vulnerable"

        assert isinstance(reconstructed_decision.communication, CommunicateAction)
        assert reconstructed_decision.communication.message == "Engaging enemy agent 1"
        assert reconstructed_decision.communication.urgency == "high"

        assert reconstructed_decision.confidence == 0.85
        assert (
            reconstructed_decision.strategic_assessment == "High probability of success"
        )

    def test_all_action_types_serialization(self):
        """Test serialization of all action types."""
        actions = [
            MoveAction(target_position=(50.0, 75.0), reason="Strategic positioning"),
            AttackAction(target_agent_id="enemy_1", reason="Eliminate threat"),
            HealAction(target_agent_id="teammate_1", reason="Teammate needs healing"),
            RepairAction(reason="Weapon is damaged"),
        ]

        for action in actions:
            decision = AgentDecision(
                primary_action=action,
                confidence=0.7,
                strategic_assessment="Test decision",
            )

            # Test serialization
            json_data = decision.model_dump_json()
            parsed_data = json.loads(json_data)

            # Test deserialization
            reconstructed = AgentDecision(**parsed_data)

            # Verify action type is preserved
            assert reconstructed.primary_action.action_type == action.action_type
            assert type(reconstructed.primary_action) == type(action)

    def test_openai_response_format_compatibility(self):
        """Test compatibility with OpenAI response format."""
        # Simulate what OpenAI might return
        openai_response_data = {
            "primary_action": {
                "action_type": "move",
                "target_position": [100.0, 150.0],
                "reason": "Moving to strategic position for better coverage",
            },
            "communication": {
                "action_type": "communicate",
                "message": "Moving to high ground, cover me",
                "urgency": "medium",
            },
            "confidence": 0.82,
            "strategic_assessment": "Good tactical position available with minimal risk",
        }

        # Should parse without errors
        decision = AgentDecision(**openai_response_data)

        assert isinstance(decision.primary_action, MoveAction)
        assert decision.primary_action.target_position == (100.0, 150.0)
        assert isinstance(decision.communication, CommunicateAction)
        assert decision.communication.message == "Moving to high ground, cover me"
        assert decision.confidence == 0.82

    def test_minimal_decision_format(self):
        """Test minimal decision without communication."""
        minimal_data = {
            "primary_action": {
                "action_type": "repair",
                "reason": "Weapon needs maintenance",
            },
            "confidence": 0.6,
            "strategic_assessment": "Weapon condition is critical",
        }

        decision = AgentDecision(**minimal_data)

        assert isinstance(decision.primary_action, RepairAction)
        assert decision.communication is None
        assert decision.confidence == 0.6

    def test_schema_validation_edge_cases(self):
        """Test schema validation with edge case values."""
        # Test minimum confidence
        decision = AgentDecision(
            primary_action=MoveAction(
                target_position=(0.0, 0.0), reason="Emergency retreat"
            ),
            confidence=0.0,  # Minimum allowed
            strategic_assessment="Critical situation",
        )
        assert decision.confidence == 0.0

        # Test maximum confidence
        decision = AgentDecision(
            primary_action=AttackAction(
                target_agent_id="enemy", reason="Perfect opportunity"
            ),
            confidence=1.0,  # Maximum allowed
            strategic_assessment="Guaranteed success",
        )
        assert decision.confidence == 1.0

        # Test maximum message length
        long_message = "A" * 200  # Exactly at limit
        decision = AgentDecision(
            primary_action=MoveAction(target_position=(50.0, 50.0), reason="Test move"),
            communication=CommunicateAction(message=long_message, urgency="low"),
            confidence=0.5,
            strategic_assessment="Testing limits",
        )
        assert len(decision.communication.message) == 200

    def test_action_type_discrimination_in_json(self):
        """Test that action types are properly discriminated in JSON."""
        decisions_data = [
            {
                "primary_action": {
                    "action_type": "move",
                    "target_position": [10.0, 20.0],
                    "reason": "Moving",
                },
                "confidence": 0.8,
                "strategic_assessment": "Move decision",
            },
            {
                "primary_action": {
                    "action_type": "attack",
                    "target_agent_id": "enemy",
                    "reason": "Attacking",
                },
                "confidence": 0.9,
                "strategic_assessment": "Attack decision",
            },
            {
                "primary_action": {
                    "action_type": "heal",
                    "target_agent_id": None,
                    "reason": "Self healing",
                },
                "confidence": 0.7,
                "strategic_assessment": "Heal decision",
            },
            {
                "primary_action": {"action_type": "repair", "reason": "Weapon repair"},
                "confidence": 0.6,
                "strategic_assessment": "Repair decision",
            },
        ]

        expected_types = [MoveAction, AttackAction, HealAction, RepairAction]

        for data, expected_type in zip(decisions_data, expected_types, strict=False):
            decision = AgentDecision(**data)
            assert isinstance(decision.primary_action, expected_type)
            assert (
                decision.primary_action.action_type
                == data["primary_action"]["action_type"]
            )

    def test_nested_validation_errors(self):
        """Test that validation errors in nested objects are properly handled."""
        # Invalid action type should raise validation error
        with pytest.raises(Exception):  # Could be ValidationError or ValueError
            AgentDecision(
                primary_action={"action_type": "invalid_action", "reason": "Test"},
                confidence=0.5,
                strategic_assessment="Test",
            )

        # Invalid confidence should raise validation error
        with pytest.raises(Exception):
            AgentDecision(
                primary_action=MoveAction(target_position=(50.0, 50.0), reason="Test"),
                confidence=1.5,  # Over maximum
                strategic_assessment="Test",
            )
