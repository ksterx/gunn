"""
Unit tests for OpenAI structured output schemas.

This module tests the Pydantic models used for OpenAI's structured outputs,
ensuring proper validation and serialization of agent decisions.
"""

import pytest
from pydantic import ValidationError

from demo.shared.schemas import (
    AgentDecision,
    AttackAction,
    CommunicateAction,
    HealAction,
    MoveAction,
    RepairAction,
)


class TestMoveAction:
    """Test MoveAction schema validation."""

    def test_valid_move_action(self):
        """Test creating a valid move action."""
        action = MoveAction(
            target_position=(50.0, 75.0), reason="Moving to strategic position"
        )

        assert action.action_type == "move"
        assert action.target_position == (50.0, 75.0)
        assert action.reason == "Moving to strategic position"

    def test_move_action_serialization(self):
        """Test move action serialization to dict."""
        action = MoveAction(target_position=(100.0, 200.0), reason="Tactical retreat")

        data = action.model_dump()
        expected = {
            "action_type": "move",
            "target_position": (100.0, 200.0),
            "reason": "Tactical retreat",
        }

        assert data == expected

    def test_move_action_from_dict(self):
        """Test creating move action from dictionary."""
        data = {
            "action_type": "move",
            "target_position": [25.5, 30.7],
            "reason": "Flanking maneuver",
        }

        action = MoveAction(**data)
        assert action.target_position == (25.5, 30.7)
        assert action.reason == "Flanking maneuver"

    def test_invalid_position_type(self):
        """Test validation error for invalid position type."""
        with pytest.raises(ValidationError):
            MoveAction(
                target_position="invalid",  # Should be tuple
                reason="Test move",
            )

    def test_missing_reason(self):
        """Test validation error for missing reason."""
        with pytest.raises(ValidationError):
            MoveAction(target_position=(10.0, 20.0))


class TestAttackAction:
    """Test AttackAction schema validation."""

    def test_valid_attack_action(self):
        """Test creating a valid attack action."""
        action = AttackAction(
            target_agent_id="team_b_agent_1", reason="Enemy is low on health"
        )

        assert action.action_type == "attack"
        assert action.target_agent_id == "team_b_agent_1"
        assert action.reason == "Enemy is low on health"

    def test_attack_action_serialization(self):
        """Test attack action serialization."""
        action = AttackAction(
            target_agent_id="enemy_agent", reason="Eliminating threat"
        )

        data = action.model_dump()
        expected = {
            "action_type": "attack",
            "target_agent_id": "enemy_agent",
            "reason": "Eliminating threat",
        }

        assert data == expected

    def test_empty_target_id(self):
        """Test validation error for empty target ID."""
        with pytest.raises(ValidationError):
            AttackAction(target_agent_id="", reason="Test attack")


class TestHealAction:
    """Test HealAction schema validation."""

    def test_self_heal_action(self):
        """Test creating a self-heal action."""
        action = HealAction(target_agent_id=None, reason="Low health, need to recover")

        assert action.action_type == "heal"
        assert action.target_agent_id is None
        assert action.reason == "Low health, need to recover"

    def test_teammate_heal_action(self):
        """Test creating a teammate heal action."""
        action = HealAction(
            target_agent_id="team_a_agent_2", reason="Teammate is critically injured"
        )

        assert action.target_agent_id == "team_a_agent_2"
        assert action.reason == "Teammate is critically injured"

    def test_heal_action_default_target(self):
        """Test heal action with default None target."""
        action = HealAction(reason="Emergency healing")

        assert action.target_agent_id is None
        assert action.reason == "Emergency healing"


class TestRepairAction:
    """Test RepairAction schema validation."""

    def test_valid_repair_action(self):
        """Test creating a valid repair action."""
        action = RepairAction(reason="Weapon is damaged, need forge")

        assert action.action_type == "repair"
        assert action.reason == "Weapon is damaged, need forge"

    def test_repair_action_serialization(self):
        """Test repair action serialization."""
        action = RepairAction(reason="Weapon broken")

        data = action.model_dump()
        expected = {"action_type": "repair", "reason": "Weapon broken"}

        assert data == expected


class TestCommunicateAction:
    """Test CommunicateAction schema validation."""

    def test_valid_communicate_action(self):
        """Test creating a valid communicate action."""
        action = CommunicateAction(
            message="Enemy spotted at coordinates 50,75", urgency="high"
        )

        assert action.action_type == "communicate"
        assert action.message == "Enemy spotted at coordinates 50,75"
        assert action.urgency == "high"

    def test_communicate_action_urgency_levels(self):
        """Test all valid urgency levels."""
        for urgency in ["low", "medium", "high"]:
            action = CommunicateAction(message="Test message", urgency=urgency)
            assert action.urgency == urgency

    def test_invalid_urgency_level(self):
        """Test validation error for invalid urgency level."""
        with pytest.raises(ValidationError):
            CommunicateAction(
                message="Test message",
                urgency="critical",  # Not in allowed values
            )

    def test_message_length_validation(self):
        """Test message length validation."""
        # Valid message within limit
        action = CommunicateAction(
            message="A" * 200,  # Exactly at limit
            urgency="medium",
        )
        assert len(action.message) == 200

        # Invalid message over limit
        with pytest.raises(ValidationError):
            CommunicateAction(
                message="A" * 201,  # Over limit
                urgency="medium",
            )

    def test_empty_message(self):
        """Test validation error for empty message."""
        with pytest.raises(ValidationError):
            CommunicateAction(message="", urgency="low")


class TestAgentDecision:
    """Test AgentDecision composite schema validation."""

    def test_decision_with_move_action(self):
        """Test decision with move as primary action."""
        move_action = MoveAction(
            target_position=(75.0, 100.0), reason="Advancing to better position"
        )

        decision = AgentDecision(
            primary_action=move_action,
            confidence=0.8,
            strategic_assessment="Good tactical position available",
        )

        assert isinstance(decision.primary_action, MoveAction)
        assert decision.communication is None
        assert decision.confidence == 0.8
        assert decision.strategic_assessment == "Good tactical position available"

    def test_decision_with_attack_and_communication(self):
        """Test decision with attack action and communication."""
        attack_action = AttackAction(
            target_agent_id="team_b_agent_3", reason="Target is isolated and vulnerable"
        )

        comm_action = CommunicateAction(
            message="Engaging enemy agent 3, need backup", urgency="high"
        )

        decision = AgentDecision(
            primary_action=attack_action,
            communication=comm_action,
            confidence=0.9,
            strategic_assessment="High probability of successful engagement",
        )

        assert isinstance(decision.primary_action, AttackAction)
        assert isinstance(decision.communication, CommunicateAction)
        assert decision.confidence == 0.9

    def test_decision_with_heal_action(self):
        """Test decision with heal action."""
        heal_action = HealAction(
            target_agent_id="team_a_agent_1",
            reason="Teammate needs immediate medical attention",
        )

        decision = AgentDecision(
            primary_action=heal_action,
            confidence=0.7,
            strategic_assessment="Team survival is priority",
        )

        assert isinstance(decision.primary_action, HealAction)
        assert decision.primary_action.target_agent_id == "team_a_agent_1"

    def test_decision_with_repair_action(self):
        """Test decision with repair action."""
        repair_action = RepairAction(reason="Weapon condition is critical")

        decision = AgentDecision(
            primary_action=repair_action,
            confidence=0.6,
            strategic_assessment="Must maintain combat effectiveness",
        )

        assert isinstance(decision.primary_action, RepairAction)

    def test_confidence_validation(self):
        """Test confidence value validation."""
        move_action = MoveAction(target_position=(50.0, 50.0), reason="Test move")

        # Valid confidence values
        for confidence in [0.0, 0.5, 1.0]:
            decision = AgentDecision(
                primary_action=move_action,
                confidence=confidence,
                strategic_assessment="Test assessment",
            )
            assert decision.confidence == confidence

        # Invalid confidence values
        for invalid_confidence in [-0.1, 1.1, 2.0]:
            with pytest.raises(ValidationError):
                AgentDecision(
                    primary_action=move_action,
                    confidence=invalid_confidence,
                    strategic_assessment="Test assessment",
                )

    def test_decision_serialization(self):
        """Test complete decision serialization."""
        move_action = MoveAction(
            target_position=(25.0, 75.0), reason="Strategic repositioning"
        )

        comm_action = CommunicateAction(message="Moving to new position", urgency="low")

        decision = AgentDecision(
            primary_action=move_action,
            communication=comm_action,
            confidence=0.85,
            strategic_assessment="Calculated tactical move",
        )

        data = decision.model_dump()

        assert data["primary_action"]["action_type"] == "move"
        assert data["primary_action"]["target_position"] == (25.0, 75.0)
        assert data["communication"]["message"] == "Moving to new position"
        assert data["confidence"] == 0.85
        assert data["strategic_assessment"] == "Calculated tactical move"

    def test_decision_from_dict(self):
        """Test creating decision from dictionary data."""
        data = {
            "primary_action": {
                "action_type": "attack",
                "target_agent_id": "enemy_1",
                "reason": "Opportunity strike",
            },
            "communication": {
                "action_type": "communicate",
                "message": "Attacking enemy 1",
                "urgency": "medium",
            },
            "confidence": 0.75,
            "strategic_assessment": "Good opportunity for elimination",
        }

        decision = AgentDecision(**data)

        assert isinstance(decision.primary_action, AttackAction)
        assert decision.primary_action.target_agent_id == "enemy_1"
        assert isinstance(decision.communication, CommunicateAction)
        assert decision.communication.message == "Attacking enemy 1"
        assert decision.confidence == 0.75

    def test_missing_required_fields(self):
        """Test validation errors for missing required fields."""
        move_action = MoveAction(target_position=(10.0, 20.0), reason="Test move")

        # Missing confidence
        with pytest.raises(ValidationError):
            AgentDecision(
                primary_action=move_action, strategic_assessment="Test assessment"
            )

        # Missing strategic_assessment
        with pytest.raises(ValidationError):
            AgentDecision(primary_action=move_action, confidence=0.5)

        # Missing primary_action
        with pytest.raises(ValidationError):
            AgentDecision(confidence=0.5, strategic_assessment="Test assessment")


class TestSchemaIntegration:
    """Test integration between different schema types."""

    def test_all_action_types_in_decisions(self):
        """Test that all action types can be used in decisions."""
        actions = [
            MoveAction(target_position=(10.0, 20.0), reason="Move test"),
            AttackAction(target_agent_id="enemy", reason="Attack test"),
            HealAction(target_agent_id=None, reason="Heal test"),
            RepairAction(reason="Repair test"),
        ]

        for action in actions:
            decision = AgentDecision(
                primary_action=action,
                confidence=0.5,
                strategic_assessment="Integration test",
            )

            assert decision.primary_action.action_type == action.action_type
            assert decision.confidence == 0.5

    def test_decision_with_all_communication_urgencies(self):
        """Test decisions with all communication urgency levels."""
        move_action = MoveAction(target_position=(50.0, 50.0), reason="Test move")

        for urgency in ["low", "medium", "high"]:
            comm = CommunicateAction(
                message=f"Test message with {urgency} urgency", urgency=urgency
            )

            decision = AgentDecision(
                primary_action=move_action,
                communication=comm,
                confidence=0.7,
                strategic_assessment=f"Testing {urgency} urgency communication",
            )

            assert decision.communication.urgency == urgency

    def test_schema_type_discrimination(self):
        """Test that action types are properly discriminated."""
        # Create decision data with different action types
        move_data = {
            "primary_action": {
                "action_type": "move",
                "target_position": [10.0, 20.0],
                "reason": "Moving",
            },
            "confidence": 0.8,
            "strategic_assessment": "Test",
        }

        attack_data = {
            "primary_action": {
                "action_type": "attack",
                "target_agent_id": "enemy",
                "reason": "Attacking",
            },
            "confidence": 0.8,
            "strategic_assessment": "Test",
        }

        move_decision = AgentDecision(**move_data)
        attack_decision = AgentDecision(**attack_data)

        assert isinstance(move_decision.primary_action, MoveAction)
        assert isinstance(attack_decision.primary_action, AttackAction)
        assert move_decision.primary_action.action_type == "move"
        assert attack_decision.primary_action.action_type == "attack"
