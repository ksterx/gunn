"""Runtime test for Move intent processing.

This script tests that Move intents are actually processed correctly
in the battle demo, addressing the user's bug report:
'コミュニケーションしているけど、全くMove溶かしていない'
"""

from unittest.mock import AsyncMock, Mock

import pytest

from demo.backend.ai_decision import AIDecisionMaker
from demo.backend.battle_agent import BattleAgent
from demo.backend.gunn_integration import BattleEffectValidator
from demo.shared.enums import AgentStatus, WeaponCondition
from demo.shared.models import Agent, BattleWorldState
from demo.shared.schemas import AgentDecision, CommunicateAction, MoveAction


@pytest.fixture
def world_state():
    """Create a test world state with one agent."""
    agents = {
        "team_a_agent_1": Agent(
            agent_id="team_a_agent_1",
            team="team_a",
            position=(50.0, 50.0),
            health=100,
            weapon_condition=WeaponCondition.EXCELLENT,
            status=AgentStatus.ALIVE,
        ),
    }
    return BattleWorldState(agents=agents)


@pytest.fixture
def mock_ai_decision_maker():
    """Create mock AI decision maker that returns Move + Communicate."""
    decision_maker = Mock(spec=AIDecisionMaker)
    decision_maker.make_decision = AsyncMock(
        return_value=AgentDecision(
            primary_action=MoveAction(
                target_position=(100.0, 100.0), reason="Testing move"
            ),
            communication=CommunicateAction(
                message="Moving to new position!", urgency="medium"
            ),
            confidence=0.9,
            strategic_assessment="Testing concurrent move and communication",
        )
    )
    return decision_maker


@pytest.mark.asyncio
async def test_move_intent_validation(world_state):
    """Test that Move intents pass validation."""
    validator = BattleEffectValidator(world_state)

    # Valid move intent
    move_intent = {
        "kind": "Move",
        "payload": {"to": [100.0, 100.0], "reason": "Testing"},
        "context_seq": 0,
        "req_id": "move_1",
        "agent_id": "team_a_agent_1",
        "priority": 1,
        "schema_version": "1.0.0",
    }

    # Should not raise ValidationError
    result = validator.validate_intent(move_intent, world_state)
    assert result is True, "Move intent should pass validation"


@pytest.mark.asyncio
async def test_move_intent_to_effect_conversion(world_state):
    """Test that Move intents are converted to Move effects."""
    validator = BattleEffectValidator(world_state)

    move_intent = {
        "kind": "Move",
        "payload": {"to": [100.0, 100.0], "reason": "Testing"},
        "context_seq": 0,
        "req_id": "move_1",
        "agent_id": "team_a_agent_1",
        "priority": 1,
        "schema_version": "1.0.0",
    }

    # Convert to effect draft
    effect_draft = validator.intent_to_effect(move_intent, world_state)

    assert effect_draft["kind"] == "Move", "Effect should be Move type"
    assert effect_draft["source_id"] == "team_a_agent_1"
    assert effect_draft["payload"]["agent_id"] == "team_a_agent_1"
    assert effect_draft["payload"]["old_position"] == [50.0, 50.0]
    assert effect_draft["payload"]["new_position"] == [100.0, 100.0]
    assert effect_draft["payload"]["reason"] == "Testing"


@pytest.mark.asyncio
async def test_concurrent_move_and_speak_intents(
    world_state, mock_ai_decision_maker
):
    """Test that both Move and Speak intents are created correctly."""
    battle_agent = BattleAgent(
        agent_id="team_a_agent_1",
        ai_decision_maker=mock_ai_decision_maker,
        world_state=world_state,
    )

    mock_observation = {"visible_entities": {}, "view_seq": 0}
    result = await battle_agent.process_observation(
        mock_observation, "team_a_agent_1"
    )

    # Should return list of 2 intents
    assert isinstance(result, list), "Should return list for concurrent actions"
    assert len(result) == 2, "Should return 2 intents"

    # Validate intent kinds
    intent_kinds = {intent["kind"] for intent in result}
    assert "Move" in intent_kinds, "Should include Move intent"
    assert "Speak" in intent_kinds, "Should include Speak intent"

    # Validate Move intent
    move_intent = next(i for i in result if i["kind"] == "Move")
    assert move_intent["payload"]["to"] == [100.0, 100.0]
    assert move_intent["payload"]["reason"] == "Testing move"

    # Validate Speak intent
    speak_intent = next(i for i in result if i["kind"] == "Speak")
    assert speak_intent["payload"]["message"] == "Moving to new position!"


@pytest.mark.asyncio
async def test_move_validation_edge_cases(world_state):
    """Test Move intent validation with invalid inputs."""
    validator = BattleEffectValidator(world_state)

    # Test 1: Missing target position
    invalid_intent_1 = {
        "kind": "Move",
        "payload": {},
        "context_seq": 0,
        "req_id": "move_1",
        "agent_id": "team_a_agent_1",
        "priority": 1,
        "schema_version": "1.0.0",
    }

    with pytest.raises(Exception) as exc_info:
        validator.validate_intent(invalid_intent_1, world_state)
    assert "Target position is required" in str(exc_info.value)

    # Test 2: Out of bounds position
    invalid_intent_2 = {
        "kind": "Move",
        "payload": {"to": [300.0, 300.0]},  # Out of bounds
        "context_seq": 0,
        "req_id": "move_2",
        "agent_id": "team_a_agent_1",
        "priority": 1,
        "schema_version": "1.0.0",
    }

    with pytest.raises(Exception) as exc_info:
        validator.validate_intent(invalid_intent_2, world_state)
    assert "out of bounds" in str(exc_info.value)

    # Test 3: Invalid position format
    invalid_intent_3 = {
        "kind": "Move",
        "payload": {"to": "invalid"},
        "context_seq": 0,
        "req_id": "move_3",
        "agent_id": "team_a_agent_1",
        "priority": 1,
        "schema_version": "1.0.0",
    }

    with pytest.raises(Exception) as exc_info:
        validator.validate_intent(invalid_intent_3, world_state)
    assert "must be a list or tuple" in str(exc_info.value)


@pytest.mark.asyncio
async def test_dead_agent_cannot_move(world_state):
    """Test that dead agents cannot submit Move intents."""
    # Kill the agent
    world_state.agents["team_a_agent_1"].health = 0
    world_state.agents["team_a_agent_1"].status = AgentStatus.DEAD

    validator = BattleEffectValidator(world_state)

    move_intent = {
        "kind": "Move",
        "payload": {"to": [100.0, 100.0]},
        "context_seq": 0,
        "req_id": "move_1",
        "agent_id": "team_a_agent_1",
        "priority": 1,
        "schema_version": "1.0.0",
    }

    with pytest.raises(Exception) as exc_info:
        validator.validate_intent(move_intent, world_state)
    assert "Dead agents cannot move" in str(exc_info.value)
