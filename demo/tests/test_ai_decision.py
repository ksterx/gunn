"""
Unit tests for AI decision making functionality.

This module tests the AIDecisionMaker class, including OpenAI integration,
error handling, fallback mechanisms, and decision validation.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from demo.backend.ai_decision import AIDecisionMaker
from demo.shared.enums import AgentStatus, LocationType, WeaponCondition
from demo.shared.models import Agent, BattleWorldState, MapLocation
from demo.shared.schemas import (
    AgentDecision,
    AttackAction,
    CommunicateAction,
    HealAction,
    MoveAction,
    RepairAction,
)


class TestAIDecisionMaker:
    """Test AIDecisionMaker class functionality."""

    @pytest.fixture
    def ai_decision_maker(self):
        """Create AIDecisionMaker instance for testing."""
        return AIDecisionMaker(api_key="test_key", model="gpt-4.1-mini")

    @pytest.fixture
    def sample_agent(self):
        """Create a sample agent for testing."""
        return Agent(
            agent_id="team_a_agent_1",
            team="team_a",
            position=(50.0, 50.0),
            health=80,
            status=AgentStatus.ALIVE,
            weapon_condition=WeaponCondition.GOOD,
        )

    @pytest.fixture
    def sample_world_state(self):
        """Create a sample world state for testing."""
        agents = {
            "team_a_agent_1": Agent(
                agent_id="team_a_agent_1",
                team="team_a",
                position=(50.0, 50.0),
                health=80,
                weapon_condition=WeaponCondition.GOOD,
            ),
            "team_a_agent_2": Agent(
                agent_id="team_a_agent_2",
                team="team_a",
                position=(60.0, 60.0),
                health=90,
                weapon_condition=WeaponCondition.EXCELLENT,
            ),
            "team_b_agent_1": Agent(
                agent_id="team_b_agent_1",
                team="team_b",
                position=(150.0, 150.0),
                health=70,
                weapon_condition=WeaponCondition.DAMAGED,
            ),
        }

        map_locations = {
            "forge_a": MapLocation(
                position=(20.0, 80.0), location_type=LocationType.FORGE
            ),
            "forge_b": MapLocation(
                position=(180.0, 20.0), location_type=LocationType.FORGE
            ),
        }

        return BattleWorldState(
            agents=agents,
            map_locations=map_locations,
            game_time=30.0,
            game_status="active",
        )

    @pytest.fixture
    def sample_observation(self):
        """Create a sample observation for testing."""
        return {
            "agent_id": "team_a_agent_1",
            "visible_entities": {
                "team_a_agent_1": {
                    "agent_id": "team_a_agent_1",
                    "team": "team_a",
                    "position": (50.0, 50.0),
                    "health": 80,
                    "status": "alive",
                    "weapon_condition": "good",
                },
                "team_a_agent_2": {
                    "agent_id": "team_a_agent_2",
                    "team": "team_a",
                    "position": (60.0, 60.0),
                    "health": 90,
                    "status": "alive",
                },
                "team_b_agent_1": {
                    "agent_id": "team_b_agent_1",
                    "team": "team_b",
                    "position": (150.0, 150.0),
                    "status": "alive",
                },
                "forge_a": {
                    "type": "map_location",
                    "location_type": "forge",
                    "position": (20.0, 80.0),
                },
            },
        }

    def test_initialization(self):
        """Test AIDecisionMaker initialization."""
        ai_maker = AIDecisionMaker(api_key="test_key", model="gpt-4.1-mini")

        assert ai_maker.model == "gpt-4.1-mini"
        assert ai_maker.request_timeout == 30.0
        assert ai_maker.max_retries == 2
        assert ai_maker.client is not None

    def test_initialization_default_model(self):
        """Test AIDecisionMaker initialization with default model."""
        ai_maker = AIDecisionMaker(api_key="test_key")

        assert ai_maker.model == "gpt-4.1-mini"

    def test_build_system_prompt(
        self, ai_decision_maker, sample_agent, sample_world_state
    ):
        """Test system prompt building."""
        prompt = ai_decision_maker._build_system_prompt(
            sample_agent, sample_world_state, {"strategy": "aggressive offense"}
        )

        assert "team_a_agent_1" in prompt
        assert "team_a" in prompt
        assert "team_a_agent_2" in prompt  # Teammate
        assert "aggressive offense" in prompt
        assert "Health: 0-100 points" in prompt
        assert "Attack range: 15.0" in prompt
        assert "Vision range: 30.0" in prompt

    def test_build_system_prompt_no_teammates(self, ai_decision_maker):
        """Test system prompt with no teammates."""
        agent = Agent(agent_id="team_a_agent_1", team="team_a", position=(50.0, 50.0))

        world_state = BattleWorldState(agents={"team_a_agent_1": agent})

        prompt = ai_decision_maker._build_system_prompt(agent, world_state)

        assert "Your teammates are: none" in prompt

    def test_build_observation_prompt(
        self, ai_decision_maker, sample_agent, sample_observation, sample_world_state
    ):
        """Test observation prompt building."""
        prompt = ai_decision_maker._build_observation_prompt(
            sample_agent, sample_observation, sample_world_state
        )

        assert "Health: 80/100" in prompt
        assert "Position: (50.0, 50.0)" in prompt
        assert "Weapon Condition: good" in prompt
        assert "team_a_agent_2" in prompt  # Teammate
        assert "team_b_agent_1" in prompt  # Enemy
        assert "forge_a" in prompt  # Map location
        assert "Time: 30.0s" in prompt

    def test_create_fallback_decision(self, ai_decision_maker):
        """Test fallback decision creation."""
        decision = ai_decision_maker._create_fallback_decision("Test error")

        assert isinstance(decision, AgentDecision)
        assert isinstance(decision.primary_action, MoveAction)
        assert decision.primary_action.target_position == (100.0, 100.0)
        assert "Test error" in decision.primary_action.reason
        assert decision.confidence == 0.1
        assert "Test error" in decision.strategic_assessment
        assert decision.communication is not None
        assert "technical difficulties" in decision.communication.message.lower()

    @pytest.mark.asyncio
    async def test_make_decision_success(
        self, ai_decision_maker, sample_agent, sample_observation, sample_world_state
    ):
        """Test successful decision making."""
        # Mock successful OpenAI response
        mock_decision = AgentDecision(
            primary_action=MoveAction(
                target_position=(75.0, 75.0), reason="Moving to strategic position"
            ),
            confidence=0.8,
            strategic_assessment="Good tactical opportunity",
        )

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.parsed = mock_decision

        with patch.object(
            ai_decision_maker.client.beta.chat.completions,
            "parse",
            new_callable=AsyncMock,
        ) as mock_parse:
            mock_parse.return_value = mock_response

            decision = await ai_decision_maker.make_decision(
                "team_a_agent_1", sample_observation, sample_world_state
            )

            assert isinstance(decision, AgentDecision)
            assert isinstance(decision.primary_action, MoveAction)
            assert decision.primary_action.target_position == (75.0, 75.0)
            assert decision.confidence == 0.8

            # Verify OpenAI was called with correct parameters
            mock_parse.assert_called_once()
            call_args = mock_parse.call_args
            assert call_args[1]["model"] == "gpt-4.1-mini"
            assert call_args[1]["response_format"] == AgentDecision

    @pytest.mark.asyncio
    async def test_make_decision_agent_not_found(
        self, ai_decision_maker, sample_observation, sample_world_state
    ):
        """Test decision making when agent is not found."""
        decision = await ai_decision_maker.make_decision(
            "nonexistent_agent", sample_observation, sample_world_state
        )

        assert isinstance(decision, AgentDecision)
        assert isinstance(decision.primary_action, MoveAction)
        assert decision.confidence == 0.1
        assert "not found" in decision.strategic_assessment

    @pytest.mark.asyncio
    async def test_make_decision_openai_timeout(
        self, ai_decision_maker, sample_agent, sample_observation, sample_world_state
    ):
        """Test decision making with OpenAI timeout."""
        with patch.object(
            ai_decision_maker.client.beta.chat.completions,
            "parse",
            new_callable=AsyncMock,
        ) as mock_parse:
            mock_parse.side_effect = TimeoutError("Request timeout")

            decision = await ai_decision_maker.make_decision(
                "team_a_agent_1", sample_observation, sample_world_state
            )

            assert isinstance(decision, AgentDecision)
            assert decision.confidence == 0.1
            assert "failed" in decision.strategic_assessment.lower()

    @pytest.mark.asyncio
    async def test_make_decision_openai_error(
        self, ai_decision_maker, sample_agent, sample_observation, sample_world_state
    ):
        """Test decision making with OpenAI API error."""
        with patch.object(
            ai_decision_maker.client.beta.chat.completions,
            "parse",
            new_callable=AsyncMock,
        ) as mock_parse:
            mock_parse.side_effect = Exception("API Error")

            decision = await ai_decision_maker.make_decision(
                "team_a_agent_1", sample_observation, sample_world_state
            )

            assert isinstance(decision, AgentDecision)
            assert decision.confidence == 0.1
            assert "failed" in decision.strategic_assessment.lower()

    @pytest.mark.asyncio
    async def test_make_decision_empty_response(
        self, ai_decision_maker, sample_agent, sample_observation, sample_world_state
    ):
        """Test decision making with empty OpenAI response."""
        mock_response = Mock()
        mock_response.choices = []

        with patch.object(
            ai_decision_maker.client.beta.chat.completions,
            "parse",
            new_callable=AsyncMock,
        ) as mock_parse:
            mock_parse.return_value = mock_response

            decision = await ai_decision_maker.make_decision(
                "team_a_agent_1", sample_observation, sample_world_state
            )

            assert isinstance(decision, AgentDecision)
            assert decision.confidence == 0.1

    @pytest.mark.asyncio
    async def test_validate_decision_valid_move(
        self, ai_decision_maker, sample_agent, sample_world_state
    ):
        """Test validation of valid move decision."""
        decision = AgentDecision(
            primary_action=MoveAction(
                target_position=(75.0, 75.0), reason="Strategic move"
            ),
            confidence=0.8,
            strategic_assessment="Good position",
        )

        is_valid, message = await ai_decision_maker.validate_decision(
            decision, sample_agent, sample_world_state
        )

        assert is_valid
        assert message == "Decision is valid"

    @pytest.mark.asyncio
    async def test_validate_decision_valid_attack(
        self, ai_decision_maker, sample_agent, sample_world_state
    ):
        """Test validation of valid attack decision."""
        decision = AgentDecision(
            primary_action=AttackAction(
                target_agent_id="team_b_agent_1", reason="Enemy in range"
            ),
            confidence=0.9,
            strategic_assessment="Good attack opportunity",
        )

        is_valid, message = await ai_decision_maker.validate_decision(
            decision, sample_agent, sample_world_state
        )

        assert is_valid
        assert message == "Decision is valid"

    @pytest.mark.asyncio
    async def test_validate_decision_dead_agent(
        self, ai_decision_maker, sample_world_state
    ):
        """Test validation with dead agent."""
        dead_agent = Agent(
            agent_id="team_a_agent_1",
            team="team_a",
            position=(50.0, 50.0),
            health=0,
            status=AgentStatus.DEAD,
        )

        decision = AgentDecision(
            primary_action=MoveAction(target_position=(75.0, 75.0), reason="Move"),
            confidence=0.8,
            strategic_assessment="Test",
        )

        is_valid, message = await ai_decision_maker.validate_decision(
            decision, dead_agent, sample_world_state
        )

        assert not is_valid
        assert "not alive" in message

    @pytest.mark.asyncio
    async def test_validate_decision_broken_weapon_attack(
        self, ai_decision_maker, sample_world_state
    ):
        """Test validation of attack with broken weapon."""
        broken_weapon_agent = Agent(
            agent_id="team_a_agent_1",
            team="team_a",
            position=(50.0, 50.0),
            health=100,
            weapon_condition=WeaponCondition.BROKEN,
        )

        decision = AgentDecision(
            primary_action=AttackAction(
                target_agent_id="team_b_agent_1", reason="Attack enemy"
            ),
            confidence=0.8,
            strategic_assessment="Test",
        )

        is_valid, message = await ai_decision_maker.validate_decision(
            decision, broken_weapon_agent, sample_world_state
        )

        assert not is_valid
        assert "cannot attack" in message.lower()

    @pytest.mark.asyncio
    async def test_validate_decision_attack_teammate(
        self, ai_decision_maker, sample_agent, sample_world_state
    ):
        """Test validation of attacking teammate."""
        decision = AgentDecision(
            primary_action=AttackAction(
                target_agent_id="team_a_agent_2",  # Teammate
                reason="Friendly fire",
            ),
            confidence=0.8,
            strategic_assessment="Test",
        )

        is_valid, message = await ai_decision_maker.validate_decision(
            decision, sample_agent, sample_world_state
        )

        assert not is_valid
        assert "teammate" in message.lower()

    @pytest.mark.asyncio
    async def test_validate_decision_heal_enemy(
        self, ai_decision_maker, sample_agent, sample_world_state
    ):
        """Test validation of healing enemy."""
        decision = AgentDecision(
            primary_action=HealAction(
                target_agent_id="team_b_agent_1",  # Enemy
                reason="Heal enemy",
            ),
            confidence=0.8,
            strategic_assessment="Test",
        )

        is_valid, message = await ai_decision_maker.validate_decision(
            decision, sample_agent, sample_world_state
        )

        assert not is_valid
        assert "enemy" in message.lower()

    @pytest.mark.asyncio
    async def test_validate_decision_out_of_bounds_move(
        self, ai_decision_maker, sample_agent, sample_world_state
    ):
        """Test validation of out-of-bounds move."""
        decision = AgentDecision(
            primary_action=MoveAction(
                target_position=(-10.0, 50.0),  # Negative coordinate
                reason="Invalid move",
            ),
            confidence=0.8,
            strategic_assessment="Test",
        )

        is_valid, message = await ai_decision_maker.validate_decision(
            decision, sample_agent, sample_world_state
        )

        assert not is_valid
        assert "out of bounds" in message.lower()

    @pytest.mark.asyncio
    async def test_validate_decision_repair_excellent_weapon(
        self, ai_decision_maker, sample_world_state
    ):
        """Test validation of repairing excellent weapon."""
        excellent_weapon_agent = Agent(
            agent_id="team_a_agent_1",
            team="team_a",
            position=(50.0, 50.0),
            weapon_condition=WeaponCondition.EXCELLENT,
        )

        decision = AgentDecision(
            primary_action=RepairAction(reason="Repair excellent weapon"),
            confidence=0.8,
            strategic_assessment="Test",
        )

        is_valid, message = await ai_decision_maker.validate_decision(
            decision, excellent_weapon_agent, sample_world_state
        )

        assert not is_valid
        assert "excellent" in message.lower()

    @pytest.mark.asyncio
    async def test_validate_decision_with_communication(
        self, ai_decision_maker, sample_agent, sample_world_state
    ):
        """Test validation of decision with valid communication."""
        decision = AgentDecision(
            primary_action=MoveAction(target_position=(75.0, 75.0), reason="Move"),
            communication=CommunicateAction(
                message="Moving to new position", urgency="medium"
            ),
            confidence=0.8,
            strategic_assessment="Test",
        )

        is_valid, message = await ai_decision_maker.validate_decision(
            decision, sample_agent, sample_world_state
        )

        assert is_valid
        assert message == "Decision is valid"

    @pytest.mark.asyncio
    async def test_batch_make_decisions(self, ai_decision_maker, sample_world_state):
        """Test batch decision making for multiple agents."""
        observations = {
            "team_a_agent_1": {"agent_id": "team_a_agent_1", "visible_entities": {}},
            "team_a_agent_2": {"agent_id": "team_a_agent_2", "visible_entities": {}},
        }

        # Mock successful responses for both agents
        mock_decision_1 = AgentDecision(
            primary_action=MoveAction(target_position=(70.0, 70.0), reason="Move 1"),
            confidence=0.8,
            strategic_assessment="Assessment 1",
        )

        mock_decision_2 = AgentDecision(
            primary_action=AttackAction(
                target_agent_id="team_b_agent_1", reason="Attack"
            ),
            confidence=0.9,
            strategic_assessment="Assessment 2",
        )

        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.parsed = mock_decision_1

        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.parsed = mock_decision_2

        with patch.object(
            ai_decision_maker.client.beta.chat.completions,
            "parse",
            new_callable=AsyncMock,
        ) as mock_parse:
            mock_parse.side_effect = [mock_response_1, mock_response_2]

            decisions = await ai_decision_maker.batch_make_decisions(
                observations, sample_world_state
            )

            assert len(decisions) == 2
            assert "team_a_agent_1" in decisions
            assert "team_a_agent_2" in decisions
            assert isinstance(decisions["team_a_agent_1"].primary_action, MoveAction)
            assert isinstance(decisions["team_a_agent_2"].primary_action, AttackAction)

    @pytest.mark.asyncio
    async def test_batch_make_decisions_with_errors(
        self, ai_decision_maker, sample_world_state
    ):
        """Test batch decision making with some errors."""
        observations = {
            "team_a_agent_1": {"agent_id": "team_a_agent_1", "visible_entities": {}},
            "team_a_agent_2": {"agent_id": "team_a_agent_2", "visible_entities": {}},
        }

        # Mock one success and one error
        mock_decision = AgentDecision(
            primary_action=MoveAction(target_position=(70.0, 70.0), reason="Move"),
            confidence=0.8,
            strategic_assessment="Success",
        )

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.parsed = mock_decision

        with patch.object(
            ai_decision_maker.client.beta.chat.completions,
            "parse",
            new_callable=AsyncMock,
        ) as mock_parse:
            mock_parse.side_effect = [mock_response, Exception("API Error")]

            decisions = await ai_decision_maker.batch_make_decisions(
                observations, sample_world_state
            )

            assert len(decisions) == 2
            assert "team_a_agent_1" in decisions
            assert "team_a_agent_2" in decisions

            # First agent should have successful decision
            assert decisions["team_a_agent_1"].confidence == 0.8

            # Second agent should have fallback decision
            assert decisions["team_a_agent_2"].confidence == 0.1
            assert "failed" in decisions["team_a_agent_2"].strategic_assessment.lower()
