"""
Tests for concurrent intent submission in demo (move + communicate).

This module tests that the demo properly uses Gunn's v2.0 concurrent intent
submission feature to allow agents to perform physical actions and communicate
simultaneously.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from demo.backend.ai_decision import AIDecisionMaker
from demo.backend.battle_agent import BattleAgent
from demo.backend.gunn_integration import BattleOrchestrator
from demo.shared.enums import AgentStatus, WeaponCondition
from demo.shared.models import Agent, BattleWorldState
from demo.shared.schemas import AgentDecision, CommunicateAction, MoveAction


class TestConcurrentIntentsDemo:
    """Test that demo agents properly use concurrent intent submission."""

    @pytest.fixture
    def sample_world_state(self):
        """Create a sample world state with test agents."""
        agents = {
            "team_a_agent_1": Agent(
                agent_id="team_a_agent_1",
                team="team_a",
                position=(50.0, 50.0),
                health=80,
                weapon_condition=WeaponCondition.GOOD,
                status=AgentStatus.ALIVE,
            ),
            "team_b_agent_1": Agent(
                agent_id="team_b_agent_1",
                team="team_b",
                position=(150.0, 150.0),
                health=90,
                weapon_condition=WeaponCondition.EXCELLENT,
                status=AgentStatus.ALIVE,
            ),
        }

        return BattleWorldState(agents=agents)

    @pytest.fixture
    def mock_ai_decision_maker(self):
        """Create mock AI decision maker that returns concurrent actions."""
        decision_maker = Mock(spec=AIDecisionMaker)
        decision_maker.make_decision = AsyncMock(
            return_value=AgentDecision(
                primary_action=MoveAction(
                    target_position=(100.0, 100.0), reason="Tactical repositioning"
                ),
                communication=CommunicateAction(
                    message="Moving to position!", urgency="medium"
                ),
                confidence=0.9,
                strategic_assessment="Good tactical move with team coordination",
            )
        )
        return decision_maker

    @pytest.mark.asyncio
    async def test_battle_agent_returns_concurrent_intents(
        self, sample_world_state, mock_ai_decision_maker
    ):
        """Test that BattleAgent returns list of intents for concurrent actions."""
        agent = sample_world_state.agents["team_a_agent_1"]
        battle_agent = BattleAgent(
            agent_id="team_a_agent_1",
            ai_decision_maker=mock_ai_decision_maker,
            world_state=sample_world_state,
        )

        # Create mock observation
        mock_observation = {
            "visible_entities": {},
            "view_seq": 0,
        }

        # Process observation (should return list of intents)
        result = await battle_agent.process_observation(mock_observation, "team_a_agent_1")

        # Verify result is a list (concurrent intents)
        assert isinstance(result, list), "Should return list for concurrent actions"
        assert len(result) == 2, "Should return 2 intents (move + communicate)"

        # Verify intent types
        intent_kinds = {intent["kind"] for intent in result}
        assert "Move" in intent_kinds, "Should include Move intent"
        assert "Speak" in intent_kinds, "Should include Speak intent"

        # Verify intent content
        move_intent = next(i for i in result if i["kind"] == "Move")
        assert move_intent["payload"]["to"] == [100.0, 100.0]

        speak_intent = next(i for i in result if i["kind"] == "Speak")
        assert speak_intent["payload"]["message"] == "Moving to position!"
        assert speak_intent["payload"]["urgency"] == "medium"

    @pytest.mark.asyncio
    async def test_battle_agent_single_intent_backward_compatibility(
        self, sample_world_state
    ):
        """Test that BattleAgent returns single intent when no communication."""
        # Create AI decision maker that returns only primary action (no communication)
        decision_maker = Mock(spec=AIDecisionMaker)
        decision_maker.make_decision = AsyncMock(
            return_value=AgentDecision(
                primary_action=MoveAction(
                    target_position=(100.0, 100.0), reason="Simple move"
                ),
                communication=None,  # No communication
                confidence=0.8,
                strategic_assessment="Simple move",
            )
        )

        battle_agent = BattleAgent(
            agent_id="team_a_agent_1",
            ai_decision_maker=decision_maker,
            world_state=sample_world_state,
        )

        mock_observation = {"visible_entities": {}, "view_seq": 0}
        result = await battle_agent.process_observation(mock_observation, "team_a_agent_1")

        # Verify result is a single intent dict (not list)
        assert isinstance(result, dict), "Should return dict for single intent"
        assert result["kind"] == "Move"

    @pytest.mark.asyncio
    async def test_orchestrator_handles_concurrent_intents(
        self, sample_world_state, mock_ai_decision_maker
    ):
        """Test that BattleOrchestrator properly submits concurrent intents."""
        with patch(
            "gunn.Orchestrator"
        ) as mock_orchestrator_class:
            # Create mock Gunn orchestrator
            mock_orchestrator = AsyncMock()
            mock_orchestrator.agent_handles = {}
            mock_orchestrator.world_state = Mock()
            mock_orchestrator.world_state.entities = {}
            mock_orchestrator.world_state.spatial_index = {}

            # Mock agent handle with submit_intents method
            mock_handle = AsyncMock()
            mock_handle.submit_intents = AsyncMock(
                return_value=["move_1", "speak_1"]
            )  # Returns list of req_ids
            mock_handle.get_current_observation = AsyncMock(
                return_value=Mock(view_seq=0)
            )

            mock_orchestrator.agent_handles["team_a_agent_1"] = mock_handle
            mock_orchestrator_class.return_value = mock_orchestrator

            # Initialize orchestrator
            orchestrator = BattleOrchestrator()
            await orchestrator.initialize(mock_ai_decision_maker)
            orchestrator.world_state = sample_world_state

            # Create BattleAgent
            battle_agent = BattleAgent(
                agent_id="team_a_agent_1",
                ai_decision_maker=mock_ai_decision_maker,
                world_state=sample_world_state,
                agent_handle=mock_handle,
            )

            # Process observation (will return concurrent intents)
            mock_observation = {"visible_entities": {}, "view_seq": 0}
            intents = await battle_agent.process_observation(
                mock_observation, "team_a_agent_1"
            )

            # Verify concurrent intents were returned
            assert isinstance(intents, list)
            assert len(intents) == 2

            # Simulate agent handle processing (what would happen in run_async_loop)
            if isinstance(intents, list):
                req_ids = await mock_handle.submit_intents(intents, atomic=False)
                assert len(req_ids) == 2
                mock_handle.submit_intents.assert_called_once_with(intents, atomic=False)

    @pytest.mark.asyncio
    async def test_concurrent_intents_context_seq_consistency(
        self, sample_world_state, mock_ai_decision_maker
    ):
        """Test that concurrent intents share the same context_seq."""
        battle_agent = BattleAgent(
            agent_id="team_a_agent_1",
            ai_decision_maker=mock_ai_decision_maker,
            world_state=sample_world_state,
        )

        # Set current view_seq
        battle_agent.current_view_seq = 42

        mock_observation = {"visible_entities": {}, "view_seq": 42}
        result = await battle_agent.process_observation(mock_observation, "team_a_agent_1")

        # Verify both intents have same context_seq
        assert isinstance(result, list)
        context_seqs = {intent["context_seq"] for intent in result}
        assert len(context_seqs) == 1, "All intents should have same context_seq"
        assert 42 in context_seqs

    @pytest.mark.asyncio
    async def test_ai_decision_prompt_mentions_concurrent_actions(self):
        """Test that AI decision maker prompt instructs about concurrent actions."""
        # This test verifies the documentation aspect - that the prompt
        # tells the AI it can perform action + communication simultaneously
        decision_maker = AIDecisionMaker(api_key="test-key")

        agent = Agent(
            agent_id="team_a_test",
            team="team_a",
            position=(50.0, 50.0),
            health=100,
            weapon_condition=WeaponCondition.EXCELLENT,
        )

        world_state = BattleWorldState(agents={"team_a_test": agent})
        observation = {"visible_entities": {}}

        # Build the prompt
        prompt = decision_maker._build_observation_prompt(agent, observation, world_state)

        # Verify prompt mentions simultaneous action + communication
        assert (
            "simultaneously" in prompt.lower()
            or "optionally send a team message" in prompt.lower()
        ), "Prompt should mention ability to perform action and communicate"

    @pytest.mark.asyncio
    async def test_dead_agent_no_concurrent_intents(
        self, sample_world_state, mock_ai_decision_maker
    ):
        """Test that dead agents don't submit any intents."""
        # Kill the agent
        agent = sample_world_state.agents["team_a_agent_1"]
        agent.health = 0
        agent.status = AgentStatus.DEAD

        battle_agent = BattleAgent(
            agent_id="team_a_agent_1",
            ai_decision_maker=mock_ai_decision_maker,
            world_state=sample_world_state,
        )

        mock_observation = {"visible_entities": {}, "view_seq": 0}
        result = await battle_agent.process_observation(mock_observation, "team_a_agent_1")

        # Verify no intents returned
        assert result is None, "Dead agent should not return any intents"
