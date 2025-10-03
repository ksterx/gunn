"""
Tests for concurrent agent decision processing.

This module tests the concurrent decision processing functionality including
parallel AI decision making, simultaneous intent submission, and deterministic
ordering based on agent_id sorting.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from demo.backend.ai_decision import AIDecisionMaker
from demo.backend.gunn_integration import BattleOrchestrator
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


class TestConcurrentProcessing:
    """Test concurrent agent decision processing functionality."""

    @pytest.fixture
    async def battle_orchestrator(self):
        """Create BattleOrchestrator instance for testing."""
        orchestrator = BattleOrchestrator()

        # Mock AI decision maker
        ai_decision_maker = Mock(spec=AIDecisionMaker)

        # Mock Gunn orchestrator to avoid actual initialization
        with patch(
            "demo.backend.gunn_integration.Orchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.agent_handles = {}
            mock_orchestrator.observation_policies = {}
            mock_orchestrator.world_state = Mock()
            mock_orchestrator.world_state.entities = {}
            mock_orchestrator.world_state.spatial_index = {}
            mock_orchestrator.world_state.metadata = {}
            mock_orchestrator_class.return_value = mock_orchestrator

            await orchestrator.initialize(ai_decision_maker)

        return orchestrator

    @pytest.fixture
    def sample_world_state(self):
        """Create a sample world state with multiple agents."""
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
            "team_b_agent_2": Agent(
                agent_id="team_b_agent_2",
                team="team_b",
                position=(160.0, 160.0),
                health=100,
                weapon_condition=WeaponCondition.EXCELLENT,
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

    @pytest.mark.asyncio
    async def test_process_agent_decision_success(
        self, battle_orchestrator, sample_world_state
    ):
        """Test successful single agent decision processing."""
        battle_orchestrator.world_state = sample_world_state

        # Mock agent handle and observation policy
        mock_handle = Mock()
        mock_policy = Mock()
        mock_observation = Mock()
        mock_observation.visible_entities = {"test": "data"}
        mock_observation.visible_relationships = {}
        mock_observation.context_digest = "test_digest"
        mock_observation.view_seq = 1

        battle_orchestrator.orchestrator.agent_handles["team_a_agent_1"] = mock_handle
        battle_orchestrator.orchestrator.observation_policies["team_a_agent_1"] = (
            mock_policy
        )
        mock_policy.filter_world_state.return_value = mock_observation

        # Mock AI decision
        mock_decision = AgentDecision(
            primary_action=MoveAction(
                target_position=(75.0, 75.0), reason="Strategic move"
            ),
            confidence=0.8,
            strategic_assessment="Good position",
        )

        battle_orchestrator.ai_decision_maker.make_decision = AsyncMock(
            return_value=mock_decision
        )

        # Test decision processing
        agent_id, result = await battle_orchestrator._process_agent_decision(
            "team_a_agent_1"
        )

        assert agent_id == "team_a_agent_1"
        assert isinstance(result, AgentDecision)
        assert isinstance(result.primary_action, MoveAction)
        assert result.primary_action.target_position == (75.0, 75.0)

        # Verify AI decision maker was called with correct parameters
        battle_orchestrator.ai_decision_maker.make_decision.assert_called_once()
        call_args = battle_orchestrator.ai_decision_maker.make_decision.call_args
        assert call_args[0][0] == "team_a_agent_1"  # agent_id
        assert "visible_entities" in call_args[0][1]  # observation dict
        assert call_args[0][2] == sample_world_state  # world_state

    @pytest.mark.asyncio
    async def test_process_agent_decision_error(
        self, battle_orchestrator, sample_world_state
    ):
        """Test agent decision processing with error."""
        battle_orchestrator.world_state = sample_world_state

        # Mock missing agent handle to trigger error
        battle_orchestrator.orchestrator.agent_handles = {}

        # Test decision processing with error
        agent_id, result = await battle_orchestrator._process_agent_decision(
            "team_a_agent_1"
        )

        assert agent_id == "team_a_agent_1"
        assert isinstance(result, Exception)
        assert "not registered" in str(result)

    @pytest.mark.asyncio
    async def test_decision_to_intents_move_action(self, battle_orchestrator):
        """Test converting move action to Gunn intents."""
        decision = AgentDecision(
            primary_action=MoveAction(
                target_position=(100.0, 100.0), reason="Strategic repositioning"
            ),
            confidence=0.9,
            strategic_assessment="Good tactical move",
        )

        intents = await battle_orchestrator._decision_to_intents(
            "team_a_agent_1", decision
        )

        assert len(intents) == 1
        intent = intents[0]
        assert intent["kind"] == "Move"
        assert intent["payload"]["target_position"] == (100.0, 100.0)
        assert intent["payload"]["reason"] == "Strategic repositioning"
        assert intent["agent_id"] == "team_a_agent_1"
        assert intent["priority"] == 0
        assert intent["schema_version"] == "1.0.0"
        assert "req_id" in intent

    @pytest.mark.asyncio
    async def test_decision_to_intents_attack_action(self, battle_orchestrator):
        """Test converting attack action to Gunn intents."""
        decision = AgentDecision(
            primary_action=AttackAction(
                target_agent_id="team_b_agent_1", reason="Enemy in range"
            ),
            confidence=0.8,
            strategic_assessment="Good attack opportunity",
        )

        intents = await battle_orchestrator._decision_to_intents(
            "team_a_agent_1", decision
        )

        assert len(intents) == 1
        intent = intents[0]
        assert intent["kind"] == "Attack"
        assert intent["payload"]["target_agent_id"] == "team_b_agent_1"
        assert intent["payload"]["reason"] == "Enemy in range"
        assert intent["agent_id"] == "team_a_agent_1"
        assert intent["priority"] == 1  # Higher priority for combat
        assert intent["schema_version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_decision_to_intents_heal_action(self, battle_orchestrator):
        """Test converting heal action to Gunn intents."""
        decision = AgentDecision(
            primary_action=HealAction(
                target_agent_id="team_a_agent_2", reason="Teammate needs healing"
            ),
            confidence=0.7,
            strategic_assessment="Support teammate",
        )

        intents = await battle_orchestrator._decision_to_intents(
            "team_a_agent_1", decision
        )

        assert len(intents) == 1
        intent = intents[0]
        assert intent["kind"] == "Heal"
        assert intent["payload"]["target_agent_id"] == "team_a_agent_2"
        assert intent["payload"]["reason"] == "Teammate needs healing"
        assert intent["agent_id"] == "team_a_agent_1"
        assert intent["priority"] == 0

    @pytest.mark.asyncio
    async def test_decision_to_intents_repair_action(self, battle_orchestrator):
        """Test converting repair action to Gunn intents."""
        decision = AgentDecision(
            primary_action=RepairAction(reason="Weapon is damaged"),
            confidence=0.6,
            strategic_assessment="Need weapon repair",
        )

        intents = await battle_orchestrator._decision_to_intents(
            "team_a_agent_1", decision
        )

        assert len(intents) == 1
        intent = intents[0]
        assert intent["kind"] == "Repair"
        assert intent["payload"]["reason"] == "Weapon is damaged"
        assert intent["agent_id"] == "team_a_agent_1"
        assert intent["priority"] == 0

    @pytest.mark.asyncio
    async def test_decision_to_intents_with_communication(self, battle_orchestrator):
        """Test converting decision with both action and communication."""
        decision = AgentDecision(
            primary_action=MoveAction(
                target_position=(80.0, 80.0), reason="Flanking maneuver"
            ),
            communication=CommunicateAction(
                message="Moving to flank enemy position", urgency="high"
            ),
            confidence=0.9,
            strategic_assessment="Coordinated attack",
        )

        intents = await battle_orchestrator._decision_to_intents(
            "team_a_agent_1", decision
        )

        assert len(intents) == 2

        # Check move intent
        move_intent = next(i for i in intents if i["kind"] == "Move")
        assert move_intent["payload"]["target_position"] == (80.0, 80.0)
        assert move_intent["priority"] == 0

        # Check communication intent
        comm_intent = next(i for i in intents if i["kind"] == "Communicate")
        assert comm_intent["payload"]["message"] == "Moving to flank enemy position"
        assert comm_intent["payload"]["urgency"] == "high"
        assert comm_intent["payload"]["team_only"] is True
        assert comm_intent["priority"] == -1  # Lower priority for communication

    @pytest.mark.asyncio
    async def test_process_concurrent_intents_success(
        self, battle_orchestrator, sample_world_state
    ):
        """Test successful concurrent intent processing."""
        battle_orchestrator.world_state = sample_world_state

        # Mock decisions for multiple agents
        agent_decisions = {
            "team_a_agent_1": AgentDecision(
                primary_action=MoveAction(
                    target_position=(70.0, 70.0), reason="Move 1"
                ),
                confidence=0.8,
                strategic_assessment="Assessment 1",
            ),
            "team_a_agent_2": AgentDecision(
                primary_action=AttackAction(
                    target_agent_id="team_b_agent_1", reason="Attack enemy"
                ),
                confidence=0.9,
                strategic_assessment="Assessment 2",
            ),
        }

        # Mock orchestrator submit_intents method
        battle_orchestrator.orchestrator.submit_intents = AsyncMock()

        # Test concurrent intent processing
        await battle_orchestrator._process_concurrent_intents(agent_decisions)

        # Verify submit_intents was called
        battle_orchestrator.orchestrator.submit_intents.assert_called_once()
        call_args = battle_orchestrator.orchestrator.submit_intents.call_args

        # Check that intents were submitted
        submitted_intents = call_args[0][0]  # First argument is the intents list
        assert len(submitted_intents) == 2  # One intent per agent

        # Check that sim_time was set
        sim_time = call_args[0][1]  # Second argument is sim_time
        assert sim_time == sample_world_state.game_time

        # Verify deterministic ordering (sorted by agent_id)
        agent_ids_in_intents = [intent["agent_id"] for intent in submitted_intents]
        assert agent_ids_in_intents == sorted(agent_ids_in_intents)

    @pytest.mark.asyncio
    async def test_process_concurrent_intents_with_errors(
        self, battle_orchestrator, sample_world_state
    ):
        """Test concurrent intent processing with some agent errors."""
        battle_orchestrator.world_state = sample_world_state

        # Mix of successful decisions and errors
        agent_decisions = {
            "team_a_agent_1": AgentDecision(
                primary_action=MoveAction(target_position=(70.0, 70.0), reason="Move"),
                confidence=0.8,
                strategic_assessment="Success",
            ),
            "team_a_agent_2": Exception("Decision error"),
            "team_b_agent_1": AgentDecision(
                primary_action=AttackAction(
                    target_agent_id="team_a_agent_1", reason="Counter attack"
                ),
                confidence=0.7,
                strategic_assessment="Counter",
            ),
        }

        # Mock orchestrator submit_intents method
        battle_orchestrator.orchestrator.submit_intents = AsyncMock()

        # Test concurrent intent processing
        await battle_orchestrator._process_concurrent_intents(agent_decisions)

        # Verify submit_intents was called with only successful decisions
        battle_orchestrator.orchestrator.submit_intents.assert_called_once()
        call_args = battle_orchestrator.orchestrator.submit_intents.call_args

        submitted_intents = call_args[0][0]
        assert len(submitted_intents) == 2  # Only successful agents

        # Verify only successful agents' intents were submitted
        agent_ids_in_intents = [intent["agent_id"] for intent in submitted_intents]
        assert "team_a_agent_1" in agent_ids_in_intents
        assert "team_b_agent_1" in agent_ids_in_intents
        assert "team_a_agent_2" not in agent_ids_in_intents  # Error case excluded

    @pytest.mark.asyncio
    async def test_process_concurrent_decisions_full_flow(
        self, battle_orchestrator, sample_world_state
    ):
        """Test the complete concurrent decision processing flow."""
        battle_orchestrator.world_state = sample_world_state

        # Mock agent handles and observation policies
        for agent_id in sample_world_state.agents.keys():
            mock_handle = Mock()
            mock_policy = Mock()
            mock_observation = Mock()
            mock_observation.visible_entities = {"test": "data"}
            mock_observation.visible_relationships = {}
            mock_observation.context_digest = "test_digest"
            mock_observation.view_seq = 1

            battle_orchestrator.orchestrator.agent_handles[agent_id] = mock_handle
            battle_orchestrator.orchestrator.observation_policies[agent_id] = (
                mock_policy
            )
            mock_policy.filter_world_state.return_value = mock_observation

        # Mock AI decisions for all agents
        def make_decision_side_effect(
            agent_id, observation, world_state, team_context=None
        ):
            return AgentDecision(
                primary_action=MoveAction(
                    target_position=(100.0, 100.0), reason=f"Move for {agent_id}"
                ),
                confidence=0.8,
                strategic_assessment=f"Assessment for {agent_id}",
            )

        battle_orchestrator.ai_decision_maker.make_decision = AsyncMock(
            side_effect=make_decision_side_effect
        )

        # Mock orchestrator submit_intents method
        battle_orchestrator.orchestrator.submit_intents = AsyncMock()

        # Test full concurrent processing
        results = await battle_orchestrator.process_concurrent_decisions()

        # Verify all living agents were processed
        living_agents = [
            agent_id
            for agent_id, agent in sample_world_state.agents.items()
            if agent.is_alive()
        ]
        assert len(results) == len(living_agents)

        # Verify all results are AgentDecision objects
        for agent_id, decision in results.items():
            assert isinstance(decision, AgentDecision)
            assert isinstance(decision.primary_action, MoveAction)

        # Verify AI decision maker was called for each agent
        assert battle_orchestrator.ai_decision_maker.make_decision.call_count == len(
            living_agents
        )

        # Verify intents were submitted
        battle_orchestrator.orchestrator.submit_intents.assert_called_once()

    @pytest.mark.asyncio
    async def test_deterministic_ordering(
        self, battle_orchestrator, sample_world_state
    ):
        """Test that agent processing follows deterministic ordering."""
        battle_orchestrator.world_state = sample_world_state

        # Mock agent handles and observation policies
        agent_ids = list(sample_world_state.agents.keys())
        for agent_id in agent_ids:
            mock_handle = Mock()
            mock_policy = Mock()
            mock_observation = Mock()
            mock_observation.visible_entities = {"test": "data"}
            mock_observation.visible_relationships = {}
            mock_observation.context_digest = "test_digest"
            mock_observation.view_seq = 1

            battle_orchestrator.orchestrator.agent_handles[agent_id] = mock_handle
            battle_orchestrator.orchestrator.observation_policies[agent_id] = (
                mock_policy
            )
            mock_policy.filter_world_state.return_value = mock_observation

        # Track the order of AI decision calls
        call_order = []

        def make_decision_side_effect(
            agent_id, observation, world_state, team_context=None
        ):
            call_order.append(agent_id)
            return AgentDecision(
                primary_action=MoveAction(
                    target_position=(100.0, 100.0), reason=f"Move for {agent_id}"
                ),
                confidence=0.8,
                strategic_assessment=f"Assessment for {agent_id}",
            )

        battle_orchestrator.ai_decision_maker.make_decision = AsyncMock(
            side_effect=make_decision_side_effect
        )

        # Mock orchestrator submit_intents method
        battle_orchestrator.orchestrator.submit_intents = AsyncMock()

        # Run multiple times to verify consistent ordering
        for _ in range(3):
            call_order.clear()
            await battle_orchestrator.process_concurrent_decisions()

            # Verify agents were processed in sorted order
            expected_order = sorted(
                [
                    agent_id
                    for agent_id, agent in sample_world_state.agents.items()
                    if agent.is_alive()
                ]
            )
            assert call_order == expected_order

    @pytest.mark.asyncio
    async def test_concurrent_processing_with_dead_agents(
        self, battle_orchestrator, sample_world_state
    ):
        """Test that dead agents are excluded from concurrent processing."""
        # Kill one agent
        sample_world_state.agents["team_a_agent_1"].health = 0
        sample_world_state.agents["team_a_agent_1"].status = AgentStatus.DEAD

        battle_orchestrator.world_state = sample_world_state

        # Mock agent handles and observation policies for living agents only
        living_agents = [
            agent_id
            for agent_id, agent in sample_world_state.agents.items()
            if agent.is_alive()
        ]

        for agent_id in living_agents:
            mock_handle = Mock()
            mock_policy = Mock()
            mock_observation = Mock()
            mock_observation.visible_entities = {"test": "data"}
            mock_observation.visible_relationships = {}
            mock_observation.context_digest = "test_digest"
            mock_observation.view_seq = 1

            battle_orchestrator.orchestrator.agent_handles[agent_id] = mock_handle
            battle_orchestrator.orchestrator.observation_policies[agent_id] = (
                mock_policy
            )
            mock_policy.filter_world_state.return_value = mock_observation

        # Mock AI decision maker
        battle_orchestrator.ai_decision_maker.make_decision = AsyncMock(
            return_value=AgentDecision(
                primary_action=MoveAction(
                    target_position=(100.0, 100.0), reason="Move"
                ),
                confidence=0.8,
                strategic_assessment="Assessment",
            )
        )

        # Mock orchestrator submit_intents method
        battle_orchestrator.orchestrator.submit_intents = AsyncMock()

        # Test concurrent processing
        results = await battle_orchestrator.process_concurrent_decisions()

        # Verify only living agents were processed
        assert len(results) == len(living_agents)
        assert "team_a_agent_1" not in results  # Dead agent excluded

        # Verify AI decision maker was called only for living agents
        assert battle_orchestrator.ai_decision_maker.make_decision.call_count == len(
            living_agents
        )

    @pytest.mark.asyncio
    async def test_concurrent_processing_thread_safety(
        self, battle_orchestrator, sample_world_state
    ):
        """Test that concurrent processing is thread-safe with processing lock."""
        battle_orchestrator.world_state = sample_world_state

        # Mock agent handles and observation policies
        for agent_id in sample_world_state.agents.keys():
            mock_handle = Mock()
            mock_policy = Mock()
            mock_observation = Mock()
            mock_observation.visible_entities = {"test": "data"}
            mock_observation.visible_relationships = {}
            mock_observation.context_digest = "test_digest"
            mock_observation.view_seq = 1

            battle_orchestrator.orchestrator.agent_handles[agent_id] = mock_handle
            battle_orchestrator.orchestrator.observation_policies[agent_id] = (
                mock_policy
            )
            mock_policy.filter_world_state.return_value = mock_observation

        # Mock AI decision maker with delay to test concurrency
        async def delayed_decision(
            agent_id, observation, world_state, team_context=None
        ):
            await asyncio.sleep(0.1)  # Small delay
            return AgentDecision(
                primary_action=MoveAction(
                    target_position=(100.0, 100.0), reason=f"Move for {agent_id}"
                ),
                confidence=0.8,
                strategic_assessment=f"Assessment for {agent_id}",
            )

        battle_orchestrator.ai_decision_maker.make_decision = delayed_decision

        # Mock orchestrator submit_intents method
        battle_orchestrator.orchestrator.submit_intents = AsyncMock()

        # Start multiple concurrent processing tasks
        tasks = [battle_orchestrator.process_concurrent_decisions() for _ in range(3)]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Verify all tasks completed successfully
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert len(result) > 0  # Should have processed some agents

        # Verify tick counter incremented correctly (should be 3 due to lock)
        assert battle_orchestrator._current_tick == 3
