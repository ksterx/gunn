"""Unit tests for RL-style facade interface."""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.facades.rl import RLFacade
from gunn.policies.observation import ObservationPolicy
from gunn.schemas.types import Intent
from gunn.utils.errors import (
    BackpressureError,
    QuotaExceededError,
    StaleContextError,
)


class TestRLFacade:
    """Test suite for RLFacade class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return OrchestratorConfig(
            max_agents=10,
            staleness_threshold=0,
            use_in_memory_dedup=True,
        )

    @pytest.fixture
    def observation_policy(self):
        """Create test observation policy."""
        policy = MagicMock(spec=ObservationPolicy)
        policy.filter_world_state = MagicMock()
        policy.should_observe_event = MagicMock(return_value=True)
        policy.calculate_observation_delta = MagicMock()
        return policy

    @pytest.fixture
    async def orchestrator(self, config):
        """Create test orchestrator."""
        orch = Orchestrator(config, world_id="test_world")
        await orch.initialize()
        return orch

    @pytest.fixture
    async def rl_facade(self, orchestrator):
        """Create test RL facade."""
        facade = RLFacade(orchestrator=orchestrator, timeout_seconds=5.0)
        await facade.initialize()
        return facade

    @pytest.fixture
    async def registered_agent(self, rl_facade, observation_policy):
        """Create and register a test agent."""
        agent_id = "test_agent"
        await rl_facade.register_agent(agent_id, observation_policy)
        return agent_id

    def test_init_with_orchestrator(self, orchestrator):
        """Test RLFacade initialization with existing orchestrator."""
        facade = RLFacade(orchestrator=orchestrator, timeout_seconds=10.0)

        assert facade._orchestrator is orchestrator
        assert facade.world_id == "test_world"
        assert facade.timeout_seconds == 10.0

    def test_init_with_config(self, config):
        """Test RLFacade initialization with config."""
        facade = RLFacade(config=config, world_id="new_world", timeout_seconds=15.0)

        assert facade._orchestrator is not None
        assert facade.world_id == "new_world"
        assert facade.timeout_seconds == 15.0

    def test_init_without_orchestrator_or_config(self):
        """Test RLFacade initialization fails without orchestrator or config."""
        with pytest.raises(
            ValueError, match="Either orchestrator or config must be provided"
        ):
            RLFacade()

    async def test_initialize(self, config):
        """Test facade initialization."""
        facade = RLFacade(config=config)

        with patch.object(
            facade._orchestrator, "initialize", new_callable=AsyncMock
        ) as mock_init:
            await facade.initialize()
            mock_init.assert_called_once()

    async def test_register_agent(self, rl_facade, observation_policy):
        """Test agent registration."""
        agent_id = "new_agent"

        with patch.object(
            rl_facade._orchestrator, "register_agent", new_callable=AsyncMock
        ) as mock_register:
            await rl_facade.register_agent(agent_id, observation_policy)
            mock_register.assert_called_once_with(agent_id, observation_policy)

    async def test_observe_success(self, rl_facade, registered_agent):
        """Test successful observation retrieval."""
        agent_id = registered_agent

        # Mock observation delta
        mock_delta = {
            "view_seq": 42,
            "patches": [{"op": "add", "path": "/test", "value": "data"}],
            "context_digest": "test_digest",
            "schema_version": "1.0.0",
        }

        # Mock agent handle
        mock_handle = MagicMock()
        mock_handle.next_observation = AsyncMock(return_value=mock_delta)
        rl_facade._orchestrator.agent_handles[agent_id] = mock_handle

        result = await rl_facade.observe(agent_id)

        assert result == mock_delta
        mock_handle.next_observation.assert_called_once()

    async def test_observe_unregistered_agent(self, rl_facade):
        """Test observation fails for unregistered agent."""
        with pytest.raises(ValueError, match="Agent unknown_agent is not registered"):
            await rl_facade.observe("unknown_agent")

    async def test_observe_timeout(self, rl_facade, registered_agent):
        """Test observation timeout handling."""
        agent_id = registered_agent

        # Mock agent handle with slow response
        mock_handle = MagicMock()
        mock_handle.next_observation = AsyncMock(side_effect=asyncio.TimeoutError())
        rl_facade._orchestrator.agent_handles[agent_id] = mock_handle

        with pytest.raises(TimeoutError, match="timed out after"):
            await rl_facade.observe(agent_id)

    async def test_step_success(self, rl_facade, registered_agent):
        """Test successful step execution."""
        agent_id = registered_agent
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 0,
            "req_id": "test_req",
            "agent_id": agent_id,
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Mock orchestrator methods
        rl_facade._orchestrator.submit_intent = AsyncMock(return_value="test_req")
        rl_facade._orchestrator._global_seq = 100
        rl_facade._orchestrator._current_sim_time = MagicMock(return_value=123.45)

        # Mock agent handle
        mock_handle = MagicMock()
        mock_observation = {
            "view_seq": 101,
            "patches": [{"op": "add", "path": "/message", "value": "Hello"}],
            "context_digest": "new_digest",
            "schema_version": "1.0.0",
        }
        mock_handle.next_observation = AsyncMock(return_value=mock_observation)
        rl_facade._orchestrator.agent_handles[agent_id] = mock_handle

        effect, observation = await rl_facade.step(agent_id, intent)

        # Verify effect structure
        assert effect["kind"] == "Speak"
        assert effect["payload"] == {"text": "Hello"}
        assert effect["source_id"] == agent_id
        assert "uuid" in effect
        assert "global_seq" in effect
        assert "sim_time" in effect

        # Verify observation
        assert observation == mock_observation

        # Verify orchestrator was called
        rl_facade._orchestrator.submit_intent.assert_called_once_with(intent)

    async def test_step_auto_fills_req_id(self, rl_facade, registered_agent):
        """Test step auto-fills missing req_id."""
        agent_id = registered_agent
        intent: Intent = {
            "kind": "Move",
            "payload": {"x": 10, "y": 20},
            "context_seq": 0,
            "req_id": "",  # Empty req_id
            "agent_id": agent_id,
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Mock orchestrator and agent handle
        rl_facade._orchestrator.submit_intent = AsyncMock(return_value="auto_req")
        rl_facade._orchestrator._global_seq = 100
        rl_facade._orchestrator._current_sim_time = MagicMock(return_value=123.45)

        mock_handle = MagicMock()
        mock_handle.next_observation = AsyncMock(
            return_value={
                "view_seq": 101,
                "patches": [],
                "context_digest": "digest",
                "schema_version": "1.0.0",
            }
        )
        rl_facade._orchestrator.agent_handles[agent_id] = mock_handle

        await rl_facade.step(agent_id, intent)

        # Verify req_id was auto-filled
        submitted_intent = rl_facade._orchestrator.submit_intent.call_args[0][0]
        assert submitted_intent["req_id"].startswith("rl_step_")
        assert len(submitted_intent["req_id"]) > 8

    async def test_step_auto_fills_agent_id(self, rl_facade, registered_agent):
        """Test step auto-fills missing agent_id."""
        agent_id = registered_agent
        intent: Intent = {
            "kind": "Interact",
            "payload": {"target": "object1"},
            "context_seq": 0,
            "req_id": "test_req",
            "agent_id": "",  # Empty agent_id
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Mock orchestrator and agent handle
        rl_facade._orchestrator.submit_intent = AsyncMock(return_value="test_req")
        rl_facade._orchestrator._global_seq = 100
        rl_facade._orchestrator._current_sim_time = MagicMock(return_value=123.45)

        mock_handle = MagicMock()
        mock_handle.next_observation = AsyncMock(
            return_value={
                "view_seq": 101,
                "patches": [],
                "context_digest": "digest",
                "schema_version": "1.0.0",
            }
        )
        rl_facade._orchestrator.agent_handles[agent_id] = mock_handle

        await rl_facade.step(agent_id, intent)

        # Verify agent_id was auto-filled
        submitted_intent = rl_facade._orchestrator.submit_intent.call_args[0][0]
        assert submitted_intent["agent_id"] == agent_id

    async def test_step_agent_id_mismatch(self, rl_facade, registered_agent):
        """Test step fails when intent agent_id doesn't match provided agent_id."""
        agent_id = registered_agent
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 0,
            "req_id": "test_req",
            "agent_id": "different_agent",  # Mismatched agent_id
            "priority": 0,
            "schema_version": "1.0.0",
        }

        with pytest.raises(
            ValueError,
            match="Intent agent_id different_agent does not match provided agent_id",
        ):
            await rl_facade.step(agent_id, intent)

    async def test_step_unregistered_agent(self, rl_facade):
        """Test step fails for unregistered agent."""
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 0,
            "req_id": "test_req",
            "agent_id": "unknown_agent",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        with pytest.raises(ValueError, match="Agent unknown_agent is not registered"):
            await rl_facade.step("unknown_agent", intent)

    async def test_step_stale_context_error(self, rl_facade, registered_agent):
        """Test step handles StaleContextError."""
        agent_id = registered_agent
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 0,
            "req_id": "test_req",
            "agent_id": agent_id,
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Mock orchestrator to raise StaleContextError
        rl_facade._orchestrator.submit_intent = AsyncMock(
            side_effect=StaleContextError("test_req", 0, 10, 5)
        )

        with pytest.raises(StaleContextError):
            await rl_facade.step(agent_id, intent)

    async def test_step_quota_exceeded_error(self, rl_facade, registered_agent):
        """Test step handles QuotaExceededError."""
        agent_id = registered_agent
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 0,
            "req_id": "test_req",
            "agent_id": agent_id,
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Mock orchestrator to raise QuotaExceededError
        rl_facade._orchestrator.submit_intent = AsyncMock(
            side_effect=QuotaExceededError(agent_id, "intents_per_minute", 60, 61)
        )

        with pytest.raises(QuotaExceededError):
            await rl_facade.step(agent_id, intent)

    async def test_step_backpressure_error(self, rl_facade, registered_agent):
        """Test step handles BackpressureError."""
        agent_id = registered_agent
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 0,
            "req_id": "test_req",
            "agent_id": agent_id,
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Mock orchestrator to raise BackpressureError
        rl_facade._orchestrator.submit_intent = AsyncMock(
            side_effect=BackpressureError(agent_id, "agent_queue", 100, 50, "defer")
        )

        with pytest.raises(BackpressureError):
            await rl_facade.step(agent_id, intent)

    async def test_step_timeout(self, rl_facade, registered_agent):
        """Test step timeout handling."""
        agent_id = registered_agent
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 0,
            "req_id": "test_req",
            "agent_id": agent_id,
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Mock orchestrator with slow response
        async def slow_submit_intent(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return "test_req"

        rl_facade._orchestrator.submit_intent = slow_submit_intent
        rl_facade.timeout_seconds = 0.1  # Very short timeout

        with pytest.raises(TimeoutError, match="timed out after"):
            await rl_facade.step(agent_id, intent)

    async def test_step_cancels_previous_pending(self, rl_facade, registered_agent):
        """Test step cancels previous pending step for same agent."""
        agent_id = registered_agent
        intent1: Intent = {
            "kind": "Speak",
            "payload": {"text": "First"},
            "context_seq": 0,
            "req_id": "req1",
            "agent_id": agent_id,
            "priority": 0,
            "schema_version": "1.0.0",
        }
        intent2: Intent = {
            "kind": "Speak",
            "payload": {"text": "Second"},
            "context_seq": 0,
            "req_id": "req2",
            "agent_id": agent_id,
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Mock slow first step
        async def slow_execute_step(*args, **kwargs):
            await asyncio.sleep(1)
            return ({}, {})

        with patch.object(rl_facade, "_execute_step", side_effect=slow_execute_step):
            # Start first step (don't await)
            task1 = asyncio.create_task(rl_facade.step(agent_id, intent1))

            # Small delay to ensure first step starts
            await asyncio.sleep(0.01)

            # Start second step - should cancel first
            with patch.object(rl_facade, "_execute_step", return_value=({}, {})):
                await rl_facade.step(agent_id, intent2)

            # First step should be cancelled
            with pytest.raises(asyncio.CancelledError):
                await task1

    async def test_shutdown(self, rl_facade, registered_agent):
        """Test facade shutdown."""
        agent_id = registered_agent

        # Create a pending step
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 0,
            "req_id": "test_req",
            "agent_id": agent_id,
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Mock slow step execution
        async def slow_execute_step(*args, **kwargs):
            await asyncio.sleep(1)
            return ({}, {})

        with patch.object(rl_facade, "_execute_step", side_effect=slow_execute_step):
            # Start step (don't await)
            task = asyncio.create_task(rl_facade.step(agent_id, intent))

            # Small delay to ensure step starts
            await asyncio.sleep(0.01)

            # Shutdown should cancel pending steps
            await rl_facade.shutdown()

            # Step should be cancelled
            with pytest.raises(asyncio.CancelledError):
                await task

        # Pending steps should be cleared
        assert len(rl_facade._pending_steps) == 0

    def test_get_orchestrator(self, rl_facade):
        """Test getting underlying orchestrator."""
        orchestrator = rl_facade.get_orchestrator()
        assert orchestrator is rl_facade._orchestrator

    def test_set_timeout(self, rl_facade):
        """Test setting timeout."""
        rl_facade.set_timeout(10.0)
        assert rl_facade.timeout_seconds == 10.0

    def test_set_timeout_invalid(self, rl_facade):
        """Test setting invalid timeout."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            rl_facade.set_timeout(0)

        with pytest.raises(ValueError, match="Timeout must be positive"):
            rl_facade.set_timeout(-5)

    async def test_get_agent_view_seq(self, rl_facade, registered_agent):
        """Test getting agent view sequence."""
        agent_id = registered_agent

        # Mock agent handle
        mock_handle = MagicMock()
        mock_handle.get_view_seq = MagicMock(return_value=42)
        rl_facade._orchestrator.agent_handles[agent_id] = mock_handle

        view_seq = await rl_facade.get_agent_view_seq(agent_id)

        assert view_seq == 42
        mock_handle.get_view_seq.assert_called_once()

    async def test_get_agent_view_seq_unregistered(self, rl_facade):
        """Test getting view seq for unregistered agent."""
        with pytest.raises(ValueError, match="Agent unknown_agent is not registered"):
            await rl_facade.get_agent_view_seq("unknown_agent")

    async def test_cancel_agent_step(self, rl_facade, registered_agent):
        """Test cancelling agent step."""
        agent_id = registered_agent

        # Create a mock pending task
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel = MagicMock()

        rl_facade._pending_steps[agent_id] = mock_task

        result = await rl_facade.cancel_agent_step(agent_id)

        assert result is True
        mock_task.cancel.assert_called_once()

    async def test_cancel_agent_step_no_pending(self, rl_facade, registered_agent):
        """Test cancelling agent step when no step is pending."""
        agent_id = registered_agent

        result = await rl_facade.cancel_agent_step(agent_id)

        assert result is False

    async def test_cancel_agent_step_unregistered(self, rl_facade):
        """Test cancelling step for unregistered agent."""
        with pytest.raises(ValueError, match="Agent unknown_agent is not registered"):
            await rl_facade.cancel_agent_step("unknown_agent")

    def test_repr(self, rl_facade):
        """Test string representation."""
        repr_str = repr(rl_facade)

        assert "RLFacade" in repr_str
        assert "world_id=test_world" in repr_str
        assert "timeout_seconds=5.0" in repr_str
        assert "agents=" in repr_str


class TestRLFacadeIntegration:
    """Integration tests for RLFacade with real orchestrator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return OrchestratorConfig(
            max_agents=5,
            staleness_threshold=0,
            use_in_memory_dedup=True,
            processing_idle_shutdown_ms=100,  # Quick shutdown for tests
        )

    @pytest.fixture
    def observation_policy(self):
        """Create real observation policy."""
        from gunn.policies.observation import ObservationPolicy, PolicyConfig
        from gunn.schemas.messages import View, WorldState

        class TestPolicy(ObservationPolicy):
            def __init__(self, config: PolicyConfig):
                super().__init__(config)

            def filter_world_state(
                self, world_state: WorldState, agent_id: str
            ) -> View:
                return View(
                    agent_id=agent_id,
                    view_seq=0,
                    visible_entities=world_state.entities,
                    visible_relationships=world_state.relationships,
                    context_digest="test_digest",
                )

            def should_observe_event(
                self, effect, agent_id: str, world_state: WorldState
            ) -> bool:
                return True

            def calculate_observation_delta(self, old_view: View, new_view: View):
                return {
                    "view_seq": new_view.view_seq,
                    "patches": [{"op": "replace", "path": "/test", "value": "updated"}],
                    "context_digest": new_view.context_digest,
                    "schema_version": "1.0.0",
                }

        config = PolicyConfig(
            distance_limit=100.0, relationship_filter=[], field_visibility={}
        )
        return TestPolicy(config)

    async def test_full_rl_workflow(self, config, observation_policy):
        """Test complete RL workflow with real orchestrator."""
        # Create facade
        facade = RLFacade(config=config, world_id="integration_test")
        await facade.initialize()

        try:
            # Register agent
            agent_id = "test_agent"
            await facade.register_agent(agent_id, observation_policy)

            # Create intent
            intent: Intent = {
                "kind": "Speak",
                "payload": {"text": "Hello world"},
                "context_seq": 0,
                "req_id": f"test_{uuid.uuid4().hex[:8]}",
                "agent_id": agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }

            # Execute step - this tests the full integration
            effect, observation = await facade.step(agent_id, intent)

            # Verify the step completed successfully
            assert effect["kind"] == "Speak"
            assert effect["payload"]["text"] == "Hello world"
            assert effect["source_id"] == agent_id
            assert "uuid" in effect
            assert "global_seq" in effect

            # Verify observation was received
            assert observation["view_seq"] == 1
            assert "patches" in observation
            assert observation["schema_version"] == "1.0.0"

        finally:
            await facade.shutdown()

    async def test_multiple_agents_rl_facade(self, config, observation_policy):
        """Test RL facade with multiple agents."""
        facade = RLFacade(config=config, world_id="multi_agent_test")
        await facade.initialize()

        try:
            # Register multiple agents
            agents = ["agent1", "agent2", "agent3"]
            for agent_id in agents:
                await facade.register_agent(agent_id, observation_policy)

            # Verify all agents are registered
            for agent_id in agents:
                view_seq = await facade.get_agent_view_seq(agent_id)
                assert view_seq == 0  # Initial view sequence

        finally:
            await facade.shutdown()
