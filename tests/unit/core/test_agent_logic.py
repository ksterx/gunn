"""Unit tests for asynchronous agent loop infrastructure.

Tests the AsyncAgentLogic interface, AgentHandle async loop methods,
and the overall observe-think-act loop behavior.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from gunn.core.agent_logic import AsyncAgentLogic, SimpleAgentLogic
from gunn.core.orchestrator import AgentHandle, Orchestrator, OrchestratorConfig
from gunn.policies.observation import ObservationPolicy, PolicyConfig
from gunn.schemas.messages import View, WorldState
from gunn.schemas.types import Effect, Intent


class MockObservationPolicy(ObservationPolicy):
    """Mock observation policy for testing."""

    def __init__(self):
        config = PolicyConfig()
        super().__init__(config)

    def filter_world_state(self, world_state: WorldState, agent_id: str) -> View:
        """Return a simple view with all entities visible."""
        return View(
            agent_id=agent_id,
            view_seq=0,
            visible_entities=world_state.entities,
            visible_relationships=world_state.relationships,
            context_digest="mock_digest",
        )

    def should_observe_event(
        self, effect: Effect, agent_id: str, world_state: WorldState
    ) -> bool:
        """Always return True for testing."""
        return True


class MockAgentLogic(AsyncAgentLogic):
    """Mock agent logic for testing."""

    def __init__(self):
        self.observations_processed = []
        self.intents_generated = []
        self.start_called = False
        self.stop_called = False
        self.error_called = False
        self.should_continue_on_error = True
        self.should_generate_intent = False
        self.process_delay = 0.0

    async def process_observation(
        self, observation: View, agent_id: str
    ) -> Intent | None:
        await asyncio.sleep(self.process_delay)
        self.observations_processed.append((observation, agent_id))

        if self.should_generate_intent:
            intent = {
                "kind": "Speak",
                "payload": {
                    "text": f"Test message from {agent_id}",
                    "agent_id": agent_id,
                },
                "context_seq": observation.view_seq,
                "req_id": str(uuid.uuid4()),
                "agent_id": agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }
            self.intents_generated.append(intent)
            return intent

        return None

    async def on_loop_start(self, agent_id: str) -> None:
        self.start_called = True

    async def on_loop_stop(self, agent_id: str) -> None:
        self.stop_called = True

    async def on_error(self, agent_id: str, error: Exception) -> bool:
        self.error_called = True
        return self.should_continue_on_error


@pytest.fixture
async def orchestrator():
    """Create a test orchestrator."""
    config = OrchestratorConfig(use_in_memory_dedup=True)
    orch = Orchestrator(config, world_id="test_world")
    await orch.initialize()
    yield orch
    await orch.shutdown()


@pytest.fixture
async def agent_handle(orchestrator):
    """Create a test agent handle."""
    policy = MockObservationPolicy()
    return await orchestrator.register_agent("test_agent", policy)


class TestAsyncAgentLogic:
    """Test the AsyncAgentLogic abstract base class."""

    def test_abstract_methods(self):
        """Test that AsyncAgentLogic cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AsyncAgentLogic()

    async def test_mock_agent_logic(self):
        """Test the mock agent logic implementation."""
        logic = MockAgentLogic()

        # Test default behavior
        view = View(
            agent_id="test_agent",
            view_seq=1,
            visible_entities={},
            visible_relationships={},
            context_digest="test_digest",
        )

        result = await logic.process_observation(view, "test_agent")
        assert result is None
        assert len(logic.observations_processed) == 1
        assert logic.observations_processed[0] == (view, "test_agent")

        # Test intent generation
        logic.should_generate_intent = True
        result = await logic.process_observation(view, "test_agent")
        assert result is not None
        assert result["kind"] == "Speak"
        assert result["agent_id"] == "test_agent"
        assert len(logic.intents_generated) == 1

        # Test lifecycle methods
        await logic.on_loop_start("test_agent")
        assert logic.start_called

        await logic.on_loop_stop("test_agent")
        assert logic.stop_called

        # Test error handling
        should_continue = await logic.on_error("test_agent", Exception("test error"))
        assert logic.error_called
        assert should_continue == logic.should_continue_on_error


class TestSimpleAgentLogic:
    """Test the SimpleAgentLogic implementation."""

    async def test_simple_agent_logic_creation(self):
        """Test creating SimpleAgentLogic with different probabilities."""
        logic = SimpleAgentLogic(action_probability=0.5)
        assert logic.action_probability == 0.5
        assert logic._observation_count == 0

    async def test_simple_agent_logic_observation_processing(self):
        """Test observation processing with SimpleAgentLogic."""
        # Test with 0% probability (never acts)
        logic = SimpleAgentLogic(action_probability=0.0)
        view = View(
            agent_id="test_agent",
            view_seq=1,
            visible_entities={},
            visible_relationships={},
            context_digest="test_digest",
        )

        result = await logic.process_observation(view, "test_agent")
        assert result is None
        assert logic._observation_count == 1

        # Test with 100% probability (always acts)
        logic = SimpleAgentLogic(action_probability=1.0)
        result = await logic.process_observation(view, "test_agent")
        assert result is not None
        assert result["kind"] == "Speak"
        assert "Hello from test_agent" in result["payload"]["text"]

    async def test_simple_agent_logic_lifecycle(self):
        """Test SimpleAgentLogic lifecycle methods."""
        logic = SimpleAgentLogic()

        # Test loop start resets counter
        logic._observation_count = 5
        await logic.on_loop_start("test_agent")
        assert logic._observation_count == 0

        # Test error handling continues
        should_continue = await logic.on_error("test_agent", Exception("test"))
        assert should_continue is True


class TestAgentHandleAsyncMethods:
    """Test the async loop methods added to AgentHandle."""

    async def test_get_current_observation(self, agent_handle):
        """Test get_current_observation method."""
        # Mock the observation policy
        with patch.object(
            agent_handle.orchestrator.observation_policies[agent_handle.agent_id],
            "filter_world_state",
        ) as mock_filter:
            expected_view = View(
                agent_id=agent_handle.agent_id,
                view_seq=0,
                visible_entities={"entity1": {"type": "test"}},
                visible_relationships={},
                context_digest="test_digest",
            )
            mock_filter.return_value = expected_view

            result = await agent_handle.get_current_observation()
            assert result == expected_view
            mock_filter.assert_called_once_with(
                agent_handle.orchestrator.world_state, agent_handle.agent_id
            )

    async def test_get_current_observation_unregistered_agent(self):
        """Test get_current_observation with unregistered agent."""
        config = OrchestratorConfig(use_in_memory_dedup=True)
        orchestrator = Orchestrator(config)
        handle = AgentHandle("unregistered_agent", orchestrator)

        with pytest.raises(
            RuntimeError, match="Agent unregistered_agent is not registered"
        ):
            await handle.get_current_observation()

    async def test_wait_for_action_completion_success(self, agent_handle):
        """Test _wait_for_action_completion with successful completion."""
        req_id = "test_req_123"

        # Mock wait_effect_applied to succeed
        with patch.object(
            agent_handle.orchestrator, "wait_effect_applied", new_callable=AsyncMock
        ) as mock_wait:
            mock_effect = {"uuid": "test", "kind": "Speak", "req_id": req_id}
            mock_wait.return_value = mock_effect

            await agent_handle._wait_for_action_completion(req_id)
            mock_wait.assert_called_once_with(req_id, timeout=30.0)

    async def test_wait_for_action_completion_timeout(self, agent_handle):
        """Test _wait_for_action_completion with timeout."""
        req_id = "test_req_123"

        # Mock wait_effect_applied to timeout
        with patch.object(
            agent_handle.orchestrator, "wait_effect_applied", new_callable=AsyncMock
        ) as mock_wait:
            mock_wait.side_effect = TimeoutError()

            # Should not raise, just log warning
            await agent_handle._wait_for_action_completion(req_id)
            mock_wait.assert_called_once_with(req_id, timeout=30.0)

    async def test_wait_for_action_completion_error(self, agent_handle):
        """Test _wait_for_action_completion with other errors."""
        req_id = "test_req_123"

        # Mock wait_effect_applied to raise error
        with patch.object(
            agent_handle.orchestrator, "wait_effect_applied", new_callable=AsyncMock
        ) as mock_wait:
            mock_wait.side_effect = Exception("Test error")

            # Should not raise, just log warning
            await agent_handle._wait_for_action_completion(req_id)
            mock_wait.assert_called_once_with(req_id, timeout=30.0)

    def test_stop_async_loop(self, agent_handle):
        """Test stop_async_loop method."""
        agent_handle._running = True
        agent_handle.stop_async_loop()
        assert agent_handle._running is False


class TestAsyncLoopExecution:
    """Test the complete async loop execution."""

    async def test_run_async_loop_basic_flow(self, agent_handle):
        """Test basic async loop execution flow."""
        logic = MockAgentLogic()

        # Mock get_current_observation
        mock_view = View(
            agent_id=agent_handle.agent_id,
            view_seq=1,
            visible_entities={},
            visible_relationships={},
            context_digest="test_digest",
        )

        with patch.object(
            agent_handle, "get_current_observation", new_callable=AsyncMock
        ) as mock_obs:
            mock_obs.return_value = mock_view

            # Start the loop in a task
            loop_task = asyncio.create_task(agent_handle.run_async_loop(logic))

            # Let it run for a short time
            await asyncio.sleep(0.2)

            # Stop the loop
            agent_handle.stop_async_loop()

            # Wait for loop to finish
            await loop_task

        # Verify lifecycle methods were called
        assert logic.start_called
        assert logic.stop_called

        # Verify observations were processed
        assert len(logic.observations_processed) > 0
        assert logic.observations_processed[0][0] == mock_view
        assert logic.observations_processed[0][1] == agent_handle.agent_id

    async def test_run_async_loop_with_intents(self, agent_handle):
        """Test async loop execution with intent generation."""
        logic = MockAgentLogic()
        logic.should_generate_intent = True

        # Mock methods
        mock_view = View(
            agent_id=agent_handle.agent_id,
            view_seq=1,
            visible_entities={},
            visible_relationships={},
            context_digest="test_digest",
        )

        with (
            patch.object(
                agent_handle, "get_current_observation", new_callable=AsyncMock
            ) as mock_obs,
            patch.object(
                agent_handle, "submit_intent", new_callable=AsyncMock
            ) as mock_submit,
            patch.object(
                agent_handle, "_wait_for_action_completion", new_callable=AsyncMock
            ) as mock_wait,
        ):
            mock_obs.return_value = mock_view
            mock_submit.return_value = "test_req_123"

            # Start the loop in a task
            loop_task = asyncio.create_task(agent_handle.run_async_loop(logic))

            # Let it run for a short time
            await asyncio.sleep(0.2)

            # Stop the loop
            agent_handle.stop_async_loop()

            # Wait for loop to finish
            await loop_task

        # Verify intents were submitted
        assert len(logic.intents_generated) > 0
        assert mock_submit.call_count > 0
        assert mock_wait.call_count > 0

    async def test_run_async_loop_error_handling_continue(self, agent_handle):
        """Test async loop error handling when continuing."""
        logic = MockAgentLogic()
        logic.should_continue_on_error = True

        # Mock get_current_observation to raise error initially
        call_count = 0

        async def mock_get_observation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Test error")
            return View(
                agent_id=agent_handle.agent_id,
                view_seq=1,
                visible_entities={},
                visible_relationships={},
                context_digest="test_digest",
            )

        with patch.object(
            agent_handle, "get_current_observation", side_effect=mock_get_observation
        ):
            # Start the loop in a task
            loop_task = asyncio.create_task(agent_handle.run_async_loop(logic))

            # Let it run for a short time
            await asyncio.sleep(0.2)

            # Stop the loop
            agent_handle.stop_async_loop()

            # Wait for loop to finish
            await loop_task

        # Verify error handling was called and loop continued
        assert logic.error_called
        assert logic.start_called
        assert logic.stop_called
        assert call_count > 1  # Should have retried after error

    async def test_run_async_loop_error_handling_stop(self, agent_handle):
        """Test async loop error handling when stopping."""
        logic = MockAgentLogic()
        logic.should_continue_on_error = False

        # Mock get_current_observation to raise error
        with patch.object(
            agent_handle, "get_current_observation", new_callable=AsyncMock
        ) as mock_obs:
            mock_obs.side_effect = Exception("Test error")

            # Start and run the loop
            await agent_handle.run_async_loop(logic)

        # Verify error handling was called and loop stopped
        assert logic.error_called
        assert logic.start_called
        assert logic.stop_called
        assert not agent_handle._running

    async def test_run_async_loop_unregistered_agent(self):
        """Test run_async_loop with unregistered agent."""
        config = OrchestratorConfig(use_in_memory_dedup=True)
        orchestrator = Orchestrator(config)
        handle = AgentHandle("unregistered_agent", orchestrator)
        logic = MockAgentLogic()

        with pytest.raises(
            RuntimeError, match="Agent unregistered_agent is not registered"
        ):
            await handle.run_async_loop(logic)


class TestAsyncLoopTiming:
    """Test timing aspects of the async loop."""

    async def test_loop_timing_consistency(self, agent_handle):
        """Test that loop timing is consistent."""
        logic = MockAgentLogic()
        logic.process_delay = 0.05  # 50ms processing delay

        mock_view = View(
            agent_id=agent_handle.agent_id,
            view_seq=1,
            visible_entities={},
            visible_relationships={},
            context_digest="test_digest",
        )

        with patch.object(
            agent_handle, "get_current_observation", new_callable=AsyncMock
        ) as mock_obs:
            mock_obs.return_value = mock_view

            start_time = asyncio.get_event_loop().time()

            # Start the loop in a task
            loop_task = asyncio.create_task(agent_handle.run_async_loop(logic))

            # Let it run for a measured time
            await asyncio.sleep(0.3)

            # Stop the loop
            agent_handle.stop_async_loop()

            # Wait for loop to finish
            await loop_task

            end_time = asyncio.get_event_loop().time()
            elapsed = end_time - start_time

        # Verify reasonable number of observations processed given timing
        # With 50ms processing + 100ms sleep, should process ~2-3 observations in 300ms
        assert len(logic.observations_processed) >= 1
        assert len(logic.observations_processed) <= 5  # Allow some variance
        assert elapsed >= 0.25  # Should take at least the sleep time

    async def test_loop_stops_promptly(self, agent_handle):
        """Test that loop stops promptly when requested."""
        logic = MockAgentLogic()

        mock_view = View(
            agent_id=agent_handle.agent_id,
            view_seq=1,
            visible_entities={},
            visible_relationships={},
            context_digest="test_digest",
        )

        with patch.object(
            agent_handle, "get_current_observation", new_callable=AsyncMock
        ) as mock_obs:
            mock_obs.return_value = mock_view

            # Start the loop in a task
            loop_task = asyncio.create_task(agent_handle.run_async_loop(logic))

            # Let it run briefly
            await asyncio.sleep(0.05)

            # Stop the loop and measure how long it takes
            stop_start = asyncio.get_event_loop().time()
            agent_handle.stop_async_loop()
            await loop_task
            stop_end = asyncio.get_event_loop().time()

            stop_time = stop_end - stop_start

        # Loop should stop within reasonable time (less than 200ms)
        assert stop_time < 0.2
        assert not agent_handle._running


if __name__ == "__main__":
    pytest.main([__file__])
