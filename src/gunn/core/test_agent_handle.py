"""Comprehensive unit tests for AgentHandle class.

Tests cover agent isolation, non-blocking operations, error handling,
and integration with TimedQueue for observation delivery.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from gunn.core.orchestrator import AgentHandle, Orchestrator, OrchestratorConfig
from gunn.policies.observation import DefaultObservationPolicy, PolicyConfig
from gunn.schemas.types import CancelToken, Intent, ObservationDelta
from gunn.utils.timing import TimedQueue


class TestAgentHandleInitialization:
    """Test AgentHandle initialization and basic properties."""

    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        """Create orchestrator for testing."""
        config = OrchestratorConfig(max_agents=10)
        return Orchestrator(config, world_id="test_world")

    def test_initialization(self, orchestrator: Orchestrator) -> None:
        """Test AgentHandle initialization with valid parameters."""
        handle = AgentHandle("test_agent", orchestrator)

        assert handle.agent_id == "test_agent"
        assert handle.orchestrator is orchestrator
        assert handle.view_seq == 0

    def test_initialization_empty_agent_id(self, orchestrator: Orchestrator) -> None:
        """Test AgentHandle initialization with empty agent ID."""
        # AgentHandle doesn't validate agent_id in constructor, but it should be valid
        handle = AgentHandle("", orchestrator)
        assert handle.agent_id == ""

    def test_get_view_seq_initial(self, orchestrator: Orchestrator) -> None:
        """Test initial view sequence is zero."""
        handle = AgentHandle("test_agent", orchestrator)
        assert handle.get_view_seq() == 0

    def test_get_view_seq_after_update(self, orchestrator: Orchestrator) -> None:
        """Test view sequence tracking after updates."""
        handle = AgentHandle("test_agent", orchestrator)

        # Simulate view sequence updates
        handle.view_seq = 5
        assert handle.get_view_seq() == 5

        handle.view_seq = 42
        assert handle.get_view_seq() == 42


class TestAgentHandleObservations:
    """Test observation handling through AgentHandle."""

    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        """Create orchestrator for testing."""
        config = OrchestratorConfig(max_agents=10)
        return Orchestrator(config, world_id="test_world")

    @pytest.fixture
    def observation_policy(self) -> DefaultObservationPolicy:
        """Create observation policy for testing."""
        return DefaultObservationPolicy(PolicyConfig(distance_limit=50.0))

    @pytest.mark.asyncio
    async def test_next_observation_unregistered_agent(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test next_observation raises error for unregistered agent."""
        handle = AgentHandle("unregistered_agent", orchestrator)

        with pytest.raises(RuntimeError, match="Agent unregistered_agent is not registered"):
            await handle.next_observation()

    @pytest.mark.asyncio
    async def test_next_observation_with_registered_agent(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test next_observation works with registered agent."""
        # Register agent through orchestrator
        handle = await orchestrator.register_agent("test_agent", observation_policy)

        # Create a mock observation delta
        mock_delta = {
            "view_seq": 5,
            "patches": [{"op": "add", "path": "/entities/1", "value": {"type": "player"}}],
            "context_digest": "abc123",
            "schema_version": "1.0.0",
        }

        # Put the delta in the agent's queue
        queue = orchestrator._per_agent_queues["test_agent"]
        await queue.put_in(0.001, mock_delta)  # Deliver almost immediately

        # Get the observation
        delta = await handle.next_observation()

        assert delta == mock_delta
        assert handle.view_seq == 5

    @pytest.mark.asyncio
    async def test_next_observation_view_seq_update_dict(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test view_seq is updated from dict-style delta."""
        handle = await orchestrator.register_agent("test_agent", observation_policy)

        mock_delta = {"view_seq": 10, "patches": [], "context_digest": "def456", "schema_version": "1.0.0"}

        queue = orchestrator._per_agent_queues["test_agent"]
        await queue.put_in(0.001, mock_delta)

        delta = await handle.next_observation()
        assert handle.view_seq == 10

    @pytest.mark.asyncio
    async def test_next_observation_view_seq_update_object(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test view_seq is updated from object-style delta."""
        handle = await orchestrator.register_agent("test_agent", observation_policy)

        # Create mock object with view_seq attribute
        class MockDelta:
            def __init__(self, view_seq: int):
                self.view_seq = view_seq
                self.patches = []
                self.context_digest = "ghi789"
                self.schema_version = "1.0.0"

        mock_delta = MockDelta(15)

        queue = orchestrator._per_agent_queues["test_agent"]
        await queue.put_in(0.001, mock_delta)

        delta = await handle.next_observation()
        assert handle.view_seq == 15

    @pytest.mark.asyncio
    async def test_next_observation_no_view_seq(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test next_observation handles delta without view_seq gracefully."""
        handle = await orchestrator.register_agent("test_agent", observation_policy)
        original_view_seq = handle.view_seq

        # Delta without view_seq
        mock_delta = {"patches": [], "context_digest": "jkl012"}

        queue = orchestrator._per_agent_queues["test_agent"]
        await queue.put_in(0.001, mock_delta)

        delta = await handle.next_observation()
        assert delta == mock_delta
        assert handle.view_seq == original_view_seq  # Should remain unchanged

    @pytest.mark.asyncio
    async def test_next_observation_multiple_deltas(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test receiving multiple observation deltas in sequence."""
        handle = await orchestrator.register_agent("test_agent", observation_policy)

        deltas = [
            {"view_seq": 1, "patches": [{"op": "add", "path": "/a", "value": 1}]},
            {"view_seq": 2, "patches": [{"op": "add", "path": "/b", "value": 2}]},
            {"view_seq": 3, "patches": [{"op": "add", "path": "/c", "value": 3}]},
        ]

        queue = orchestrator._per_agent_queues["test_agent"]

        # Queue all deltas with small delays
        for i, delta in enumerate(deltas):
            await queue.put_in(0.001 * (i + 1), delta)

        # Receive all deltas
        received_deltas = []
        for _ in range(3):
            delta = await handle.next_observation()
            received_deltas.append(delta)

        assert received_deltas == deltas
        assert handle.view_seq == 3  # Should have the latest view_seq


class TestAgentHandleIntentSubmission:
    """Test intent submission through AgentHandle."""

    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        """Create orchestrator for testing."""
        config = OrchestratorConfig(max_agents=10)
        return Orchestrator(config, world_id="test_world")

    @pytest.fixture
    def observation_policy(self) -> DefaultObservationPolicy:
        """Create observation policy for testing."""
        return DefaultObservationPolicy(PolicyConfig())

    @pytest.mark.asyncio
    async def test_submit_intent_success(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test successful intent submission."""
        handle = await orchestrator.register_agent("test_agent", observation_policy)

        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello world"},
            "context_seq": 1,
            "req_id": "test_req_1",
            "agent_id": "test_agent",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        req_id = await handle.submit_intent(intent)
        assert req_id == "test_req_1"

        # Verify effect was created in event log
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1
        assert entries[0].effect["kind"] == "Speak"
        assert entries[0].effect["payload"]["text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_submit_intent_delegates_to_orchestrator(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test that submit_intent properly delegates to orchestrator."""
        handle = AgentHandle("test_agent", orchestrator)

        # Mock the orchestrator's submit_intent method
        orchestrator.submit_intent = AsyncMock(return_value="mocked_req_id")

        intent: Intent = {
            "kind": "Custom",
            "payload": {"data": "test"},
            "context_seq": 0,
            "req_id": "test_req",
            "agent_id": "test_agent",
            "priority": 5,
            "schema_version": "1.0.0",
        }

        req_id = await handle.submit_intent(intent)

        assert req_id == "mocked_req_id"
        orchestrator.submit_intent.assert_called_once_with(intent)

    @pytest.mark.asyncio
    async def test_submit_intent_error_propagation(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test that errors from orchestrator are properly propagated."""
        handle = AgentHandle("test_agent", orchestrator)

        # Mock orchestrator to raise an error
        orchestrator.submit_intent = AsyncMock(side_effect=ValueError("Test error"))

        intent: Intent = {
            "kind": "Custom",
            "payload": {},
            "context_seq": 0,
            "req_id": "test_req",
            "agent_id": "test_agent",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        with pytest.raises(ValueError, match="Test error"):
            await handle.submit_intent(intent)


class TestAgentHandleCancellation:
    """Test cancellation functionality through AgentHandle."""

    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        """Create orchestrator for testing."""
        config = OrchestratorConfig(max_agents=10)
        return Orchestrator(config, world_id="test_world")

    @pytest.mark.asyncio
    async def test_cancel_existing_token(self, orchestrator: Orchestrator) -> None:
        """Test cancelling an existing cancel token."""
        handle = AgentHandle("test_agent", orchestrator)

        # Issue a cancel token
        token = orchestrator.issue_cancel_token("test_agent", "test_req")
        assert not token.cancelled

        # Cancel through handle
        await handle.cancel("test_req")

        assert token.cancelled
        assert token.reason == "user_requested"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_token(self, orchestrator: Orchestrator) -> None:
        """Test cancelling a non-existent token (should not raise error)."""
        handle = AgentHandle("test_agent", orchestrator)

        # Cancel non-existent token - should not raise error
        await handle.cancel("nonexistent_req")

    @pytest.mark.asyncio
    async def test_cancel_token_key_format(self, orchestrator: Orchestrator) -> None:
        """Test that cancel uses correct key format."""
        handle = AgentHandle("test_agent", orchestrator)

        # Issue token and verify key format
        token = orchestrator.issue_cancel_token("test_agent", "test_req")
        expected_key = ("test_world", "test_agent", "test_req")

        assert expected_key in orchestrator._cancel_tokens
        assert orchestrator._cancel_tokens[expected_key] == token

        # Cancel and verify token is cancelled
        await handle.cancel("test_req")
        assert token.cancelled

    @pytest.mark.asyncio
    async def test_cancel_multiple_tokens(self, orchestrator: Orchestrator) -> None:
        """Test cancelling multiple tokens for the same agent."""
        handle = AgentHandle("test_agent", orchestrator)

        # Issue multiple tokens
        tokens = []
        for i in range(3):
            token = orchestrator.issue_cancel_token("test_agent", f"req_{i}")
            tokens.append(token)

        # Cancel all tokens
        for i in range(3):
            await handle.cancel(f"req_{i}")

        # Verify all are cancelled
        for token in tokens:
            assert token.cancelled
            assert token.reason == "user_requested"


class TestAgentHandleIsolation:
    """Test agent isolation and non-blocking operations."""

    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        """Create orchestrator for testing."""
        config = OrchestratorConfig(max_agents=10)
        return Orchestrator(config, world_id="test_world")

    @pytest.fixture
    def observation_policy(self) -> DefaultObservationPolicy:
        """Create observation policy for testing."""
        return DefaultObservationPolicy(PolicyConfig())

    @pytest.mark.asyncio
    async def test_agent_isolation_separate_queues(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test that agents have separate observation queues."""
        # Register multiple agents
        handle_a = await orchestrator.register_agent("agent_a", observation_policy)
        handle_b = await orchestrator.register_agent("agent_b", observation_policy)

        # Verify separate queues
        queue_a = orchestrator._per_agent_queues["agent_a"]
        queue_b = orchestrator._per_agent_queues["agent_b"]

        assert queue_a is not queue_b
        assert isinstance(queue_a, TimedQueue)
        assert isinstance(queue_b, TimedQueue)

    @pytest.mark.asyncio
    async def test_agent_isolation_independent_view_seq(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test that agents maintain independent view sequences."""
        handle_a = await orchestrator.register_agent("agent_a", observation_policy)
        handle_b = await orchestrator.register_agent("agent_b", observation_policy)

        # Update view sequences independently
        handle_a.view_seq = 10
        handle_b.view_seq = 20

        assert handle_a.get_view_seq() == 10
        assert handle_b.get_view_seq() == 20

        # Verify they don't affect each other
        handle_a.view_seq = 15
        assert handle_b.get_view_seq() == 20

    @pytest.mark.asyncio
    async def test_non_blocking_observations(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test that observation operations don't block between agents."""
        handle_a = await orchestrator.register_agent("agent_a", observation_policy)
        handle_b = await orchestrator.register_agent("agent_b", observation_policy)

        # Put observation for agent_a only
        queue_a = orchestrator._per_agent_queues["agent_a"]
        await queue_a.put_in(0.001, {"view_seq": 1, "data": "for_a"})

        # Agent A should get observation immediately
        delta_a = await handle_a.next_observation()
        assert delta_a["data"] == "for_a"

        # Agent B should not be affected and should timeout quickly
        queue_b = orchestrator._per_agent_queues["agent_b"]
        
        # Test that agent B's queue is empty
        assert queue_b.empty()

    @pytest.mark.asyncio
    async def test_concurrent_intent_submission(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test concurrent intent submission from multiple agents."""
        # Register multiple agents
        handles = []
        for i in range(3):
            handle = await orchestrator.register_agent(f"agent_{i}", observation_policy)
            handles.append(handle)

        # Create intents for each agent
        intents = []
        for i, handle in enumerate(handles):
            intent: Intent = {
                "kind": "Custom",
                "payload": {"agent_index": i},
                "context_seq": 0,
                "req_id": f"req_{i}",
                "agent_id": handle.agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }
            intents.append(intent)

        # Submit all intents concurrently
        tasks = [handle.submit_intent(intent) for handle, intent in zip(handles, intents)]
        req_ids = await asyncio.gather(*tasks)

        # Verify all succeeded
        assert req_ids == ["req_0", "req_1", "req_2"]

        # Verify all effects were logged
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 3

        # Verify each effect has correct agent
        for i, entry in enumerate(entries):
            assert entry.effect["source_id"] == f"agent_{i}"
            assert entry.effect["payload"]["agent_index"] == i

    @pytest.mark.asyncio
    async def test_concurrent_cancellation(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test concurrent cancellation operations."""
        handles = []
        for i in range(3):
            handle = AgentHandle(f"agent_{i}", orchestrator)
            handles.append(handle)

        # Issue tokens for each agent
        tokens = []
        for i, handle in enumerate(handles):
            token = orchestrator.issue_cancel_token(handle.agent_id, f"req_{i}")
            tokens.append(token)

        # Cancel all concurrently
        cancel_tasks = [handle.cancel(f"req_{i}") for i, handle in enumerate(handles)]
        await asyncio.gather(*cancel_tasks)

        # Verify all are cancelled
        for token in tokens:
            assert token.cancelled
            assert token.reason == "user_requested"


class TestAgentHandleErrorHandling:
    """Test error handling in AgentHandle operations."""

    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        """Create orchestrator for testing."""
        config = OrchestratorConfig(max_agents=10)
        return Orchestrator(config, world_id="test_world")

    @pytest.fixture
    def observation_policy(self) -> DefaultObservationPolicy:
        """Create observation policy for testing."""
        return DefaultObservationPolicy(PolicyConfig())

    @pytest.mark.asyncio
    async def test_next_observation_queue_closed(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test next_observation behavior when queue is closed."""
        # Create handle but don't register agent
        handle = AgentHandle("test_agent", orchestrator)

        # Manually create and close a queue
        queue = TimedQueue()
        orchestrator._per_agent_queues["test_agent"] = queue
        await queue.close()

        # Should raise RuntimeError when trying to get from closed empty queue
        with pytest.raises(RuntimeError, match="Queue is closed and empty"):
            await handle.next_observation()

    @pytest.mark.asyncio
    async def test_submit_intent_orchestrator_error(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test error handling when orchestrator raises exception."""
        handle = AgentHandle("test_agent", orchestrator)

        # Create invalid intent (missing required fields)
        invalid_intent: Intent = {
            "kind": "Custom",
            "payload": {},
            "context_seq": 0,
            "req_id": "",  # Empty req_id should cause error
            "agent_id": "test_agent",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        with pytest.raises(ValueError, match="Intent must have 'req_id' field"):
            await handle.submit_intent(invalid_intent)

    def test_agent_handle_string_representation(self, orchestrator: Orchestrator) -> None:
        """Test string representation of AgentHandle for debugging."""
        handle = AgentHandle("test_agent", orchestrator)

        # AgentHandle doesn't have __str__ or __repr__, but we can test basic properties
        assert handle.agent_id == "test_agent"
        assert handle.view_seq == 0

    @pytest.mark.asyncio
    async def test_operations_after_orchestrator_shutdown(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test AgentHandle operations after orchestrator shutdown."""
        handle = await orchestrator.register_agent("test_agent", observation_policy)

        # Shutdown orchestrator
        await orchestrator.shutdown()

        # Operations should fail gracefully
        with pytest.raises(RuntimeError, match="Agent test_agent is not registered"):
            await handle.next_observation()

        # Cancel should not raise error (token won't exist)
        await handle.cancel("some_req")  # Should not raise


class TestAgentHandlePerformance:
    """Test performance characteristics of AgentHandle operations."""

    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        """Create orchestrator for testing."""
        config = OrchestratorConfig(max_agents=100)
        return Orchestrator(config, world_id="perf_test")

    @pytest.fixture
    def observation_policy(self) -> DefaultObservationPolicy:
        """Create observation policy for testing."""
        return DefaultObservationPolicy(PolicyConfig())

    @pytest.mark.asyncio
    async def test_observation_delivery_timing(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test that observations are delivered within reasonable time bounds."""
        handle = await orchestrator.register_agent("test_agent", observation_policy)

        # Schedule observation for immediate delivery
        queue = orchestrator._per_agent_queues["test_agent"]
        start_time = asyncio.get_running_loop().time()
        
        await queue.put_in(0.001, {"view_seq": 1, "data": "test"})

        # Get observation and measure time
        delta = await handle.next_observation()
        end_time = asyncio.get_running_loop().time()

        delivery_time = end_time - start_time
        
        # Should be delivered quickly (within 50ms for this test)
        assert delivery_time < 0.05
        assert delta["data"] == "test"

    @pytest.mark.asyncio
    async def test_multiple_agents_performance(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test performance with multiple agents operating concurrently."""
        # Register multiple agents
        num_agents = 10
        handles = []
        for i in range(num_agents):
            handle = await orchestrator.register_agent(f"agent_{i}", observation_policy)
            handles.append(handle)

        # Submit intents from all agents concurrently
        start_time = asyncio.get_running_loop().time()
        
        tasks = []
        for i, handle in enumerate(handles):
            intent: Intent = {
                "kind": "Custom",
                "payload": {"index": i},
                "context_seq": 0,
                "req_id": f"perf_req_{i}",
                "agent_id": handle.agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }
            tasks.append(handle.submit_intent(intent))

        req_ids = await asyncio.gather(*tasks)
        end_time = asyncio.get_running_loop().time()

        processing_time = end_time - start_time

        # All should complete within reasonable time (1 second for 10 agents)
        assert processing_time < 1.0
        assert len(req_ids) == num_agents

        # Verify all effects were logged
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == num_agents


if __name__ == "__main__":
    pytest.main([__file__])