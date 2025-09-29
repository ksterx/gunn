"""Unit tests for MessageFacade."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gunn.core.orchestrator import Orchestrator
from gunn.facades.message import MessageFacade, MessageSubscription
from gunn.policies.observation import ObservationPolicy
from gunn.utils.errors import BackpressureError


class MockObservationPolicy(ObservationPolicy):
    """Mock observation policy for testing."""

    def __init__(self):
        from gunn.policies.observation import PolicyConfig

        config = PolicyConfig()
        super().__init__(config)

    def filter_world_state(self, world_state, agent_id: str):
        return MagicMock(view_seq=1, context_digest="test_digest")

    def should_observe_event(self, effect, agent_id: str, world_state) -> bool:
        return True

    def calculate_observation_delta(self, old_view, new_view):
        return {
            "view_seq": new_view.view_seq,
            "patches": [],
            "context_digest": new_view.context_digest,
            "schema_version": "1.0.0",
        }


class MockOrchestratorConfig:
    """Mock orchestrator configuration."""

    def __init__(self):
        self.max_agents = 100
        self.staleness_threshold = 0
        self.debounce_ms = 100.0
        self.deadline_ms = 30000.0
        self.token_budget = 1000
        self.backpressure_policy = "defer"
        self.default_priority = 0
        self.use_in_memory_dedup = True
        self.dedup_ttl_minutes = 60
        self.max_dedup_entries = 10000
        self.dedup_warmup_minutes = 5


@pytest.fixture
def mock_config():
    """Provide mock orchestrator configuration."""
    return MockOrchestratorConfig()


@pytest.fixture
def mock_orchestrator():
    """Provide mock orchestrator."""
    orchestrator = MagicMock(spec=Orchestrator)
    orchestrator.world_id = "test_world"
    orchestrator.agent_handles = {}
    orchestrator.initialize = AsyncMock()
    orchestrator.register_agent = AsyncMock()
    orchestrator.broadcast_event = AsyncMock()
    orchestrator.shutdown = AsyncMock()
    return orchestrator


@pytest.fixture
def message_facade(mock_orchestrator):
    """Provide MessageFacade instance with mock orchestrator."""
    return MessageFacade(orchestrator=mock_orchestrator, timeout_seconds=5.0)


@pytest.fixture
def message_facade_with_config(mock_config):
    """Provide MessageFacade instance with config."""
    return MessageFacade(config=mock_config, world_id="test_world", timeout_seconds=5.0)


class TestMessageFacadeInitialization:
    """Test MessageFacade initialization."""

    def test_init_with_orchestrator(self, mock_orchestrator):
        """Test initialization with existing orchestrator."""
        facade = MessageFacade(orchestrator=mock_orchestrator)

        assert facade._orchestrator is mock_orchestrator
        assert facade.world_id == "test_world"
        assert facade.timeout_seconds == 30.0
        assert facade._subscriptions == {}
        assert facade._message_queues == {}

    def test_init_with_config(self, mock_config):
        """Test initialization with configuration."""
        with patch("gunn.facades.message.Orchestrator") as mock_orch_class:
            mock_orch_instance = MagicMock()
            mock_orch_instance.world_id = "test_world"
            mock_orch_class.return_value = mock_orch_instance

            facade = MessageFacade(config=mock_config, world_id="test_world")

            assert facade._orchestrator is mock_orch_instance
            assert facade.world_id == "test_world"
            mock_orch_class.assert_called_once_with(mock_config, "test_world")

    def test_init_without_orchestrator_or_config(self):
        """Test initialization fails without orchestrator or config."""
        with pytest.raises(
            ValueError, match="Either orchestrator or config must be provided"
        ):
            MessageFacade()

    def test_custom_timeout(self, mock_orchestrator):
        """Test initialization with custom timeout."""
        facade = MessageFacade(orchestrator=mock_orchestrator, timeout_seconds=10.0)
        assert facade.timeout_seconds == 10.0


class TestMessageFacadeAgentManagement:
    """Test agent registration and management."""

    @pytest.mark.asyncio
    async def test_register_agent(self, message_facade, mock_orchestrator):
        """Test agent registration."""
        policy = MockObservationPolicy()
        agent_id = "test_agent"

        await message_facade.register_agent(agent_id, policy)

        mock_orchestrator.register_agent.assert_called_once_with(agent_id, policy)
        assert agent_id in message_facade._message_queues
        assert agent_id in message_facade._subscriptions
        assert agent_id in message_facade._delivery_tasks

    @pytest.mark.asyncio
    async def test_initialize(self, message_facade, mock_orchestrator):
        """Test facade initialization."""
        # Add some existing agents to orchestrator
        mock_orchestrator.agent_handles = {"agent1": MagicMock(), "agent2": MagicMock()}

        await message_facade.initialize()

        mock_orchestrator.initialize.assert_called_once()
        assert "agent1" in message_facade._message_queues
        assert "agent2" in message_facade._message_queues


class TestMessageEmission:
    """Test message emission functionality."""

    @pytest.mark.asyncio
    async def test_emit_basic(self, message_facade, mock_orchestrator):
        """Test basic message emission."""
        message_type = "test_message"
        payload = {"data": "test"}
        source_id = "test_source"

        await message_facade.emit(message_type, payload, source_id)

        mock_orchestrator.broadcast_event.assert_called_once()
        call_args = mock_orchestrator.broadcast_event.call_args[0][0]

        assert call_args["kind"] == message_type
        assert call_args["payload"] == payload
        assert call_args["source_id"] == source_id
        assert call_args["schema_version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_emit_with_custom_schema(self, message_facade, mock_orchestrator):
        """Test message emission with custom schema version."""
        await message_facade.emit(
            "test_message", {"data": "test"}, "test_source", schema_version="2.0.0"
        )

        call_args = mock_orchestrator.broadcast_event.call_args[0][0]
        assert call_args["schema_version"] == "2.0.0"

    @pytest.mark.asyncio
    async def test_emit_empty_message_type(self, message_facade):
        """Test emission fails with empty message type."""
        with pytest.raises(ValueError, match="message_type cannot be empty"):
            await message_facade.emit("", {"data": "test"}, "test_source")

    @pytest.mark.asyncio
    async def test_emit_empty_source_id(self, message_facade):
        """Test emission fails with empty source ID."""
        with pytest.raises(ValueError, match="source_id cannot be empty"):
            await message_facade.emit("test_message", {"data": "test"}, "")

    @pytest.mark.asyncio
    async def test_emit_timeout(self, message_facade, mock_orchestrator):
        """Test emission timeout handling."""
        mock_orchestrator.broadcast_event.side_effect = TimeoutError()

        with pytest.raises(TimeoutError, match="Message emission timed out"):
            await message_facade.emit("test_message", {"data": "test"}, "test_source")

    @pytest.mark.asyncio
    async def test_emit_backpressure_error(self, message_facade, mock_orchestrator):
        """Test emission with backpressure error."""
        mock_orchestrator.broadcast_event.side_effect = BackpressureError(
            "test_agent", "intent_queue", 100, 50, "defer"
        )

        with pytest.raises(BackpressureError):
            await message_facade.emit("test_message", {"data": "test"}, "test_source")


class TestMessageSubscription:
    """Test message subscription functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_all_messages(self, message_facade, mock_orchestrator):
        """Test subscribing to all message types."""
        agent_id = "test_agent"
        mock_orchestrator.agent_handles = {agent_id: MagicMock()}

        subscription = await message_facade.subscribe(agent_id)

        assert subscription.agent_id == agent_id
        assert subscription.message_types == set()  # Empty set means all types
        assert subscription.active
        assert subscription in message_facade._subscriptions[agent_id]

    @pytest.mark.asyncio
    async def test_subscribe_specific_types(self, message_facade, mock_orchestrator):
        """Test subscribing to specific message types."""
        agent_id = "test_agent"
        message_types = {"type1", "type2"}
        mock_orchestrator.agent_handles = {agent_id: MagicMock()}

        subscription = await message_facade.subscribe(agent_id, message_types)

        assert subscription.message_types == message_types

    @pytest.mark.asyncio
    async def test_subscribe_with_handler(self, message_facade, mock_orchestrator):
        """Test subscribing with message handler."""
        agent_id = "test_agent"
        handler = MagicMock()
        mock_orchestrator.agent_handles = {agent_id: MagicMock()}

        subscription = await message_facade.subscribe(agent_id, handler=handler)

        assert subscription.handler is handler

    @pytest.mark.asyncio
    async def test_subscribe_unregistered_agent(
        self, message_facade, mock_orchestrator
    ):
        """Test subscribing fails for unregistered agent."""
        mock_orchestrator.agent_handles = {}

        with pytest.raises(ValueError, match="Agent test_agent is not registered"):
            await message_facade.subscribe("test_agent")

    @pytest.mark.asyncio
    async def test_unsubscribe(self, message_facade, mock_orchestrator):
        """Test unsubscribing from messages."""
        agent_id = "test_agent"
        mock_orchestrator.agent_handles = {agent_id: MagicMock()}

        subscription = await message_facade.subscribe(agent_id)
        await message_facade.unsubscribe(subscription)

        assert not subscription.active
        assert subscription not in message_facade._subscriptions[agent_id]


class TestMessageSubscriptionMatching:
    """Test MessageSubscription matching logic."""

    def test_matches_all_types(self):
        """Test subscription matches all types when no specific types set."""
        subscription = MessageSubscription("agent1")

        assert subscription.matches("any_type")
        assert subscription.matches("another_type")

    def test_matches_specific_types(self):
        """Test subscription matches only specific types."""
        subscription = MessageSubscription("agent1", {"type1", "type2"})

        assert subscription.matches("type1")
        assert subscription.matches("type2")
        assert not subscription.matches("type3")

    def test_inactive_subscription_no_match(self):
        """Test inactive subscription doesn't match any types."""
        subscription = MessageSubscription("agent1")
        subscription.active = False

        assert not subscription.matches("any_type")


class TestMessageRetrieval:
    """Test message retrieval functionality."""

    @pytest.mark.asyncio
    async def test_get_messages_empty(self, message_facade, mock_orchestrator):
        """Test getting messages when queue is empty."""
        agent_id = "test_agent"
        mock_orchestrator.agent_handles = {agent_id: MagicMock()}
        message_facade._message_queues[agent_id] = asyncio.Queue()

        messages = await message_facade.get_messages(agent_id)

        assert messages == []

    @pytest.mark.asyncio
    async def test_get_messages_with_data(self, message_facade, mock_orchestrator):
        """Test getting messages when queue has data."""
        agent_id = "test_agent"
        mock_orchestrator.agent_handles = {agent_id: MagicMock()}
        queue = asyncio.Queue()
        message_facade._message_queues[agent_id] = queue

        # Add test messages
        test_messages = [
            {"type": "msg1", "data": "test1"},
            {"type": "msg2", "data": "test2"},
        ]
        for msg in test_messages:
            await queue.put(msg)

        messages = await message_facade.get_messages(agent_id)

        assert len(messages) == 2
        assert messages == test_messages

    @pytest.mark.asyncio
    async def test_get_messages_unregistered_agent(
        self, message_facade, mock_orchestrator
    ):
        """Test getting messages fails for unregistered agent."""
        mock_orchestrator.agent_handles = {}

        with pytest.raises(ValueError, match="Agent test_agent is not registered"):
            await message_facade.get_messages("test_agent")

    @pytest.mark.asyncio
    async def test_wait_for_message(self, message_facade, mock_orchestrator):
        """Test waiting for a specific message."""
        agent_id = "test_agent"
        mock_orchestrator.agent_handles = {agent_id: MagicMock()}
        queue = asyncio.Queue()
        message_facade._message_queues[agent_id] = queue

        test_message = {"type": "target_type", "data": "test"}

        # Add message to queue in background
        async def add_message():
            await asyncio.sleep(0.1)
            await queue.put(test_message)

        asyncio.create_task(add_message())  # noqa

        # Wait for the message
        received = await message_facade.wait_for_message(
            agent_id, "target_type", timeout=1.0
        )

        assert received == test_message

    @pytest.mark.asyncio
    async def test_wait_for_message_timeout(self, message_facade, mock_orchestrator):
        """Test waiting for message times out."""
        agent_id = "test_agent"
        mock_orchestrator.agent_handles = {agent_id: MagicMock()}
        message_facade._message_queues[agent_id] = asyncio.Queue()

        with pytest.raises(
            TimeoutError, match="No message of type 'missing_type' received"
        ):
            await message_facade.wait_for_message(agent_id, "missing_type", timeout=0.1)


class TestMessageDelivery:
    """Test message delivery functionality."""

    @pytest.mark.asyncio
    async def test_observation_to_message_dict(self, message_facade):
        """Test converting dict observation to message."""
        observation = {
            "view_seq": 5,
            "patches": [{"op": "add", "path": "/test", "value": "data"}],
            "context_digest": "digest123",
            "schema_version": "1.0.0",
        }

        message = await message_facade._observation_to_message("agent1", observation)

        assert message["type"] == "observation_update"
        assert message["agent_id"] == "agent1"
        assert message["data"] == observation
        assert "timestamp" in message

    @pytest.mark.asyncio
    async def test_observation_to_message_object(self, message_facade):
        """Test converting object observation to message."""
        observation = MagicMock()
        observation.view_seq = 3
        observation.patches = []
        observation.context_digest = "digest456"
        observation.schema_version = "2.0.0"

        message = await message_facade._observation_to_message("agent1", observation)

        assert message["type"] == "observation_update"
        assert message["data"]["view_seq"] == 3
        assert message["data"]["schema_version"] == "2.0.0"

    @pytest.mark.asyncio
    async def test_deliver_message_to_agent_with_subscription(self, message_facade):
        """Test delivering message to agent with matching subscription."""
        agent_id = "test_agent"
        message_facade._message_queues[agent_id] = asyncio.Queue()

        # Create subscription
        subscription = MessageSubscription(agent_id, {"observation_update"})
        message_facade._subscriptions[agent_id] = [subscription]

        test_message = {"type": "observation_update", "data": "test"}

        await message_facade._deliver_message_to_agent(agent_id, test_message)

        # Check message was queued
        assert message_facade._message_queues[agent_id].qsize() == 1
        queued_message = await message_facade._message_queues[agent_id].get()
        assert queued_message == test_message

    @pytest.mark.asyncio
    async def test_deliver_message_with_handler(self, message_facade):
        """Test delivering message calls handler."""
        agent_id = "test_agent"
        handler = MagicMock()
        message_facade._message_queues[agent_id] = asyncio.Queue()

        # Create subscription with handler
        subscription = MessageSubscription(agent_id, {"test_type"}, handler)
        message_facade._subscriptions[agent_id] = [subscription]

        test_message = {"type": "test_type", "data": "test"}

        await message_facade._deliver_message_to_agent(agent_id, test_message)

        handler.assert_called_once_with(agent_id, test_message)

    @pytest.mark.asyncio
    async def test_deliver_message_async_handler(self, message_facade):
        """Test delivering message calls async handler."""
        agent_id = "test_agent"
        handler = AsyncMock()
        message_facade._message_queues[agent_id] = asyncio.Queue()

        # Create subscription with async handler
        subscription = MessageSubscription(agent_id, {"test_type"}, handler)
        message_facade._subscriptions[agent_id] = [subscription]

        test_message = {"type": "test_type", "data": "test"}

        await message_facade._deliver_message_to_agent(agent_id, test_message)

        handler.assert_called_once_with(agent_id, test_message)


class TestMessageFacadeUtilities:
    """Test utility methods."""

    def test_get_orchestrator(self, message_facade, mock_orchestrator):
        """Test getting underlying orchestrator."""
        assert message_facade.get_orchestrator() is mock_orchestrator

    def test_set_timeout_valid(self, message_facade):
        """Test setting valid timeout."""
        message_facade.set_timeout(15.0)
        assert message_facade.timeout_seconds == 15.0

    def test_set_timeout_invalid(self, message_facade):
        """Test setting invalid timeout fails."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            message_facade.set_timeout(0)

    def test_get_agent_subscriptions(self, message_facade, mock_orchestrator):
        """Test getting agent subscriptions."""
        agent_id = "test_agent"
        mock_orchestrator.agent_handles = {agent_id: MagicMock()}

        # Add some subscriptions
        sub1 = MessageSubscription(agent_id, {"type1"})
        sub2 = MessageSubscription(agent_id, {"type2"})
        sub2.active = False  # Inactive subscription
        message_facade._subscriptions[agent_id] = [sub1, sub2]

        active_subs = message_facade.get_agent_subscriptions(agent_id)

        assert len(active_subs) == 1
        assert active_subs[0] is sub1

    def test_get_message_queue_size(self, message_facade, mock_orchestrator):
        """Test getting message queue size."""
        agent_id = "test_agent"
        mock_orchestrator.agent_handles = {agent_id: MagicMock()}

        # No queue yet
        assert message_facade.get_message_queue_size(agent_id) == 0

        # Add queue with messages
        queue = asyncio.Queue()
        message_facade._message_queues[agent_id] = queue

        # Add some messages
        for i in range(3):
            queue.put_nowait({"msg": i})

        assert message_facade.get_message_queue_size(agent_id) == 3

    def test_repr(self, message_facade, mock_orchestrator):
        """Test string representation."""
        mock_orchestrator.agent_handles = {"agent1": MagicMock(), "agent2": MagicMock()}
        message_facade._subscriptions = {
            "agent1": [MagicMock(), MagicMock()],
            "agent2": [MagicMock()],
        }

        repr_str = repr(message_facade)

        assert "MessageFacade" in repr_str
        assert "world_id=test_world" in repr_str
        assert "agents=2" in repr_str
        assert "active_subscriptions=3" in repr_str


class TestMessageFacadeShutdown:
    """Test facade shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown(self, message_facade, mock_orchestrator):
        """Test facade shutdown."""

        # Create real asyncio tasks that we can control
        async def dummy_task():
            await asyncio.sleep(10)  # Long running task

        task1 = asyncio.create_task(dummy_task())
        task2 = asyncio.create_task(dummy_task())
        task2.cancel()  # Make task2 already done

        # Wait a bit to ensure task2 is cancelled
        await asyncio.sleep(0.01)

        message_facade._delivery_tasks = {"agent1": task1, "agent2": task2}
        message_facade._message_queues = {"agent1": asyncio.Queue()}
        message_facade._subscriptions = {"agent1": [MagicMock()]}

        await message_facade.shutdown()

        # Check shutdown event was set
        assert message_facade._shutdown_event.is_set()

        # Check task1 was cancelled
        assert task1.cancelled()

        # Check cleanup
        assert message_facade._delivery_tasks == {}
        assert message_facade._message_queues == {}
        assert message_facade._subscriptions == {}

        # Check orchestrator shutdown
        mock_orchestrator.shutdown.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
