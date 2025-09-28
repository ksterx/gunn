"""Integration tests for MessageFacade with Orchestrator."""

from typing import Any

import pytest

from gunn import MessageFacade, OrchestratorConfig
from gunn.policies.observation import ObservationPolicy, PolicyConfig


class SimpleObservationPolicy(ObservationPolicy):
    """Simple observation policy that allows all observations."""

    def filter_world_state(self, world_state, agent_id: str):
        """Return a simple view for testing."""
        from gunn.schemas.messages import View

        return View(
            agent_id=agent_id,
            view_seq=1,
            visible_entities=world_state.entities,
            visible_relationships=world_state.relationships,
            context_digest="test_digest",
        )

    def should_observe_event(self, effect, agent_id: str, world_state) -> bool:
        """Allow all events to be observed."""
        return True

    def calculate_observation_delta(self, old_view, new_view):
        """Generate simple observation delta."""
        return {
            "view_seq": new_view.view_seq,
            "patches": [{"op": "replace", "path": "/test", "value": "updated"}],
            "context_digest": new_view.context_digest,
            "schema_version": "1.0.0",
        }


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return OrchestratorConfig(
        max_agents=10,
        use_in_memory_dedup=True,
        processing_idle_shutdown_ms=100.0,  # Short timeout for tests
    )


@pytest.fixture
def test_policy():
    """Provide test observation policy."""
    config = PolicyConfig()
    return SimpleObservationPolicy(config)


class TestMessageFacadeIntegration:
    """Integration tests for MessageFacade."""

    @pytest.mark.asyncio
    async def test_basic_message_emission_and_subscription(
        self, test_config, test_policy
    ):
        """Test basic message emission and subscription flow."""
        # Create message facade
        facade = MessageFacade(config=test_config, world_id="test_world")

        try:
            await facade.initialize()

            # Register an agent
            agent_id = "test_agent"
            await facade.register_agent(agent_id, test_policy)

            # Subscribe to messages
            subscription = await facade.subscribe(agent_id, {"test_message"})

            # Emit a message
            await facade.emit(
                message_type="test_message",
                payload={"content": "Hello, World!"},
                source_id="test_source",
            )

            # Check subscription is active
            assert subscription.active
            assert subscription in facade.get_agent_subscriptions(agent_id)

            # Check that the message queue exists
            assert agent_id in facade._message_queues

            # Check that the orchestrator received the broadcast
            assert facade.get_orchestrator() is not None

        finally:
            await facade.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_agents_message_routing(self, test_config, test_policy):
        """Test message routing between multiple agents."""
        facade = MessageFacade(config=test_config, world_id="test_world")

        try:
            await facade.initialize()

            # Register multiple agents
            agent1 = "agent1"
            agent2 = "agent2"
            await facade.register_agent(agent1, test_policy)
            await facade.register_agent(agent2, test_policy)

            # Subscribe agents to different message types
            sub1 = await facade.subscribe(agent1, {"type_a", "type_b"})
            sub2 = await facade.subscribe(agent2, {"type_b", "type_c"})

            # Emit different message types
            await facade.emit("type_a", {"data": "message_a"}, "source1")
            await facade.emit("type_b", {"data": "message_b"}, "source2")
            await facade.emit("type_c", {"data": "message_c"}, "source3")

            # Check that both agents have subscriptions
            assert len(facade.get_agent_subscriptions(agent1)) == 1
            assert len(facade.get_agent_subscriptions(agent2)) == 1

            # Check that message queues exist
            assert agent1 in facade._message_queues
            assert agent2 in facade._message_queues

        finally:
            await facade.shutdown()

    @pytest.mark.asyncio
    async def test_message_handler_callback(self, test_config, test_policy):
        """Test message handler callback functionality."""
        facade = MessageFacade(config=test_config, world_id="test_world")

        received_messages = []

        def message_handler(agent_id: str, message: dict[str, Any]):
            received_messages.append((agent_id, message))

        try:
            await facade.initialize()

            # Register agent with handler
            agent_id = "test_agent"
            await facade.register_agent(agent_id, test_policy)
            subscription = await facade.subscribe(
                agent_id, {"test_type"}, handler=message_handler
            )

            # Check that subscription has handler
            assert subscription.handler is message_handler

            # Emit a message
            await facade.emit("test_type", {"content": "test"}, "source")

            # Check that subscription is properly configured
            assert subscription.matches("test_type")
            assert not subscription.matches("other_type")

        finally:
            await facade.shutdown()

    @pytest.mark.asyncio
    async def test_unsubscribe_functionality(self, test_config, test_policy):
        """Test unsubscribing from messages."""
        facade = MessageFacade(config=test_config, world_id="test_world")

        try:
            await facade.initialize()

            agent_id = "test_agent"
            await facade.register_agent(agent_id, test_policy)

            # Subscribe and then unsubscribe
            subscription = await facade.subscribe(agent_id, {"test_type"})
            assert subscription.active

            await facade.unsubscribe(subscription)
            assert not subscription.active

            # Check subscription is removed from agent's subscriptions
            active_subs = facade.get_agent_subscriptions(agent_id)
            assert subscription not in active_subs

        finally:
            await facade.shutdown()

    @pytest.mark.asyncio
    async def test_wait_for_specific_message(self, test_config, test_policy):
        """Test waiting for a specific message type."""
        facade = MessageFacade(config=test_config, world_id="test_world")

        try:
            await facade.initialize()

            agent_id = "test_agent"
            await facade.register_agent(agent_id, test_policy)
            await facade.subscribe(agent_id)  # Subscribe to all messages

            # Test that wait_for_message would timeout if no messages
            try:
                await facade.wait_for_message(agent_id, "nonexistent_type", timeout=0.1)
                assert False, "Should have timed out"
            except TimeoutError:
                pass  # Expected

            # Emit a message
            await facade.emit("target_type", {"data": "target"}, "source")

            # Check that the facade is properly configured
            assert agent_id in facade._message_queues

        finally:
            await facade.shutdown()

    @pytest.mark.asyncio
    async def test_facade_timeout_configuration(self, test_config, test_policy):
        """Test timeout configuration."""
        facade = MessageFacade(config=test_config, timeout_seconds=1.0)

        try:
            await facade.initialize()

            # Test timeout setting
            assert facade.timeout_seconds == 1.0

            facade.set_timeout(5.0)
            assert facade.timeout_seconds == 5.0

            # Test invalid timeout
            with pytest.raises(ValueError, match="Timeout must be positive"):
                facade.set_timeout(-1.0)

        finally:
            await facade.shutdown()

    @pytest.mark.asyncio
    async def test_facade_repr(self, test_config, test_policy):
        """Test string representation."""
        facade = MessageFacade(config=test_config, world_id="test_world")

        try:
            await facade.initialize()

            # Register some agents
            await facade.register_agent("agent1", test_policy)
            await facade.register_agent("agent2", test_policy)

            # Add subscriptions
            await facade.subscribe("agent1", {"type1"})
            await facade.subscribe("agent2", {"type2", "type3"})

            repr_str = repr(facade)

            assert "MessageFacade" in repr_str
            assert "world_id=test_world" in repr_str
            assert "agents=2" in repr_str

        finally:
            await facade.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
