"""Unit tests for conversational agent implementation."""

import time
from unittest.mock import AsyncMock

import pytest

from gunn.core.conversational_agent import (
    ConversationalAgent,
    ConversationMemory,
    LLMResponse,
    MockLLMClient,
)
from gunn.schemas.messages import View


class TestConversationMemory:
    """Test cases for ConversationMemory class."""

    def test_init(self):
        """Test ConversationMemory initialization."""
        memory = ConversationMemory(max_messages=10, max_age_seconds=60.0)
        assert memory.max_messages == 10
        assert memory.max_age_seconds == 60.0
        assert memory.messages == []
        assert memory.agent_positions == {}
        assert memory.last_interaction_time == {}

    def test_add_message(self):
        """Test adding messages to memory."""
        memory = ConversationMemory()
        timestamp = time.time()
        position = (1.0, 2.0, 3.0)

        memory.add_message("agent_1", "Hello world", timestamp, position)

        assert len(memory.messages) == 1
        message = memory.messages[0]
        assert message["speaker_id"] == "agent_1"
        assert message["text"] == "Hello world"
        assert message["timestamp"] == timestamp
        assert message["speaker_position"] == position
        assert memory.last_interaction_time["agent_1"] == timestamp

    def test_update_agent_position(self):
        """Test updating agent positions."""
        memory = ConversationMemory()
        position = (5.0, 10.0, 0.0)

        memory.update_agent_position("agent_1", position)

        assert memory.agent_positions["agent_1"] == position

    def test_get_recent_messages(self):
        """Test retrieving recent messages."""
        memory = ConversationMemory()
        base_time = time.time()

        # Add messages at different times
        memory.add_message("agent_1", "Message 1", base_time)
        memory.add_message("agent_2", "Message 2", base_time + 1)
        memory.add_message("agent_1", "Message 3", base_time + 2)

        # Get all messages
        all_messages = memory.get_recent_messages()
        assert len(all_messages) == 3

        # Get messages since a specific time
        recent = memory.get_recent_messages(since_timestamp=base_time + 0.5)
        assert len(recent) == 2
        assert recent[0]["text"] == "Message 2"
        assert recent[1]["text"] == "Message 3"

        # Get limited count
        limited = memory.get_recent_messages(max_count=2)
        assert len(limited) == 2
        assert limited[0]["text"] == "Message 2"
        assert limited[1]["text"] == "Message 3"

    def test_get_nearby_agents(self):
        """Test finding nearby agents."""
        memory = ConversationMemory()

        # Add agent positions
        memory.update_agent_position("agent_1", (0.0, 0.0, 0.0))
        memory.update_agent_position("agent_2", (3.0, 4.0, 0.0))  # Distance 5
        memory.update_agent_position("agent_3", (10.0, 0.0, 0.0))  # Distance 10

        # Find agents within distance 6
        nearby = memory.get_nearby_agents((0.0, 0.0, 0.0), max_distance=6.0)

        assert len(nearby) == 2
        assert nearby[0][0] == "agent_1"  # Closest (distance 0)
        assert nearby[0][2] == 0.0
        assert nearby[1][0] == "agent_2"  # Next closest (distance 5)
        assert nearby[1][2] == 5.0

    def test_cleanup_old_messages(self):
        """Test cleanup of old messages."""
        memory = ConversationMemory(max_messages=2, max_age_seconds=1.0)
        base_time = time.time()

        # Add messages
        memory.add_message("agent_1", "Old message", base_time - 2.0)
        memory.add_message("agent_2", "Recent message 1", base_time)
        memory.add_message("agent_3", "Recent message 2", base_time)

        # Trigger cleanup by adding another message
        memory.add_message("agent_4", "New message", base_time + 0.1)

        # Should have only 2 most recent messages (max_messages=2)
        # and old message should be removed (older than max_age_seconds)
        assert len(memory.messages) == 2
        texts = [m["text"] for m in memory.messages]
        assert "Old message" not in texts
        assert "Recent message 2" in texts
        assert "New message" in texts


class TestLLMResponse:
    """Test cases for LLMResponse class."""

    def test_init_speak_response(self):
        """Test LLMResponse initialization for speak action."""
        response = LLMResponse(
            action_type="speak", text="Hello there!", reasoning="Greeting someone"
        )

        assert response.action_type == "speak"
        assert response.text == "Hello there!"
        assert response.reasoning == "Greeting someone"
        assert response.target_position is None
        assert response.target_agent is None

    def test_init_move_response(self):
        """Test LLMResponse initialization for move action."""
        response = LLMResponse(
            action_type="move",
            target_position=[1.0, 2.0, 0.0],
            reasoning="Exploring the area",
        )

        assert response.action_type == "move"
        assert response.target_position == [1.0, 2.0, 0.0]
        assert response.reasoning == "Exploring the area"
        assert response.text is None
        assert response.target_agent is None

    def test_init_interact_response(self):
        """Test LLMResponse initialization for interact action."""
        response = LLMResponse(
            action_type="interact",
            target_agent="agent_2",
            reasoning="Want to talk to them",
        )

        assert response.action_type == "interact"
        assert response.target_agent == "agent_2"
        assert response.reasoning == "Want to talk to them"
        assert response.text is None
        assert response.target_position is None


class TestMockLLMClient:
    """Test cases for MockLLMClient class."""

    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test mock LLM response generation."""
        client = MockLLMClient(response_delay=0.01)

        response = await client.generate_response(
            "Hello world", "friendly", max_tokens=100, temperature=0.7
        )

        assert isinstance(response, LLMResponse)
        assert response.action_type in ["speak", "move", "wait"]
        assert response.reasoning is not None

    @pytest.mark.asyncio
    async def test_response_delay(self):
        """Test that mock client respects response delay."""
        client = MockLLMClient(response_delay=0.1)

        start_time = time.time()
        await client.generate_response("test", "test")
        elapsed = time.time() - start_time

        assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_response_patterns(self):
        """Test mock client response patterns."""
        client = MockLLMClient(response_delay=0.01)

        # Test greeting response
        response = await client.generate_response("Hello there", "friendly")
        # Should trigger speak response due to "hello" in context

        # Test multiple responses to see variety
        responses = []
        for i in range(10):
            resp = await client.generate_response(f"Context {i}", "test")
            responses.append(resp.action_type)

        # Should have variety in response types
        unique_types = set(responses)
        assert len(unique_types) > 1


class TestConversationalAgent:
    """Test cases for ConversationalAgent class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMClient(response_delay=0.01)
        self.agent = ConversationalAgent(
            llm_client=self.mock_llm,
            personality="friendly test agent",
            name="TestAgent",
            conversation_distance=10.0,
            response_probability=1.0,  # Always respond for testing
            movement_probability=0.0,  # Don't move randomly in tests
        )

    def test_init(self):
        """Test ConversationalAgent initialization."""
        assert self.agent.personality == "friendly test agent"
        assert self.agent.name == "TestAgent"
        assert self.agent.conversation_distance == 10.0
        assert self.agent.response_probability == 1.0
        assert self.agent.movement_probability == 0.0
        assert isinstance(self.agent.memory, ConversationMemory)

    def test_extract_nearby_agents(self):
        """Test extracting nearby agents from observation."""
        # Create observation with multiple agents
        observation = View(
            agent_id="test_agent",
            view_seq=1,
            visible_entities={
                "test_agent": {"name": "TestAgent", "position": [0.0, 0.0, 0.0]},
                "agent_1": {
                    "name": "Agent1",
                    "position": [3.0, 4.0, 0.0],
                },  # Distance 5
                "agent_2": {
                    "name": "Agent2",
                    "position": [20.0, 0.0, 0.0],
                },  # Distance 20
                "landmark_1": {
                    "name": "Landmark",
                    "position": [1.0, 1.0, 0.0],
                },  # Not an agent
            },
            visible_relationships={},
            context_digest="test",
        )

        # Set agent position
        self.agent.current_position = (0.0, 0.0, 0.0)

        nearby = self.agent._extract_nearby_agents(observation, "test_agent")

        # Should find agent_1 (within distance) but not agent_2 (too far) or landmark_1 (not agent)
        assert len(nearby) == 1
        assert nearby[0]["agent_id"] == "agent_1"
        assert nearby[0]["name"] == "Agent1"
        assert nearby[0]["distance"] == 5.0

    def test_extract_recent_messages(self):
        """Test extracting recent messages from observation."""
        current_time = time.time()

        observation = View(
            agent_id="test_agent",
            view_seq=1,
            visible_entities={
                "test_agent": {"name": "TestAgent"},
                "agent_1": {
                    "name": "Agent1",
                    "recent_message": {
                        "text": "Hello everyone!",
                        "timestamp": current_time - 1.0,
                    },
                    "position": [1.0, 1.0, 0.0],
                },
                "agent_2": {
                    "name": "Agent2",
                    "recent_message": {
                        "text": "How is everyone doing?",
                        "timestamp": current_time - 0.5,
                    },
                },
            },
            visible_relationships={},
            context_digest="test",
        )

        messages = self.agent._extract_recent_messages(observation, current_time)

        assert len(messages) >= 2
        # Messages should be sorted by timestamp
        assert messages[-1]["text"] == "How is everyone doing?"
        assert messages[-2]["text"] == "Hello everyone!"

    def test_build_context(self):
        """Test building context string for LLM."""
        # Set up agent state
        self.agent.current_position = (5.0, 5.0, 0.0)

        nearby_agents = [
            {
                "agent_id": "agent_1",
                "name": "Agent1",
                "distance": 3.0,
                "position": (2.0, 5.0, 0.0),
                "data": {},
            }
        ]

        recent_messages = [
            {
                "speaker_id": "agent_1",
                "text": "Hello there!",
                "timestamp": time.time() - 1.0,
                "speaker_position": (2.0, 5.0, 0.0),
            }
        ]

        context = self.agent._build_context(
            nearby_agents, recent_messages, "test_agent"
        )

        assert "TestAgent" in context
        assert "friendly test agent" in context
        assert "position (5.0, 5.0, 0.0)" in context
        assert "Agent1 at distance 3.0" in context
        assert "agent_1: Hello there!" in context
        assert "speak" in context and "move" in context  # Action guidance

    def test_should_take_action(self):
        """Test action decision logic."""
        current_time = time.time()
        self.agent.last_action_time = current_time - 5.0  # 5 seconds ago

        # Should respond to messages mentioning the agent
        messages_with_mention = [
            {
                "speaker_id": "agent_1",
                "text": "Hey TestAgent, how are you?",
                "timestamp": current_time - 1.0,
                "speaker_position": None,
            }
        ]

        should_act = self.agent._should_take_action(
            context_changed=False,
            recent_messages=messages_with_mention,
            nearby_agents=[],
            current_time=current_time,
        )

        assert should_act  # Should respond to direct mention

        # Should act when context changes and there are nearby agents
        should_act = self.agent._should_take_action(
            context_changed=True,
            recent_messages=[],
            nearby_agents=[{"agent_id": "agent_1"}],
            current_time=current_time,
        )

        # With response_probability=1.0, should always act when context changes
        assert should_act

    def test_create_intent_from_response(self):
        """Test creating intents from LLM responses."""
        # Test speak intent
        speak_response = LLMResponse(
            action_type="speak", text="Hello everyone!", reasoning="Greeting"
        )

        intent = self.agent._create_intent_from_response(
            speak_response, context_seq=5, agent_id="test_agent"
        )

        assert intent is not None
        assert intent["kind"] == "Speak"
        assert intent["payload"]["text"] == "Hello everyone!"
        assert intent["payload"]["agent_id"] == "test_agent"
        assert intent["context_seq"] == 5
        assert intent["agent_id"] == "test_agent"

        # Test move intent
        self.agent.current_position = (0.0, 0.0, 0.0)
        move_response = LLMResponse(
            action_type="move", target_position=[5.0, 5.0, 0.0], reasoning="Exploring"
        )

        intent = self.agent._create_intent_from_response(
            move_response, context_seq=6, agent_id="test_agent"
        )

        assert intent is not None
        assert intent["kind"] == "Move"
        assert intent["payload"]["to"] == [5.0, 5.0, 0.0]
        assert intent["payload"]["from"] == [0.0, 0.0, 0.0]

        # Test interact intent
        interact_response = LLMResponse(
            action_type="interact", target_agent="agent_1", reasoning="Want to talk"
        )

        intent = self.agent._create_intent_from_response(
            interact_response, context_seq=7, agent_id="test_agent"
        )

        assert intent is not None
        assert intent["kind"] == "Interact"
        assert intent["payload"]["target_id"] == "agent_1"
        assert intent["payload"]["interaction_type"] == "talk"

        # Test wait response (should return None)
        wait_response = LLMResponse(action_type="wait", reasoning="Observing")

        intent = self.agent._create_intent_from_response(
            wait_response, context_seq=8, agent_id="test_agent"
        )

        assert intent is None

    @pytest.mark.asyncio
    async def test_process_observation(self):
        """Test processing observations and generating intents."""
        # Create observation with nearby agent and message
        current_time = time.time()
        observation = View(
            agent_id="test_agent",
            view_seq=10,
            visible_entities={
                "test_agent": {"name": "TestAgent", "position": [0.0, 0.0, 0.0]},
                "agent_1": {
                    "name": "Agent1",
                    "position": [3.0, 4.0, 0.0],
                    "recent_message": {
                        "text": "Hello TestAgent!",
                        "timestamp": current_time - 1.0,
                    },
                },
            },
            visible_relationships={},
            context_digest="test",
        )

        intent = await self.agent.process_observation(observation, "test_agent")

        # Should generate an intent (likely speak due to direct mention)
        assert intent is not None
        assert intent["agent_id"] == "test_agent"
        assert intent["context_seq"] == 10
        assert "req_id" in intent
        assert intent["schema_version"] == "1.0.0"

        # Check that agent state was updated
        assert self.agent.observations_processed == 1
        assert self.agent.current_position == (0.0, 0.0, 0.0)

    @pytest.mark.asyncio
    async def test_lifecycle_methods(self):
        """Test agent lifecycle methods."""
        # Test on_loop_start
        await self.agent.on_loop_start("test_agent")
        # Should not raise any exceptions

        # Test on_loop_stop
        await self.agent.on_loop_stop("test_agent")
        # Should not raise any exceptions

        # Test on_error
        test_error = Exception("Test error")
        should_continue = await self.agent.on_error("test_agent", test_error)
        assert should_continue is True  # Should continue on error

    def test_get_stats(self):
        """Test getting agent statistics."""
        # Set some state
        self.agent.observations_processed = 5
        self.agent.messages_sent = 3
        self.agent.moves_made = 2
        self.agent.current_position = (1.0, 2.0, 3.0)

        stats = self.agent.get_stats()

        assert stats["name"] == "TestAgent"
        assert stats["personality"] == "friendly test agent"
        assert stats["observations_processed"] == 5
        assert stats["messages_sent"] == 3
        assert stats["moves_made"] == 2
        assert stats["current_position"] == (1.0, 2.0, 3.0)
        assert "memory_messages" in stats
        assert "known_agents" in stats

    @pytest.mark.asyncio
    async def test_error_handling_in_process_observation(self):
        """Test error handling during observation processing."""
        # Create a mock LLM client that raises an exception
        error_llm = AsyncMock()
        error_llm.generate_response.side_effect = Exception("LLM error")

        agent = ConversationalAgent(
            llm_client=error_llm, personality="test", response_probability=1.0
        )

        observation = View(
            agent_id="test_agent",
            view_seq=1,
            visible_entities={
                "test_agent": {"name": "TestAgent"},
                "agent_1": {
                    "recent_message": {
                        "text": "Hello TestAgent!",
                        "timestamp": time.time(),
                    }
                },
            },
            visible_relationships={},
            context_digest="test",
        )

        # Should handle the error gracefully and return None
        intent = await agent.process_observation(observation, "test_agent")
        assert intent is None

    def test_context_length_limiting(self):
        """Test that context is limited to max_context_length."""
        agent = ConversationalAgent(
            llm_client=self.mock_llm,
            personality="test",
            max_context_length=100,  # Very short limit
        )

        # Create a scenario that would generate long context
        nearby_agents = [
            {
                "agent_id": f"agent_{i}",
                "name": f"VeryLongAgentNameThatTakesUpSpace_{i}",
                "distance": float(i),
                "position": (i, i, 0),
                "data": {},
            }
            for i in range(10)
        ]

        recent_messages = [
            {
                "speaker_id": f"agent_{i}",
                "text": f"This is a very long message that takes up a lot of space in the context string - message number {i}",
                "timestamp": time.time() - i,
                "speaker_position": (i, i, 0),
            }
            for i in range(10)
        ]

        context = agent._build_context(nearby_agents, recent_messages, "test_agent")

        # Context should be truncated
        assert len(context) <= 103  # 100 + "..." = 103
        if len(context) == 103:
            assert context.endswith("...")


if __name__ == "__main__":
    pytest.main([__file__])
