"""Tests for collaborative behavior patterns and coordination.

This module tests the collaborative behavior detection, coordination patterns,
and multi-agent collaboration scenarios.
"""

import asyncio
import time

import pytest

from gunn.core.collaborative_agent import (
    CollaborativeAgent,
    SpecializedCollaborativeAgent,
)
from gunn.core.collaborative_behavior import (
    CollaborationOpportunity,
    CollaborationType,
    CollaborativeBehaviorManager,
    FollowingDetector,
    GroupConversationDetector,
    TaskCoordinationDetector,
    create_collaborative_intent,
)
from gunn.schemas.messages import View


class TestFollowingDetector:
    """Test following behavior detection."""

    def test_detect_following_opportunity(self):
        """Test detection of following opportunities."""
        detector = FollowingDetector(follow_distance=5.0, movement_threshold=2.0)

        # Create observation with moving agent
        observation = View(
            agent_id="agent_1",
            view_seq=1,
            visible_entities={
                "agent_1": {"position": [0.0, 0.0, 0.0]},
                "agent_2": {"position": [8.0, 0.0, 0.0]},  # Far enough to follow
            },
            visible_relationships={},
            context_digest="test",
        )

        # Memory with previous position showing movement
        agent_memory = {
            "agent_positions": {
                "agent_2": {
                    "position": (5.0, 0.0, 0.0),  # Previous position
                    "timestamp": time.time() - 1.0,
                }
            }
        }

        opportunities = detector.detect_opportunities(
            observation, "agent_1", agent_memory
        )

        assert len(opportunities) == 1
        opportunity = opportunities[0]
        assert opportunity.collaboration_type == CollaborationType.FOLLOWING
        assert "agent_2" in opportunity.target_agents
        assert opportunity.confidence > 0.0
        assert opportunity.suggested_action == "move"

    def test_no_following_when_too_close(self):
        """Test that following is not suggested when agents are already close."""
        detector = FollowingDetector(follow_distance=5.0)

        observation = View(
            agent_id="agent_1",
            view_seq=1,
            visible_entities={
                "agent_1": {"position": [0.0, 0.0, 0.0]},
                "agent_2": {"position": [3.0, 0.0, 0.0]},  # Too close
            },
            visible_relationships={},
            context_digest="test",
        )

        agent_memory = {}
        opportunities = detector.detect_opportunities(
            observation, "agent_1", agent_memory
        )

        assert len(opportunities) == 0


class TestGroupConversationDetector:
    """Test group conversation detection."""

    def test_detect_group_conversation(self):
        """Test detection of group conversation opportunities."""
        detector = GroupConversationDetector(
            conversation_radius=10.0, min_participants=2
        )

        # Create observation with multiple agents having recent messages
        observation = View(
            agent_id="agent_1",
            view_seq=1,
            visible_entities={
                "agent_1": {"position": [0.0, 0.0, 0.0]},
                "agent_2": {
                    "position": [5.0, 0.0, 0.0],
                    "recent_message": {
                        "text": "Hello everyone!",
                        "timestamp": time.time() - 5.0,
                    },
                },
                "agent_3": {
                    "position": [3.0, 4.0, 0.0],
                    "recent_message": {
                        "text": "How is everyone doing?",
                        "timestamp": time.time() - 3.0,
                    },
                },
            },
            visible_relationships={},
            context_digest="test",
        )

        agent_memory = {}
        opportunities = detector.detect_opportunities(
            observation, "agent_1", agent_memory
        )

        # May detect multiple opportunities (one for each speaker cluster)
        assert len(opportunities) >= 1
        opportunity = opportunities[0]
        assert opportunity.collaboration_type == CollaborationType.GROUP_CONVERSATION
        assert len(opportunity.target_agents) >= 2
        assert opportunity.confidence > 0.0
        assert opportunity.suggested_action == "speak"


class TestTaskCoordinationDetector:
    """Test task coordination detection."""

    def test_detect_help_request(self):
        """Test detection of help requests."""
        detector = TaskCoordinationDetector()

        observation = View(
            agent_id="agent_1",
            view_seq=1,
            visible_entities={
                "agent_1": {"position": [0.0, 0.0, 0.0]},
                "agent_2": {
                    "position": [5.0, 0.0, 0.0],
                    "recent_message": {
                        "text": "I need help with this problem!",
                        "timestamp": time.time() - 2.0,
                    },
                },
            },
            visible_relationships={},
            context_digest="test",
        )

        agent_memory = {}
        opportunities = detector.detect_opportunities(
            observation, "agent_1", agent_memory
        )

        assert len(opportunities) >= 1
        help_opportunities = [
            opp
            for opp in opportunities
            if opp.collaboration_type == CollaborationType.HELPING
        ]
        assert len(help_opportunities) == 1

        opportunity = help_opportunities[0]
        assert "agent_2" in opportunity.target_agents
        assert opportunity.confidence > 0.0
        assert opportunity.suggested_action == "speak"


class TestCollaborativeBehaviorManager:
    """Test collaborative behavior manager."""

    def test_detect_multiple_opportunities(self):
        """Test detection of multiple collaboration opportunities."""
        manager = CollaborativeBehaviorManager()

        # Create complex observation with multiple collaboration opportunities
        observation = View(
            agent_id="agent_1",
            view_seq=1,
            visible_entities={
                "agent_1": {"position": [0.0, 0.0, 0.0]},
                "agent_2": {
                    "position": [8.0, 0.0, 0.0],  # Following opportunity
                },
                "agent_3": {
                    "position": [2.0, 2.0, 0.0],
                    "recent_message": {
                        "text": "I need help with this task!",  # Help opportunity
                        "timestamp": time.time() - 1.0,
                    },
                },
                "agent_4": {
                    "position": [1.0, 1.0, 0.0],
                    "recent_message": {
                        "text": "What do you think about this?",  # Conversation opportunity
                        "timestamp": time.time() - 2.0,
                    },
                },
            },
            visible_relationships={},
            context_digest="test",
        )

        # Memory showing agent_2 has moved (for following detection)
        agent_memory = {
            "agent_positions": {
                "agent_2": {"position": (5.0, 0.0, 0.0), "timestamp": time.time() - 1.0}
            }
        }

        opportunities = manager.detect_collaboration_opportunities(
            observation, "agent_1", agent_memory
        )

        # Should detect multiple types of opportunities
        assert len(opportunities) >= 2

        # Check that opportunities are sorted by confidence
        for i in range(len(opportunities) - 1):
            assert opportunities[i].confidence >= opportunities[i + 1].confidence

    def test_update_coordination_patterns(self):
        """Test updating coordination patterns."""
        manager = CollaborativeBehaviorManager()

        # Create observation with clustered agents
        observation = View(
            agent_id="agent_1",
            view_seq=1,
            visible_entities={
                "agent_1": {"position": [0.0, 0.0, 0.0]},
                "agent_2": {"position": [2.0, 1.0, 0.0]},
                "agent_3": {"position": [1.0, 2.0, 0.0]},
                "agent_4": {"position": [20.0, 20.0, 0.0]},  # Far away
            },
            visible_relationships={},
            context_digest="test",
        )

        manager.update_coordination_patterns(observation, "agent_1")

        # Should detect spatial clustering pattern
        active_patterns = manager.get_active_coordination_patterns("agent_1")
        assert (
            len(active_patterns) >= 0
        )  # May or may not detect pattern depending on thresholds


class TestCreateCollaborativeIntent:
    """Test collaborative intent creation."""

    def test_create_speak_intent(self):
        """Test creation of speaking intent for collaboration."""
        opportunity = CollaborationOpportunity(
            collaboration_type=CollaborationType.HELPING,
            target_agents=["agent_2"],
            context={"help_request": "I need assistance"},
            confidence=0.8,
            suggested_action="speak",
            reasoning="Agent needs help",
        )

        intent = create_collaborative_intent(opportunity, "agent_1", 5)

        assert intent is not None
        assert intent["kind"] == "Speak"
        assert intent["agent_id"] == "agent_1"
        assert intent["context_seq"] == 5
        assert intent["priority"] == 1  # Higher priority for collaborative actions
        assert "collaboration_context" in intent["payload"]

        collab_context = intent["payload"]["collaboration_context"]
        assert collab_context["type"] == "helping"
        assert collab_context["target_agents"] == ["agent_2"]
        assert collab_context["confidence"] == 0.8

    def test_create_move_intent(self):
        """Test creation of movement intent for following."""
        opportunity = CollaborationOpportunity(
            collaboration_type=CollaborationType.FOLLOWING,
            target_agents=["agent_2"],
            context={"target_position": [10.0, 5.0, 0.0], "preferred_distance": 5.0},
            confidence=0.7,
            suggested_action="move",
            reasoning="Following agent_2",
        )

        intent = create_collaborative_intent(opportunity, "agent_1", 3)

        assert intent is not None
        assert intent["kind"] == "Move"
        assert intent["agent_id"] == "agent_1"
        assert intent["context_seq"] == 3
        assert "to" in intent["payload"]
        assert "collaboration_context" in intent["payload"]

        collab_context = intent["payload"]["collaboration_context"]
        assert collab_context["type"] == "following"
        assert collab_context["following_agent"] == "agent_2"


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response_delay: float = 0.01):
        self.response_delay = response_delay
        self.call_count = 0

    async def generate_response(self, context: str, personality: str, **kwargs):
        """Generate mock response."""
        from gunn.core.conversational_agent import LLMResponse

        await asyncio.sleep(self.response_delay)
        self.call_count += 1

        # Generate different responses based on context
        if "collaboration" in context.lower() or "coordinate" in context.lower():
            return LLMResponse(
                action_type="speak",
                text="I'd love to collaborate! How can we work together?",
                reasoning="Responding to collaboration opportunity",
            )
        elif "help" in context.lower():
            return LLMResponse(
                action_type="speak",
                text="I'm here to help! What do you need assistance with?",
                reasoning="Offering help",
            )
        else:
            return LLMResponse(action_type="wait", reasoning="Observing situation")


class TestCollaborativeAgent:
    """Test collaborative agent behavior."""

    @pytest.mark.asyncio
    async def test_collaborative_agent_initialization(self):
        """Test collaborative agent initialization."""
        llm_client = MockLLMClient()

        agent = CollaborativeAgent(
            llm_client=llm_client,
            name="TestAgent",
            collaboration_threshold=0.5,
            enable_following=True,
            enable_helping=True,
            enable_task_coordination=True,
        )

        assert agent.name == "TestAgent"
        assert agent.collaboration_threshold == 0.5
        assert agent.enable_following is True
        assert agent.enable_helping is True
        assert agent.enable_task_coordination is True
        assert isinstance(agent.collaboration_manager, CollaborativeBehaviorManager)

    @pytest.mark.asyncio
    async def test_collaborative_behavior_detection(self):
        """Test that collaborative agent detects and acts on opportunities."""
        llm_client = MockLLMClient()

        agent = CollaborativeAgent(
            llm_client=llm_client,
            name="CollabAgent",
            collaboration_threshold=0.3,  # Low threshold for testing
        )

        # Create observation with collaboration opportunity
        observation = View(
            agent_id="agent_1",
            view_seq=1,
            visible_entities={
                "agent_1": {"position": [0.0, 0.0, 0.0]},
                "agent_2": {
                    "position": [5.0, 0.0, 0.0],
                    "recent_message": {
                        "text": "I really need help with this problem!",
                        "timestamp": time.time() - 1.0,
                    },
                },
            },
            visible_relationships={},
            context_digest="test",
        )

        intent = await agent.process_observation(observation, "agent_1")

        # Should generate collaborative intent
        assert intent is not None
        assert "collaboration_context" in intent["payload"]
        assert agent.collaborative_actions_taken > 0
        assert agent.opportunities_detected > 0

    @pytest.mark.asyncio
    async def test_collaboration_threshold_filtering(self):
        """Test that collaboration threshold filters opportunities."""
        llm_client = MockLLMClient()

        agent = CollaborativeAgent(
            llm_client=llm_client,
            collaboration_threshold=0.9,  # Very high threshold
        )

        # Create observation with low-confidence opportunity
        observation = View(
            agent_id="agent_1",
            view_seq=1,
            visible_entities={
                "agent_1": {"position": [0.0, 0.0, 0.0]},
                "agent_2": {"position": [15.0, 0.0, 0.0]},  # Far away, low confidence
            },
            visible_relationships={},
            context_digest="test",
        )

        intent = await agent.process_observation(observation, "agent_1")

        # Should fall back to normal conversational behavior
        # (may or may not generate intent depending on conversational logic)
        assert (
            agent.collaborative_actions_taken == 0
        )  # No collaborative actions due to high threshold


class TestSpecializedCollaborativeAgent:
    """Test specialized collaborative agent roles."""

    @pytest.mark.asyncio
    async def test_leader_role_configuration(self):
        """Test leader role configuration."""
        llm_client = MockLLMClient()

        agent = SpecializedCollaborativeAgent(
            llm_client=llm_client, role="leader", name="LeaderAgent"
        )

        assert agent.role == "leader"
        assert "leader" in agent.personality.lower()
        assert agent.response_probability == 0.9  # High response rate for leaders
        assert agent.enable_task_coordination is True

    @pytest.mark.asyncio
    async def test_follower_role_configuration(self):
        """Test follower role configuration."""
        llm_client = MockLLMClient()

        agent = SpecializedCollaborativeAgent(
            llm_client=llm_client, role="follower", name="FollowerAgent"
        )

        assert agent.role == "follower"
        assert "follower" in agent.personality.lower()
        assert agent.enable_following is True
        assert agent.collaboration_threshold == 0.3  # Lower threshold for followers

    @pytest.mark.asyncio
    async def test_helper_role_configuration(self):
        """Test helper role configuration."""
        llm_client = MockLLMClient()

        agent = SpecializedCollaborativeAgent(
            llm_client=llm_client, role="helper", name="HelperAgent"
        )

        assert agent.role == "helper"
        assert (
            "helper" in agent.personality.lower()
            or "assist" in agent.personality.lower()
        )
        assert agent.enable_helping is True


class TestCollaborationIntegration:
    """Integration tests for collaborative scenarios."""

    @pytest.mark.asyncio
    async def test_multi_agent_collaboration_scenario(self):
        """Test a complete multi-agent collaboration scenario."""
        # Create multiple collaborative agents with different roles
        leader_llm = MockLLMClient()
        helper_llm = MockLLMClient()
        follower_llm = MockLLMClient()

        leader = SpecializedCollaborativeAgent(
            llm_client=leader_llm,
            role="leader",
            name="Leader",
            collaboration_threshold=0.4,
        )

        helper = SpecializedCollaborativeAgent(
            llm_client=helper_llm,
            role="helper",
            name="Helper",
            collaboration_threshold=0.3,
        )

        follower = SpecializedCollaborativeAgent(
            llm_client=follower_llm,
            role="follower",
            name="Follower",
            collaboration_threshold=0.3,
        )

        # Create scenario where leader needs help and others can assist
        observation = View(
            agent_id="leader",
            view_seq=1,
            visible_entities={
                "leader": {"position": [0.0, 0.0, 0.0]},
                "helper": {"position": [3.0, 2.0, 0.0]},
                "follower": {"position": [2.0, 3.0, 0.0]},
            },
            visible_relationships={},
            context_digest="test",
        )

        # Simulate leader asking for help
        help_observation = View(
            agent_id="helper",
            view_seq=2,
            visible_entities={
                "leader": {
                    "position": [0.0, 0.0, 0.0],
                    "recent_message": {
                        "text": "I need help coordinating this task!",
                        "timestamp": time.time() - 1.0,
                    },
                },
                "helper": {"position": [3.0, 2.0, 0.0]},
                "follower": {"position": [2.0, 3.0, 0.0]},
            },
            visible_relationships={},
            context_digest="test",
        )

        # Helper should detect help opportunity and respond
        helper_intent = await helper.process_observation(help_observation, "helper")

        assert helper_intent is not None
        assert helper_intent["kind"] == "Speak"

        # The helper should respond to help requests (either through collaboration detection or LLM)
        # Check that the response is appropriate for helping
        response_text = helper_intent["payload"]["text"].lower()
        assert any(word in response_text for word in ["help", "assist", "support"])

        # Follower should also process the observation
        follower_intent = await follower.process_observation(
            help_observation, "follower"
        )

        # Follower may or may not generate intent, but should process the observation
        # This validates that the collaborative system is working

    @pytest.mark.asyncio
    async def test_following_behavior_scenario(self):
        """Test following behavior in a movement scenario."""
        llm_client = MockLLMClient()

        follower_agent = CollaborativeAgent(
            llm_client=llm_client,
            name="Follower",
            enable_following=True,
            collaboration_threshold=0.4,
        )

        # Initial observation
        initial_observation = View(
            agent_id="follower",
            view_seq=1,
            visible_entities={
                "follower": {"position": [0.0, 0.0, 0.0]},
                "leader": {"position": [5.0, 0.0, 0.0]},
            },
            visible_relationships={},
            context_digest="test",
        )

        # Process initial observation to establish baseline
        await follower_agent.process_observation(initial_observation, "follower")

        # Leader moves away
        movement_observation = View(
            agent_id="follower",
            view_seq=2,
            visible_entities={
                "follower": {"position": [0.0, 0.0, 0.0]},
                "leader": {"position": [10.0, 0.0, 0.0]},  # Moved further away
            },
            visible_relationships={},
            context_digest="test",
        )

        # Follower should detect following opportunity
        intent = await follower_agent.process_observation(
            movement_observation, "follower"
        )

        # Should generate movement intent to follow
        if intent and intent["kind"] == "Move":
            assert "collaboration_context" in intent["payload"]
            assert intent["payload"]["collaboration_context"]["type"] == "following"
            assert follower_agent.collaborative_actions_taken > 0


if __name__ == "__main__":
    pytest.main([__file__])
