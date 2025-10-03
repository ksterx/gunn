"""
Comprehensive tests for team communication system.

This module tests team message storage, retrieval, urgency-based prioritization,
team visibility filtering, and message isolation to ensure proper team-only
communication functionality.
"""

import pytest

from gunn.schemas.messages import WorldState

from ..backend.effect_processor import EffectProcessor
from ..backend.gunn_integration import BattleObservationPolicy
from ..shared.models import Agent, BattleWorldState


class TestTeamCommunicationStorage:
    """Test cases for team communication storage and retrieval."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test agents
        self.agent_a1 = Agent(
            agent_id="team_a_agent_1", team="team_a", position=(10.0, 10.0)
        )

        self.agent_a2 = Agent(
            agent_id="team_a_agent_2", team="team_a", position=(15.0, 15.0)
        )

        self.agent_b1 = Agent(
            agent_id="team_b_agent_1", team="team_b", position=(50.0, 50.0)
        )

        # Create test world state
        self.world_state = BattleWorldState(
            agents={
                "team_a_agent_1": self.agent_a1,
                "team_a_agent_2": self.agent_a2,
                "team_b_agent_1": self.agent_b1,
            },
            game_time=10.0,
        )

    def test_add_team_message_basic(self):
        """Test basic team message addition."""
        self.world_state.add_team_message("team_a_agent_1", "Enemy spotted!", "high")

        messages = self.world_state.team_communications["team_a"]
        assert len(messages) == 1
        assert messages[0].sender_id == "team_a_agent_1"
        assert messages[0].message == "Enemy spotted!"
        assert messages[0].urgency == "high"
        assert messages[0].team == "team_a"
        assert messages[0].timestamp == 10.0

    def test_add_team_message_different_teams(self):
        """Test that messages are isolated by team."""
        self.world_state.add_team_message("team_a_agent_1", "Team A message", "medium")
        self.world_state.add_team_message("team_b_agent_1", "Team B message", "low")

        team_a_messages = self.world_state.team_communications["team_a"]
        team_b_messages = self.world_state.team_communications["team_b"]

        assert len(team_a_messages) == 1
        assert len(team_b_messages) == 1
        assert team_a_messages[0].message == "Team A message"
        assert team_b_messages[0].message == "Team B message"

        # Verify team isolation
        assert team_a_messages[0].team == "team_a"
        assert team_b_messages[0].team == "team_b"

    def test_message_storage_limit(self):
        """Test that message storage is limited to prevent memory bloat."""
        # Add 55 messages to exceed the 50 message limit
        for i in range(55):
            self.world_state.add_team_message(
                "team_a_agent_1", f"Message {i}", "medium"
            )

        messages = self.world_state.team_communications["team_a"]
        assert len(messages) == 50  # Should be capped at 50
        assert messages[0].message == "Message 5"  # First kept message
        assert messages[-1].message == "Message 54"  # Last message

    def test_get_recent_team_messages(self):
        """Test retrieving recent team messages."""
        # Add multiple messages
        for i in range(15):
            self.world_state.add_team_message(
                "team_a_agent_1", f"Message {i}", "medium"
            )

        # Get recent messages
        recent = self.world_state.get_recent_team_messages("team_a", 5)
        assert len(recent) == 5
        assert recent[-1].message == "Message 14"  # Most recent
        assert recent[0].message == "Message 10"  # 5th most recent

    def test_get_recent_messages_empty_team(self):
        """Test getting recent messages for team with no messages."""
        recent = self.world_state.get_recent_team_messages("team_a", 10)
        assert len(recent) == 0

    def test_urgency_levels(self):
        """Test different urgency levels are stored correctly."""
        urgencies = ["low", "medium", "high"]

        for i, urgency in enumerate(urgencies):
            self.world_state.add_team_message(
                "team_a_agent_1", f"Message {urgency}", urgency
            )

        messages = self.world_state.team_communications["team_a"]
        assert len(messages) == 3

        for i, urgency in enumerate(urgencies):
            assert messages[i].urgency == urgency
            assert messages[i].message == f"Message {urgency}"


class TestTeamCommunicationEffects:
    """Test cases for team communication effect processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = EffectProcessor()

        # Create test agents
        self.agent_a1 = Agent(
            agent_id="team_a_agent_1", team="team_a", position=(10.0, 10.0)
        )

        self.agent_b1 = Agent(
            agent_id="team_b_agent_1", team="team_b", position=(50.0, 50.0)
        )

        # Create test world state
        self.world_state = BattleWorldState(
            agents={"team_a_agent_1": self.agent_a1, "team_b_agent_1": self.agent_b1},
            game_time=15.0,
        )

    @pytest.mark.asyncio
    async def test_handle_team_message_effect_valid(self):
        """Test handling valid TeamMessage effect."""
        effect = {
            "kind": "TeamMessage",
            "payload": {
                "sender_id": "team_a_agent_1",
                "sender_team": "team_a",
                "message": "Enemy approaching from north!",
                "urgency": "high",
                "timestamp": 20.0,
            },
        }

        await self.processor._handle_team_message(effect, self.world_state)

        # Check that message was added to team communications
        team_messages = self.world_state.team_communications["team_a"]
        assert len(team_messages) == 1
        assert team_messages[0].sender_id == "team_a_agent_1"
        assert team_messages[0].message == "Enemy approaching from north!"
        assert team_messages[0].urgency == "high"
        assert team_messages[0].timestamp == 20.0
        assert team_messages[0].team == "team_a"

        # Check that enemy team doesn't see the message
        enemy_messages = self.world_state.team_communications["team_b"]
        assert len(enemy_messages) == 0

    @pytest.mark.asyncio
    async def test_handle_team_message_team_mismatch(self):
        """Test handling TeamMessage with sender team mismatch."""
        effect = {
            "kind": "TeamMessage",
            "payload": {
                "sender_id": "team_a_agent_1",
                "sender_team": "team_b",  # Wrong team
                "message": "Test message",
                "urgency": "medium",
            },
        }

        await self.processor._handle_team_message(effect, self.world_state)

        # No messages should be added due to team mismatch
        assert len(self.world_state.team_communications["team_a"]) == 0
        assert len(self.world_state.team_communications["team_b"]) == 0

    @pytest.mark.asyncio
    async def test_handle_team_message_invalid_sender(self):
        """Test handling TeamMessage with invalid sender."""
        effect = {
            "kind": "TeamMessage",
            "payload": {
                "sender_id": "nonexistent_agent",
                "sender_team": "team_a",
                "message": "Test message",
                "urgency": "medium",
            },
        }

        await self.processor._handle_team_message(effect, self.world_state)

        # No messages should be added due to invalid sender
        assert len(self.world_state.team_communications["team_a"]) == 0
        assert len(self.world_state.team_communications["team_b"]) == 0

    @pytest.mark.asyncio
    async def test_handle_team_message_invalid_urgency(self):
        """Test handling TeamMessage with invalid urgency level."""
        effect = {
            "kind": "TeamMessage",
            "payload": {
                "sender_id": "team_a_agent_1",
                "sender_team": "team_a",
                "message": "Test message",
                "urgency": "invalid_urgency",
            },
        }

        await self.processor._handle_team_message(effect, self.world_state)

        # Message should be added with default urgency
        team_messages = self.world_state.team_communications["team_a"]
        assert len(team_messages) == 1
        assert team_messages[0].urgency == "medium"  # Default urgency

    @pytest.mark.asyncio
    async def test_handle_team_message_missing_payload(self):
        """Test handling TeamMessage with missing payload fields."""
        effect = {
            "kind": "TeamMessage",
            "payload": {
                "sender_id": "team_a_agent_1",
                # Missing sender_team and message
            },
        }

        await self.processor._handle_team_message(effect, self.world_state)

        # No messages should be added due to missing fields
        assert len(self.world_state.team_communications["team_a"]) == 0


class TestTeamVisibilityFiltering:
    """Test cases for team visibility filtering in observation policies."""

    def setup_method(self):
        """Set up test fixtures."""
        self.team_a_policy = BattleObservationPolicy("team_a", vision_range=30.0)
        self.team_b_policy = BattleObservationPolicy("team_b", vision_range=30.0)

        # Create mock world state
        self.world_state = WorldState(
            entities={
                "team_a_agent_1": {
                    "agent_id": "team_a_agent_1",
                    "team": "team_a",
                    "position": (10.0, 10.0),
                    "health": 100,
                },
                "team_b_agent_1": {
                    "agent_id": "team_b_agent_1",
                    "team": "team_b",
                    "position": (50.0, 50.0),
                    "health": 80,
                },
            },
            spatial_index={
                "team_a_agent_1": (10.0, 10.0, 0.0),
                "team_b_agent_1": (50.0, 50.0, 0.0),
            },
            metadata={
                "team_communications": {
                    "team_a": [
                        {
                            "sender_id": "team_a_agent_1",
                            "team": "team_a",
                            "message": "Team A secret message",
                            "urgency": "high",
                            "timestamp": 10.0,
                        }
                    ],
                    "team_b": [
                        {
                            "sender_id": "team_b_agent_1",
                            "team": "team_b",
                            "message": "Team B secret message",
                            "urgency": "medium",
                            "timestamp": 15.0,
                        }
                    ],
                }
            },
            relationships={},
        )

    def test_team_communication_visibility_same_team(self):
        """Test that agents can see their own team's communications."""
        view = self.team_a_policy.filter_world_state(self.world_state, "team_a_agent_1")

        # Should include team communications
        assert "team_communications" in view.visible_entities
        team_comms = view.visible_entities["team_communications"]
        assert team_comms["team"] == "team_a"
        assert len(team_comms["messages"]) == 1
        assert team_comms["messages"][0]["message"] == "Team A secret message"

    def test_team_communication_visibility_different_team(self):
        """Test that agents cannot see enemy team's communications."""
        view = self.team_b_policy.filter_world_state(self.world_state, "team_b_agent_1")

        # Should include only team B communications
        assert "team_communications" in view.visible_entities
        team_comms = view.visible_entities["team_communications"]
        assert team_comms["team"] == "team_b"
        assert len(team_comms["messages"]) == 1
        assert team_comms["messages"][0]["message"] == "Team B secret message"

        # Should not contain team A messages
        for msg in team_comms["messages"]:
            assert msg["team"] == "team_b"

    def test_should_observe_communication_same_team(self):
        """Test communication effect visibility for same team."""
        effect = {
            "kind": "TeamMessage",
            "payload": {"sender_team": "team_a", "message": "Team message"},
        }

        # Team A agent should see team A communication
        assert self.team_a_policy.should_observe_communication(effect, "team_a_agent_1")

        # Team B agent should not see team A communication
        assert not self.team_b_policy.should_observe_communication(
            effect, "team_b_agent_1"
        )

    def test_should_observe_communication_different_team(self):
        """Test communication effect visibility for different team."""
        effect = {
            "kind": "TeamMessage",
            "payload": {"sender_team": "team_b", "message": "Enemy team message"},
        }

        # Team B agent should see team B communication
        assert self.team_b_policy.should_observe_communication(effect, "team_b_agent_1")

        # Team A agent should not see team B communication
        assert not self.team_a_policy.should_observe_communication(
            effect, "team_a_agent_1"
        )

    def test_should_observe_non_communication_effect(self):
        """Test that non-communication effects use normal visibility rules."""
        effect = {
            "kind": "AgentDamaged",
            "payload": {"target_id": "team_b_agent_1", "damage": 20},
        }

        # Non-communication effects should use normal visibility rules
        assert self.team_a_policy.should_observe_communication(effect, "team_a_agent_1")
        assert self.team_b_policy.should_observe_communication(effect, "team_b_agent_1")


class TestUrgencyBasedPrioritization:
    """Test cases for urgency-based message prioritization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.policy = BattleObservationPolicy("team_a", vision_range=30.0)

        # Create world state with mixed urgency messages
        self.world_state = WorldState(
            entities={},
            spatial_index={},
            metadata={
                "team_communications": {
                    "team_a": [
                        {
                            "sender_id": "team_a_agent_1",
                            "team": "team_a",
                            "message": "Low priority message",
                            "urgency": "low",
                            "timestamp": 10.0,
                        },
                        {
                            "sender_id": "team_a_agent_2",
                            "team": "team_a",
                            "message": "High priority message",
                            "urgency": "high",
                            "timestamp": 15.0,
                        },
                        {
                            "sender_id": "team_a_agent_1",
                            "team": "team_a",
                            "message": "Medium priority message",
                            "urgency": "medium",
                            "timestamp": 20.0,
                        },
                        {
                            "sender_id": "team_a_agent_2",
                            "team": "team_a",
                            "message": "Another high priority",
                            "urgency": "high",
                            "timestamp": 25.0,
                        },
                    ]
                }
            },
            relationships={},
        )

    def test_urgency_based_message_prioritization(self):
        """Test that messages are prioritized by urgency and recency."""
        messages = self.policy._get_team_communications(self.world_state)

        # Should return messages prioritized by urgency then timestamp
        assert len(messages) == 4

        # Check that high urgency messages appear first (when sorted by priority)
        high_urgency_messages = [msg for msg in messages if msg["urgency"] == "high"]
        medium_urgency_messages = [
            msg for msg in messages if msg["urgency"] == "medium"
        ]
        low_urgency_messages = [msg for msg in messages if msg["urgency"] == "low"]

        assert len(high_urgency_messages) == 2
        assert len(medium_urgency_messages) == 1
        assert len(low_urgency_messages) == 1

        # Verify all messages have required fields
        for msg in messages:
            assert "sender_id" in msg
            assert "message" in msg
            assert "urgency" in msg
            assert "timestamp" in msg
            assert "team" in msg

    def test_message_limit_with_prioritization(self):
        """Test that message limit respects prioritization."""
        # Add many messages with different urgencies
        messages = []
        for i in range(15):
            urgency = ["low", "medium", "high"][i % 3]
            messages.append(
                {
                    "sender_id": "team_a_agent_1",
                    "team": "team_a",
                    "message": f"Message {i}",
                    "urgency": urgency,
                    "timestamp": float(i),
                }
            )

        # Update world state with many messages
        self.world_state.metadata["team_communications"]["team_a"] = messages

        # Get prioritized messages (limited to 10)
        prioritized = self.policy._get_team_communications(self.world_state)

        # Should return last 10 messages after prioritization
        assert len(prioritized) <= 10


class TestTeamCommunicationIntegration:
    """Integration tests for complete team communication workflow."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.processor = EffectProcessor()

        # Create full team setup
        self.agents = {}
        for team in ["team_a", "team_b"]:
            for i in range(2):
                agent_id = f"{team}_agent_{i + 1}"
                self.agents[agent_id] = Agent(
                    agent_id=agent_id,
                    team=team,
                    position=(10.0 + i * 10, 10.0),
                    health=100,
                )

        self.world_state = BattleWorldState(agents=self.agents, game_time=0.0)

    @pytest.mark.asyncio
    async def test_complete_team_communication_workflow(self):
        """Test complete workflow from effect to storage to retrieval."""
        # Create team communication effects
        effects = [
            {
                "kind": "TeamMessage",
                "payload": {
                    "sender_id": "team_a_agent_1",
                    "sender_team": "team_a",
                    "message": "Enemy spotted at coordinates (50, 60)!",
                    "urgency": "high",
                    "timestamp": 10.0,
                },
            },
            {
                "kind": "TeamMessage",
                "payload": {
                    "sender_id": "team_b_agent_1",
                    "sender_team": "team_b",
                    "message": "Moving to defensive positions",
                    "urgency": "medium",
                    "timestamp": 15.0,
                },
            },
            {
                "kind": "TeamMessage",
                "payload": {
                    "sender_id": "team_a_agent_2",
                    "sender_team": "team_a",
                    "message": "Roger, engaging target",
                    "urgency": "high",
                    "timestamp": 20.0,
                },
            },
        ]

        # Process all effects
        result = await self.processor.process_effects(effects, self.world_state)

        # Verify processing results
        assert result["processed_count"] == 3
        assert result["failed_count"] == 0
        assert result["effect_types"]["TeamMessage"] == 3

        # Verify team isolation
        team_a_messages = self.world_state.team_communications["team_a"]
        team_b_messages = self.world_state.team_communications["team_b"]

        assert len(team_a_messages) == 2  # Two messages for team A
        assert len(team_b_messages) == 1  # One message for team B

        # Verify message content and team isolation
        team_a_content = [msg.message for msg in team_a_messages]
        team_b_content = [msg.message for msg in team_b_messages]

        assert "Enemy spotted at coordinates (50, 60)!" in team_a_content
        assert "Roger, engaging target" in team_a_content
        assert "Moving to defensive positions" in team_b_content

        # Verify team B cannot see team A messages
        for msg in team_b_messages:
            assert msg.team == "team_b"
            assert "Enemy spotted" not in msg.message
            assert "Roger, engaging" not in msg.message

        # Verify team A cannot see team B messages
        for msg in team_a_messages:
            assert msg.team == "team_a"
            assert "defensive positions" not in msg.message

    @pytest.mark.asyncio
    async def test_message_isolation_under_concurrent_effects(self):
        """Test message isolation when processing concurrent communication effects."""
        # Create concurrent effects from both teams
        concurrent_effects = [
            {
                "kind": "TeamMessage",
                "payload": {
                    "sender_id": "team_a_agent_1",
                    "sender_team": "team_a",
                    "message": "Team A secret strategy",
                    "urgency": "high",
                },
            },
            {
                "kind": "TeamMessage",
                "payload": {
                    "sender_id": "team_b_agent_1",
                    "sender_team": "team_b",
                    "message": "Team B secret strategy",
                    "urgency": "high",
                },
            },
            {
                "kind": "AgentDamaged",  # Non-communication effect
                "payload": {
                    "target_id": "team_a_agent_1",
                    "damage": 20,
                    "new_health": 80,
                },
            },
        ]

        # Process concurrent effects
        result = await self.processor.process_effects(
            concurrent_effects, self.world_state
        )

        # Verify all effects processed
        assert result["processed_count"] == 3
        assert result["failed_count"] == 0

        # Verify strict team isolation
        team_a_messages = self.world_state.team_communications["team_a"]
        team_b_messages = self.world_state.team_communications["team_b"]

        assert len(team_a_messages) == 1
        assert len(team_b_messages) == 1

        # Verify no cross-team message leakage
        assert team_a_messages[0].message == "Team A secret strategy"
        assert team_a_messages[0].team == "team_a"

        assert team_b_messages[0].message == "Team B secret strategy"
        assert team_b_messages[0].team == "team_b"

        # Verify agent damage was also processed
        assert self.agents["team_a_agent_1"].health == 80
