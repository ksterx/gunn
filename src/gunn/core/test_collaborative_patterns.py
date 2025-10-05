"""Unit tests for collaborative behavior patterns."""

import pytest

from gunn.core.collaborative_patterns import (
    CollaborationDetector,
    CollaborationOpportunity,
    CoordinationPatternTracker,
    detect_following_pattern,
    suggest_collaborative_action,
)
from gunn.schemas.messages import View


class TestCollaborationDetector:
    """Test suite for CollaborationDetector."""

    def test_detect_spatial_clustering(self):
        """Test detection of spatial clustering opportunities."""
        detector = CollaborationDetector(proximity_threshold=15.0)

        # Create observation with nearby agents
        observation = View(
            agent_id="agent_a",
            view_seq=1,
            visible_entities={
                "agent_a": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
                "agent_b": {"name": "Bob", "position": [5.0, 5.0, 0.0]},
                "agent_c": {"name": "Charlie", "position": [8.0, 2.0, 0.0]},
                "agent_d": {"name": "Diana", "position": [50.0, 50.0, 0.0]},  # Far away
            },
            visible_relationships={},
            context_digest="test",
        )

        opportunities = detector.detect_opportunities(observation, "agent_a")

        # Should detect spatial clustering with nearby agents
        spatial_opps = [
            o for o in opportunities if o.opportunity_type == "spatial_clustering"
        ]
        assert len(spatial_opps) > 0

        opp = spatial_opps[0]
        assert "agent_a" in opp.involved_agents
        assert "agent_b" in opp.involved_agents
        assert "agent_c" in opp.involved_agents
        assert "agent_d" not in opp.involved_agents  # Too far away

    def test_detect_task_collaboration(self):
        """Test detection of task collaboration opportunities."""
        detector = CollaborationDetector()

        # Create observation with a collaborative task
        observation = View(
            agent_id="agent_a",
            view_seq=1,
            visible_entities={
                "agent_a": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
                "agent_b": {"name": "Bob", "position": [5.0, 5.0, 0.0]},
                "task_1": {
                    "type": "task",
                    "description": "Build a structure",
                    "collaboration_required": True,
                    "difficulty": "hard",
                    "position": [2.0, 2.0, 0.0],
                },
            },
            visible_relationships={},
            context_digest="test",
        )

        opportunities = detector.detect_opportunities(observation, "agent_a")

        # Should detect task collaboration opportunity
        task_opps = [
            o for o in opportunities if o.opportunity_type == "task_collaboration"
        ]
        assert len(task_opps) > 0

        opp = task_opps[0]
        assert opp.description.lower().find("build a structure") >= 0
        assert "agent_a" in opp.involved_agents
        assert opp.priority >= 5  # Hard tasks have higher priority

    def test_detect_conversation_opportunities(self):
        """Test detection of group conversation opportunities."""
        detector = CollaborationDetector()

        # Create observation with multiple speakers
        observation = View(
            agent_id="agent_a",
            view_seq=1,
            visible_entities={
                "agent_a": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
                "agent_b": {
                    "name": "Bob",
                    "position": [5.0, 5.0, 0.0],
                    "recent_message": {"text": "Hello everyone!", "timestamp": 100.0},
                },
                "agent_c": {
                    "name": "Charlie",
                    "position": [8.0, 2.0, 0.0],
                    "recent_message": {"text": "Hi Bob!", "timestamp": 101.0},
                },
            },
            visible_relationships={},
            context_digest="test",
        )

        opportunities = detector.detect_opportunities(observation, "agent_a")

        # Should detect conversation opportunity
        conv_opps = [
            o for o in opportunities if o.opportunity_type == "group_conversation"
        ]
        assert len(conv_opps) > 0

        opp = conv_opps[0]
        assert len(opp.involved_agents) >= 3  # Alice + Bob + Charlie

    def test_detect_helping_opportunities(self):
        """Test detection of helping opportunities."""
        detector = CollaborationDetector()

        # Create observation with agent needing help
        observation = View(
            agent_id="agent_a",
            view_seq=1,
            visible_entities={
                "agent_a": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
                "agent_b": {
                    "name": "Bob",
                    "position": [5.0, 5.0, 0.0],
                    "recent_message": {
                        "text": "I'm stuck and need help!",
                        "timestamp": 100.0,
                    },
                },
            },
            visible_relationships={},
            context_digest="test",
        )

        opportunities = detector.detect_opportunities(observation, "agent_a")

        # Should detect helping opportunity
        help_opps = [o for o in opportunities if o.opportunity_type == "helping"]
        assert len(help_opps) > 0

        opp = help_opps[0]
        assert "agent_b" in opp.involved_agents
        assert opp.priority >= 7  # Helping has high priority

    def test_detect_resource_sharing(self):
        """Test detection of resource sharing opportunities."""
        detector = CollaborationDetector(proximity_threshold=20.0)

        # Create observation with shared resource
        observation = View(
            agent_id="agent_a",
            view_seq=1,
            visible_entities={
                "agent_a": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
                "agent_b": {"name": "Bob", "position": [5.0, 5.0, 0.0]},
                "resource_1": {
                    "type": "resource",
                    "name": "Information Hub",
                    "position": [2.0, 2.0, 0.0],
                },
            },
            visible_relationships={},
            context_digest="test",
        )

        opportunities = detector.detect_opportunities(observation, "agent_a")

        # Should detect resource sharing opportunity
        resource_opps = [
            o for o in opportunities if o.opportunity_type == "resource_sharing"
        ]
        assert len(resource_opps) > 0

        opp = resource_opps[0]
        assert "agent_a" in opp.involved_agents
        assert "agent_b" in opp.involved_agents

    def test_no_opportunities_when_alone(self):
        """Test that no collaboration opportunities are detected when agent is alone."""
        detector = CollaborationDetector()

        # Create observation with only the agent
        observation = View(
            agent_id="agent_a",
            view_seq=1,
            visible_entities={
                "agent_a": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
            },
            visible_relationships={},
            context_digest="test",
        )

        opportunities = detector.detect_opportunities(observation, "agent_a")

        # Should not detect any collaboration opportunities
        assert len(opportunities) == 0


class TestCoordinationPatternTracker:
    """Test suite for CoordinationPatternTracker."""

    def test_start_pattern(self):
        """Test starting a new coordination pattern."""
        tracker = CoordinationPatternTracker()

        pattern_id = tracker.start_pattern(
            pattern_type="following",
            initiator="agent_a",
            participants=["agent_a", "agent_b"],
            metadata={"distance": 5.0},
        )

        assert pattern_id in tracker.active_patterns
        pattern = tracker.active_patterns[pattern_id]
        assert pattern.pattern_type == "following"
        assert pattern.initiator == "agent_a"
        assert "agent_b" in pattern.participants
        assert pattern.status == "active"

    def test_update_pattern(self):
        """Test updating an existing pattern."""
        tracker = CoordinationPatternTracker()

        pattern_id = tracker.start_pattern(
            pattern_type="helping",
            initiator="agent_a",
            participants=["agent_a", "agent_b"],
        )

        # Update pattern status
        success = tracker.update_pattern(pattern_id, status="completed")
        assert success
        assert pattern_id not in tracker.active_patterns  # Moved to history
        assert len(tracker.pattern_history) == 1

    def test_get_active_patterns(self):
        """Test retrieving active patterns."""
        tracker = CoordinationPatternTracker()

        # Create multiple patterns
        pattern_id_1 = tracker.start_pattern(
            pattern_type="following",
            initiator="agent_a",
            participants=["agent_a", "agent_b"],
        )

        pattern_id_2 = tracker.start_pattern(
            pattern_type="helping",
            initiator="agent_c",
            participants=["agent_c", "agent_d"],
        )

        # Get all active patterns
        all_patterns = tracker.get_active_patterns()
        assert len(all_patterns) == 2

        # Filter by agent
        agent_a_patterns = tracker.get_active_patterns(agent_id="agent_a")
        assert len(agent_a_patterns) == 1
        assert agent_a_patterns[0][0] == pattern_id_1

        # Filter by type
        following_patterns = tracker.get_active_patterns(pattern_type="following")
        assert len(following_patterns) == 1
        assert following_patterns[0][0] == pattern_id_1

    def test_is_agent_coordinating(self):
        """Test checking if agent is coordinating."""
        tracker = CoordinationPatternTracker()

        tracker.start_pattern(
            pattern_type="following",
            initiator="agent_a",
            participants=["agent_a", "agent_b"],
        )

        assert tracker.is_agent_coordinating("agent_a")
        assert tracker.is_agent_coordinating("agent_b")
        assert not tracker.is_agent_coordinating("agent_c")

    def test_get_coordination_partners(self):
        """Test getting coordination partners."""
        tracker = CoordinationPatternTracker()

        tracker.start_pattern(
            pattern_type="group_task",
            initiator="agent_a",
            participants=["agent_a", "agent_b", "agent_c"],
        )

        partners = tracker.get_coordination_partners("agent_a")
        assert "agent_b" in partners
        assert "agent_c" in partners
        assert "agent_a" not in partners  # Should not include self

    def test_pattern_history_limit(self):
        """Test that pattern history is limited."""
        tracker = CoordinationPatternTracker()
        tracker.max_history_size = 5

        # Create and complete more patterns than the limit
        for i in range(10):
            pattern_id = tracker.start_pattern(
                pattern_type="test",
                initiator=f"agent_{i}",
                participants=[f"agent_{i}"],
            )
            tracker.update_pattern(pattern_id, status="completed")

        # History should be limited
        assert len(tracker.pattern_history) == 5


class TestFollowingPattern:
    """Test suite for following pattern detection."""

    def test_detect_following_within_threshold(self):
        """Test detecting following when agents are close."""
        observation = View(
            agent_id="agent_a",
            view_seq=1,
            visible_entities={
                "agent_a": {"position": [0.0, 0.0, 0.0]},
                "agent_b": {"position": [3.0, 4.0, 0.0]},  # Distance = 5.0
            },
            visible_relationships={},
            context_digest="test",
        )

        is_following = detect_following_pattern(
            observation, "agent_a", "agent_b", distance_threshold=6.0
        )
        assert is_following

    def test_detect_not_following_beyond_threshold(self):
        """Test not detecting following when agents are far apart."""
        observation = View(
            agent_id="agent_a",
            view_seq=1,
            visible_entities={
                "agent_a": {"position": [0.0, 0.0, 0.0]},
                "agent_b": {"position": [10.0, 10.0, 0.0]},  # Distance > 14
            },
            visible_relationships={},
            context_digest="test",
        )

        is_following = detect_following_pattern(
            observation, "agent_a", "agent_b", distance_threshold=5.0
        )
        assert not is_following

    def test_detect_following_missing_position(self):
        """Test following detection when position is missing."""
        observation = View(
            agent_id="agent_a",
            view_seq=1,
            visible_entities={
                "agent_a": {"name": "Alice"},  # No position
                "agent_b": {"position": [10.0, 10.0, 0.0]},
            },
            visible_relationships={},
            context_digest="test",
        )

        is_following = detect_following_pattern(
            observation, "agent_a", "agent_b", distance_threshold=5.0
        )
        assert not is_following


class TestCollaborativeActionSuggestions:
    """Test suite for collaborative action suggestions."""

    def test_suggest_move_to_cluster(self):
        """Test suggesting movement to join a cluster."""
        opportunity = CollaborationOpportunity(
            opportunity_type="spatial_clustering",
            description="Agents nearby",
            involved_agents=["agent_a", "agent_b"],
            location=(10.0, 10.0, 0.0),
            priority=5,
        )

        agent_position = (0.0, 0.0, 0.0)
        suggestion = suggest_collaborative_action(
            opportunity, "agent_a", agent_position
        )

        assert suggestion["action_type"] == "move"
        assert suggestion["target_position"] == [10.0, 10.0, 0.0]

    def test_suggest_speak_when_at_cluster(self):
        """Test suggesting conversation when already at cluster."""
        opportunity = CollaborationOpportunity(
            opportunity_type="spatial_clustering",
            description="Agents nearby",
            involved_agents=["agent_a", "agent_b"],
            location=(1.0, 1.0, 0.0),
            priority=5,
        )

        agent_position = (0.0, 0.0, 0.0)  # Very close to cluster
        suggestion = suggest_collaborative_action(
            opportunity, "agent_a", agent_position
        )

        assert suggestion["action_type"] == "speak"
        assert "collaborate" in suggestion["text"].lower()

    def test_suggest_task_collaboration(self):
        """Test suggesting task collaboration."""
        opportunity = CollaborationOpportunity(
            opportunity_type="task_collaboration",
            description="Task requires help",
            involved_agents=["agent_a", "agent_b"],
            priority=7,
            metadata={"task_data": {"description": "build structure"}},
        )

        suggestion = suggest_collaborative_action(opportunity, "agent_a", None)

        assert suggestion["action_type"] == "speak"
        assert "help" in suggestion["text"].lower()

    def test_suggest_helping(self):
        """Test suggesting helping action."""
        opportunity = CollaborationOpportunity(
            opportunity_type="helping",
            description="Agent needs help",
            involved_agents=["agent_a", "agent_b"],
            priority=8,
            metadata={"target_agent": "agent_b"},
        )

        suggestion = suggest_collaborative_action(opportunity, "agent_a", None)

        assert suggestion["action_type"] == "speak"
        assert "help" in suggestion["text"].lower()

    def test_suggest_resource_sharing(self):
        """Test suggesting resource sharing."""
        opportunity = CollaborationOpportunity(
            opportunity_type="resource_sharing",
            description="Resource available",
            involved_agents=["agent_a", "agent_b"],
            priority=5,
            metadata={"resource_data": {"name": "Information Hub"}},
        )

        suggestion = suggest_collaborative_action(opportunity, "agent_a", None)

        assert suggestion["action_type"] == "speak"
        assert (
            "coordinate" in suggestion["text"].lower()
            or "resource" in suggestion["text"].lower()
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
