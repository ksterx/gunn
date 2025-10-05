"""Integration tests for collaborative behavior patterns.

This module tests emergent collaborative behaviors including:
- Following patterns
- Helping behaviors
- Group conversations
- Task coordination
- Resource sharing

Requirements tested:
- 3.6: Multi-agent task coordination without explicit synchronization
- 4.6: Collaborative opportunities through observation
- 14.9: Coordination through observed actions and communication
"""

import pytest

from gunn import Orchestrator, OrchestratorConfig
from gunn.core.collaborative_patterns import (
    CollaborationDetector,
    CoordinationPatternTracker,
    detect_following_pattern,
    suggest_collaborative_action,
)
from gunn.policies.observation import ObservationPolicy, PolicyConfig
from gunn.schemas.types import EffectDraft


class CollaborativeObservationPolicy(ObservationPolicy):
    """Observation policy for collaborative scenarios."""

    def __init__(self, vision_range: float = 30.0):
        config = PolicyConfig(distance_limit=vision_range)
        super().__init__(config)
        self.vision_range = vision_range

    def filter_world_state(self, world_state, agent_id: str):
        """Filter world state based on vision range."""
        from gunn.schemas.messages import View

        visible_entities = {}
        visible_relationships = {}

        # Get agent's position
        agent_pos = world_state.spatial_index.get(agent_id)

        if agent_pos:
            # Filter entities by distance
            for entity_id, entity_data in world_state.entities.items():
                entity_pos = world_state.spatial_index.get(entity_id)

                if entity_pos:
                    # Calculate distance
                    distance = (
                        sum(
                            (a - b) ** 2
                            for a, b in zip(agent_pos, entity_pos, strict=False)
                        )
                        ** 0.5
                    )

                    if distance <= self.vision_range or entity_id == agent_id:
                        visible_entities[entity_id] = entity_data
                else:
                    # Non-spatial entities are always visible
                    visible_entities[entity_id] = entity_data
        else:
            # If agent has no position, show all entities
            visible_entities = dict(world_state.entities)

        # Filter relationships
        for entity_id, related_ids in world_state.relationships.items():
            if entity_id in visible_entities:
                visible_related = [
                    rid for rid in related_ids if rid in visible_entities
                ]
                if visible_related:
                    visible_relationships[entity_id] = visible_related

        return View(
            agent_id=agent_id,
            view_seq=0,
            visible_entities=visible_entities,
            visible_relationships=visible_relationships,
            context_digest=f"collab_view_{agent_id}_{len(visible_entities)}",
        )

    def should_observe_event(self, effect, agent_id: str, world_state):
        """Determine if agent should observe this effect."""
        # Always observe own effects
        if effect.get("source_id") == agent_id:
            return True

        # Check if effect involves nearby entities
        payload = effect.get("payload", {})

        # For speak events, check if speaker is nearby
        if effect["kind"] == "Speak":
            speaker_id = payload.get("agent_id")
            if speaker_id:
                agent_pos = world_state.spatial_index.get(agent_id)
                speaker_pos = world_state.spatial_index.get(speaker_id)

                if agent_pos and speaker_pos:
                    distance = (
                        sum(
                            (a - b) ** 2
                            for a, b in zip(agent_pos, speaker_pos, strict=False)
                        )
                        ** 0.5
                    )
                    return distance <= self.vision_range

        # Environmental events are visible to all
        if effect["kind"] in ["EnvironmentalEvent", "TaskEvent", "CollaborativeEvent"]:
            return True

        return False


@pytest.mark.asyncio
async def test_spatial_clustering_detection():
    """Test detection of spatial clustering opportunities."""
    config = OrchestratorConfig(use_in_memory_dedup=True)
    orchestrator = Orchestrator(config, world_id="collab_test_spatial")
    await orchestrator.initialize()

    try:
        # Create observation policy
        policy = CollaborativeObservationPolicy(vision_range=30.0)

        # Register agents
        agent_a = await orchestrator.register_agent("agent_a", policy)
        agent_b = await orchestrator.register_agent("agent_b", policy)
        agent_c = await orchestrator.register_agent("agent_c", policy)

        # Set up world state with agents in close proximity
        orchestrator.world_state.entities.update(
            {
                "agent_a": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
                "agent_b": {"name": "Bob", "position": [5.0, 5.0, 0.0]},
                "agent_c": {"name": "Charlie", "position": [8.0, 2.0, 0.0]},
            }
        )
        orchestrator.world_state.spatial_index.update(
            {
                "agent_a": (0.0, 0.0, 0.0),
                "agent_b": (5.0, 5.0, 0.0),
                "agent_c": (8.0, 2.0, 0.0),
            }
        )

        # Get observation for agent_a
        observation = await agent_a.get_current_observation()

        # Detect collaboration opportunities
        detector = CollaborationDetector(proximity_threshold=15.0)
        opportunities = detector.detect_opportunities(observation, "agent_a")

        # Should detect spatial clustering
        spatial_opps = [
            o for o in opportunities if o.opportunity_type == "spatial_clustering"
        ]
        assert len(spatial_opps) > 0

        opp = spatial_opps[0]
        assert "agent_a" in opp.involved_agents
        assert "agent_b" in opp.involved_agents
        assert "agent_c" in opp.involved_agents

    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_task_collaboration_scenario():
    """Test collaborative task coordination."""
    config = OrchestratorConfig(use_in_memory_dedup=True)
    orchestrator = Orchestrator(config, world_id="collab_test_task")
    await orchestrator.initialize()

    try:
        policy = CollaborativeObservationPolicy(vision_range=30.0)

        # Register agents
        agent_a = await orchestrator.register_agent("agent_a", policy)
        agent_b = await orchestrator.register_agent("agent_b", policy)

        # Set up world state
        orchestrator.world_state.entities.update(
            {
                "agent_a": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
                "agent_b": {"name": "Bob", "position": [5.0, 5.0, 0.0]},
                "task_1": {
                    "type": "task",
                    "description": "Build a structure",
                    "collaboration_required": True,
                    "difficulty": "hard",
                    "position": [2.0, 2.0, 0.0],
                },
            }
        )
        orchestrator.world_state.spatial_index.update(
            {
                "agent_a": (0.0, 0.0, 0.0),
                "agent_b": (5.0, 5.0, 0.0),
            }
        )

        # Broadcast task event
        await orchestrator.broadcast_event(
            EffectDraft(
                kind="TaskEvent",
                payload={
                    "task_id": "task_1",
                    "description": "Build a structure",
                    "collaboration_required": True,
                    "difficulty": "hard",
                },
                source_id="environment",
                schema_version="1.0.0",
            )
        )

        # Get observation
        observation = await agent_a.get_current_observation()

        # Detect collaboration opportunities
        detector = CollaborationDetector()
        opportunities = detector.detect_opportunities(observation, "agent_a")

        # Should detect task collaboration opportunity
        task_opps = [
            o for o in opportunities if o.opportunity_type == "task_collaboration"
        ]
        assert len(task_opps) > 0

        opp = task_opps[0]
        assert "agent_a" in opp.involved_agents
        assert opp.priority >= 5  # Hard tasks have higher priority

        # Get action suggestion
        suggestion = suggest_collaborative_action(opp, "agent_a", (0.0, 0.0, 0.0))
        assert suggestion["action_type"] == "speak"
        assert "help" in suggestion["text"].lower()

    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_following_pattern():
    """Test following pattern detection and tracking."""
    config = OrchestratorConfig(use_in_memory_dedup=True)
    orchestrator = Orchestrator(config, world_id="collab_test_following")
    await orchestrator.initialize()

    try:
        policy = CollaborativeObservationPolicy(vision_range=30.0)

        # Register agents
        agent_a = await orchestrator.register_agent("agent_a", policy)
        agent_b = await orchestrator.register_agent("agent_b", policy)

        # Set up initial positions (agent_a following agent_b)
        orchestrator.world_state.entities.update(
            {
                "agent_a": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
                "agent_b": {
                    "name": "Bob",
                    "position": [3.0, 4.0, 0.0],
                },  # Distance = 5.0
            }
        )
        orchestrator.world_state.spatial_index.update(
            {
                "agent_a": (0.0, 0.0, 0.0),
                "agent_b": (3.0, 4.0, 0.0),
            }
        )

        # Get observation
        observation = await agent_a.get_current_observation()

        # Detect following pattern
        is_following = detect_following_pattern(
            observation, "agent_a", "agent_b", distance_threshold=6.0
        )
        assert is_following

        # Track the pattern
        tracker = CoordinationPatternTracker()
        pattern_id = tracker.start_pattern(
            pattern_type="following",
            initiator="agent_a",
            participants=["agent_a", "agent_b"],
            metadata={"distance": 5.0},
        )

        # Verify pattern is tracked
        assert tracker.is_agent_coordinating("agent_a")
        assert tracker.is_agent_coordinating("agent_b")

        partners = tracker.get_coordination_partners("agent_a")
        assert "agent_b" in partners

        # Simulate agent_b moving
        orchestrator.world_state.entities["agent_b"]["position"] = [10.0, 10.0, 0.0]
        orchestrator.world_state.spatial_index["agent_b"] = (10.0, 10.0, 0.0)

        # Get new observation
        observation = await agent_a.get_current_observation()

        # Following pattern should break (too far)
        is_following = detect_following_pattern(
            observation, "agent_a", "agent_b", distance_threshold=6.0
        )
        assert not is_following

        # Update pattern status
        tracker.update_pattern(pattern_id, status="completed")
        assert not tracker.is_agent_coordinating("agent_a")

    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_helping_behavior():
    """Test helping behavior detection and coordination."""
    config = OrchestratorConfig(use_in_memory_dedup=True)
    orchestrator = Orchestrator(config, world_id="collab_test_helping")
    await orchestrator.initialize()

    try:
        policy = CollaborativeObservationPolicy(vision_range=30.0)

        # Register agents
        agent_a = await orchestrator.register_agent("agent_a", policy)
        agent_b = await orchestrator.register_agent("agent_b", policy)

        # Set up world state with agent_b needing help
        orchestrator.world_state.entities.update(
            {
                "agent_a": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
                "agent_b": {
                    "name": "Bob",
                    "position": [5.0, 5.0, 0.0],
                    "recent_message": {
                        "text": "I'm stuck and need help!",
                        "timestamp": 100.0,
                    },
                },
            }
        )
        orchestrator.world_state.spatial_index.update(
            {
                "agent_a": (0.0, 0.0, 0.0),
                "agent_b": (5.0, 5.0, 0.0),
            }
        )

        # Get observation
        observation = await agent_a.get_current_observation()

        # Detect helping opportunity
        detector = CollaborationDetector()
        opportunities = detector.detect_opportunities(observation, "agent_a")

        help_opps = [o for o in opportunities if o.opportunity_type == "helping"]
        assert len(help_opps) > 0

        opp = help_opps[0]
        assert "agent_b" in opp.involved_agents
        assert opp.priority >= 7  # Helping has high priority

        # Get action suggestion
        suggestion = suggest_collaborative_action(opp, "agent_a", (0.0, 0.0, 0.0))
        assert suggestion["action_type"] == "speak"
        assert "help" in suggestion["text"].lower()

        # Track helping pattern
        tracker = CoordinationPatternTracker()
        pattern_id = tracker.start_pattern(
            pattern_type="helping",
            initiator="agent_a",
            participants=["agent_a", "agent_b"],
            metadata={"help_reason": "stuck"},
        )

        assert tracker.is_agent_coordinating("agent_a")
        assert tracker.is_agent_coordinating("agent_b")

    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_group_conversation():
    """Test group conversation detection and coordination."""
    config = OrchestratorConfig(use_in_memory_dedup=True)
    orchestrator = Orchestrator(config, world_id="collab_test_conversation")
    await orchestrator.initialize()

    try:
        policy = CollaborativeObservationPolicy(vision_range=30.0)

        # Register agents
        agent_a = await orchestrator.register_agent("agent_a", policy)
        agent_b = await orchestrator.register_agent("agent_b", policy)
        agent_c = await orchestrator.register_agent("agent_c", policy)

        # Set up world state with active conversation
        orchestrator.world_state.entities.update(
            {
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
            }
        )
        orchestrator.world_state.spatial_index.update(
            {
                "agent_a": (0.0, 0.0, 0.0),
                "agent_b": (5.0, 5.0, 0.0),
                "agent_c": (8.0, 2.0, 0.0),
            }
        )

        # Get observation
        observation = await agent_a.get_current_observation()

        # Detect conversation opportunity
        detector = CollaborationDetector()
        opportunities = detector.detect_opportunities(observation, "agent_a")

        conv_opps = [
            o for o in opportunities if o.opportunity_type == "group_conversation"
        ]
        assert len(conv_opps) > 0

        opp = conv_opps[0]
        assert len(opp.involved_agents) >= 3  # Alice + Bob + Charlie

        # Get action suggestion
        suggestion = suggest_collaborative_action(opp, "agent_a", (0.0, 0.0, 0.0))
        assert suggestion["action_type"] == "speak"

        # Track conversation pattern
        tracker = CoordinationPatternTracker()
        pattern_id = tracker.start_pattern(
            pattern_type="group_conversation",
            initiator="agent_b",
            participants=["agent_a", "agent_b", "agent_c"],
            metadata={"topic": "general"},
        )

        # All agents should be coordinating
        assert tracker.is_agent_coordinating("agent_a")
        assert tracker.is_agent_coordinating("agent_b")
        assert tracker.is_agent_coordinating("agent_c")

        # Check coordination partners
        partners_a = tracker.get_coordination_partners("agent_a")
        assert "agent_b" in partners_a
        assert "agent_c" in partners_a

    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_resource_sharing():
    """Test resource sharing detection and coordination."""
    config = OrchestratorConfig(use_in_memory_dedup=True)
    orchestrator = Orchestrator(config, world_id="collab_test_resource")
    await orchestrator.initialize()

    try:
        policy = CollaborativeObservationPolicy(vision_range=30.0)

        # Register agents
        agent_a = await orchestrator.register_agent("agent_a", policy)
        agent_b = await orchestrator.register_agent("agent_b", policy)

        # Set up world state with shared resource
        orchestrator.world_state.entities.update(
            {
                "agent_a": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
                "agent_b": {"name": "Bob", "position": [5.0, 5.0, 0.0]},
                "resource_1": {
                    "type": "resource",
                    "name": "Information Hub",
                    "position": [2.0, 2.0, 0.0],
                },
            }
        )
        orchestrator.world_state.spatial_index.update(
            {
                "agent_a": (0.0, 0.0, 0.0),
                "agent_b": (5.0, 5.0, 0.0),
            }
        )

        # Get observation
        observation = await agent_a.get_current_observation()

        # Detect resource sharing opportunity
        detector = CollaborationDetector(proximity_threshold=20.0)
        opportunities = detector.detect_opportunities(observation, "agent_a")

        resource_opps = [
            o for o in opportunities if o.opportunity_type == "resource_sharing"
        ]
        assert len(resource_opps) > 0

        opp = resource_opps[0]
        assert "agent_a" in opp.involved_agents
        assert "agent_b" in opp.involved_agents

        # Get action suggestion
        suggestion = suggest_collaborative_action(opp, "agent_a", (0.0, 0.0, 0.0))
        assert suggestion["action_type"] == "speak"
        assert (
            "resource" in suggestion["text"].lower()
            or "coordinate" in suggestion["text"].lower()
        )

        # Track resource sharing pattern
        tracker = CoordinationPatternTracker()
        pattern_id = tracker.start_pattern(
            pattern_type="resource_sharing",
            initiator="agent_a",
            participants=["agent_a", "agent_b"],
            metadata={"resource_id": "resource_1"},
        )

        assert tracker.is_agent_coordinating("agent_a")
        assert tracker.is_agent_coordinating("agent_b")

    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_emergent_coordination_without_synchronization():
    """Test that agents can coordinate without explicit synchronization.

    This test demonstrates that agents can detect and respond to collaboration
    opportunities purely through observation, without any explicit coordination
    protocol or synchronization mechanism.
    """
    config = OrchestratorConfig(use_in_memory_dedup=True)
    orchestrator = Orchestrator(config, world_id="collab_test_emergent")
    await orchestrator.initialize()

    try:
        policy = CollaborativeObservationPolicy(vision_range=30.0)

        # Register multiple agents
        agents = []
        for i in range(4):
            agent = await orchestrator.register_agent(f"agent_{i}", policy)
            agents.append(agent)

        # Set up world state with agents scattered
        orchestrator.world_state.entities.update(
            {
                "agent_0": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
                "agent_1": {"name": "Bob", "position": [5.0, 5.0, 0.0]},
                "agent_2": {"name": "Charlie", "position": [8.0, 2.0, 0.0]},
                "agent_3": {"name": "Diana", "position": [50.0, 50.0, 0.0]},  # Far away
            }
        )
        orchestrator.world_state.spatial_index.update(
            {
                "agent_0": (0.0, 0.0, 0.0),
                "agent_1": (5.0, 5.0, 0.0),
                "agent_2": (8.0, 2.0, 0.0),
                "agent_3": (50.0, 50.0, 0.0),
            }
        )

        # Each agent independently detects collaboration opportunities
        detector = CollaborationDetector(proximity_threshold=15.0)
        tracker = CoordinationPatternTracker()

        for i, agent in enumerate(agents[:3]):  # First 3 agents are nearby
            observation = await agent.get_current_observation()
            opportunities = detector.detect_opportunities(observation, f"agent_{i}")

            # Each agent should detect the same spatial clustering
            spatial_opps = [
                o for o in opportunities if o.opportunity_type == "spatial_clustering"
            ]
            assert len(spatial_opps) > 0

            # Agents can independently decide to coordinate
            if i == 0:  # First agent initiates
                pattern_id = tracker.start_pattern(
                    pattern_type="spatial_clustering",
                    initiator=f"agent_{i}",
                    participants=[f"agent_{j}" for j in range(3)],
                )

        # Verify emergent coordination
        assert tracker.is_agent_coordinating("agent_0")
        assert tracker.is_agent_coordinating("agent_1")
        assert tracker.is_agent_coordinating("agent_2")
        assert not tracker.is_agent_coordinating("agent_3")  # Too far away

        # Verify coordination partners
        partners_0 = tracker.get_coordination_partners("agent_0")
        assert "agent_1" in partners_0
        assert "agent_2" in partners_0
        assert "agent_3" not in partners_0

    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
