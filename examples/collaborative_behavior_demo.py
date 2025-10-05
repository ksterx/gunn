#!/usr/bin/env python3
"""Collaborative behavior patterns demonstration.

This demo showcases emergent collaborative behaviors including:
- Following patterns (agents following each other)
- Helping behaviors (agents assisting others)
- Group conversations (multi-agent discussions)
- Task coordination (collaborative task completion)
- Resource sharing (coordinated resource usage)

Requirements demonstrated:
- 3.6: Multi-agent task coordination without explicit synchronization
- 4.6: Collaborative opportunities through observation
- 14.9: Coordination through observed actions and communication
"""

import asyncio
import time

from gunn import Orchestrator, OrchestratorConfig
from gunn.core.collaborative_patterns import (
    CollaborationDetector,
    CoordinationPatternTracker,
    detect_following_pattern,
    suggest_collaborative_action,
)
from gunn.core.conversational_agent import (
    ConversationalAgent,
)
from gunn.policies.observation import ObservationPolicy, PolicyConfig
from gunn.schemas.messages import View
from gunn.schemas.types import Effect, EffectDraft


class CollaborativeObservationPolicy(ObservationPolicy):
    """Observation policy that enables collaborative behavior detection."""

    def __init__(self, vision_range: float = 30.0):
        config = PolicyConfig(distance_limit=vision_range)
        super().__init__(config)
        self.vision_range = vision_range

    def filter_world_state(self, world_state, agent_id: str) -> View:
        """Filter world state based on vision range."""
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
                        # Add distance information
                        visible_entity = dict(entity_data)
                        visible_entity["distance"] = distance
                        visible_entities[entity_id] = visible_entity
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

    def should_observe_event(self, effect: Effect, agent_id: str, world_state) -> bool:
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


class CollaborativeAgent(ConversationalAgent):
    """Agent with collaborative behavior capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = CollaborationDetector(proximity_threshold=20.0)
        self.tracker = CoordinationPatternTracker()
        self.active_collaborations: list[str] = []

    async def process_observation(self, observation: View, agent_id: str):
        """Process observation with collaboration detection."""
        # Detect collaboration opportunities
        opportunities = self.detector.detect_opportunities(observation, agent_id)

        if opportunities:
            print(
                f"\n🤝 {self.name} detected {len(opportunities)} collaboration opportunities:"
            )
            for opp in opportunities[:3]:  # Show top 3
                print(f"   - {opp.opportunity_type}: {opp.description}")

                # Get action suggestion
                agent_pos = self.current_position
                suggestion = suggest_collaborative_action(opp, agent_id, agent_pos)

                if suggestion["action_type"] != "wait":
                    print(f"     → Suggested action: {suggestion['action_type']}")
                    print(f"     → Reasoning: {suggestion['reasoning']}")

        # Use parent class logic for actual intent generation
        return await super().process_observation(observation, agent_id)


async def demonstrate_following_pattern(orchestrator: Orchestrator):
    """Demonstrate following pattern between agents."""
    print("\n" + "=" * 70)
    print("🚶 DEMONSTRATION 1: Following Pattern")
    print("=" * 70)
    print("Scenario: Agent Bob follows Agent Alice as she moves around")
    print()

    policy = CollaborativeObservationPolicy(vision_range=30.0)

    # Register agents
    alice = await orchestrator.register_agent("agent_alice", policy)
    bob = await orchestrator.register_agent("agent_bob", policy)

    # Set initial positions
    orchestrator.world_state.entities.update(
        {
            "agent_alice": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
            "agent_bob": {"name": "Bob", "position": [3.0, 4.0, 0.0]},  # Distance = 5.0
        }
    )
    orchestrator.world_state.spatial_index.update(
        {
            "agent_alice": (0.0, 0.0, 0.0),
            "agent_bob": (3.0, 4.0, 0.0),
        }
    )

    print("Initial positions:")
    print("  Alice: (0.0, 0.0, 0.0)")
    print("  Bob: (3.0, 4.0, 0.0) - Distance: 5.0")

    # Check following pattern
    observation = await bob.get_current_observation()
    is_following = detect_following_pattern(
        observation, "agent_bob", "agent_alice", distance_threshold=6.0
    )

    print(f"\n✓ Following pattern detected: {is_following}")

    # Alice moves
    print("\nAlice moves to (10.0, 10.0, 0.0)...")
    orchestrator.world_state.entities["agent_alice"]["position"] = [10.0, 10.0, 0.0]
    orchestrator.world_state.spatial_index["agent_alice"] = (10.0, 10.0, 0.0)

    await orchestrator.broadcast_event(
        EffectDraft(
            kind="Move",
            payload={
                "agent_id": "agent_alice",
                "from": [0.0, 0.0, 0.0],
                "to": [10.0, 10.0, 0.0],
            },
            source_id="agent_alice",
            schema_version="1.0.0",
        )
    )

    # Bob observes and could follow
    observation = await bob.get_current_observation()
    is_following = detect_following_pattern(
        observation, "agent_bob", "agent_alice", distance_threshold=6.0
    )

    print(f"✗ Following pattern broken (too far): {not is_following}")
    print("\n✅ Following pattern demonstration complete")


async def demonstrate_helping_behavior(orchestrator: Orchestrator):
    """Demonstrate helping behavior detection."""
    print("\n" + "=" * 70)
    print("🆘 DEMONSTRATION 2: Helping Behavior")
    print("=" * 70)
    print("Scenario: Agent Charlie needs help, Agent Diana offers assistance")
    print()

    policy = CollaborativeObservationPolicy(vision_range=30.0)

    # Register agents
    charlie = await orchestrator.register_agent("agent_charlie", policy)
    diana = await orchestrator.register_agent("agent_diana", policy)

    # Set up world state with Charlie needing help
    orchestrator.world_state.entities.update(
        {
            "agent_charlie": {
                "name": "Charlie",
                "position": [0.0, 0.0, 0.0],
                "recent_message": {
                    "text": "I'm stuck and need help!",
                    "timestamp": time.time(),
                },
            },
            "agent_diana": {"name": "Diana", "position": [5.0, 5.0, 0.0]},
        }
    )
    orchestrator.world_state.spatial_index.update(
        {
            "agent_charlie": (0.0, 0.0, 0.0),
            "agent_diana": (5.0, 5.0, 0.0),
        }
    )

    print("Charlie's status: 'I'm stuck and need help!'")
    print("Diana's position: (5.0, 5.0, 0.0) - nearby")

    # Diana detects helping opportunity
    observation = await diana.get_current_observation()
    detector = CollaborationDetector()
    opportunities = detector.detect_opportunities(observation, "agent_diana")

    help_opps = [o for o in opportunities if o.opportunity_type == "helping"]

    if help_opps:
        opp = help_opps[0]
        print("\n✓ Diana detected helping opportunity:")
        print(f"  Description: {opp.description}")
        print(f"  Priority: {opp.priority}/10")

        # Get action suggestion
        suggestion = suggest_collaborative_action(opp, "agent_diana", (5.0, 5.0, 0.0))
        print(f"\n  Suggested action: {suggestion['action_type']}")
        print(f"  Message: {suggestion.get('text', 'N/A')}")

    print("\n✅ Helping behavior demonstration complete")


async def demonstrate_group_conversation(orchestrator: Orchestrator):
    """Demonstrate group conversation detection."""
    print("\n" + "=" * 70)
    print("💬 DEMONSTRATION 3: Group Conversation")
    print("=" * 70)
    print("Scenario: Multiple agents engage in a group conversation")
    print()

    policy = CollaborativeObservationPolicy(vision_range=30.0)

    # Register agents with unique IDs
    agents = []
    for name in ["Alice", "Bob", "Charlie"]:
        agent = await orchestrator.register_agent(f"agent_{name.lower()}_conv", policy)
        agents.append((name, agent))

    # Set up world state with active conversation
    orchestrator.world_state.entities.update(
        {
            "agent_alice_conv": {
                "name": "Alice",
                "position": [0.0, 0.0, 0.0],
                "recent_message": {"text": "Hello everyone!", "timestamp": time.time()},
            },
            "agent_bob_conv": {
                "name": "Bob",
                "position": [5.0, 5.0, 0.0],
                "recent_message": {"text": "Hi Alice!", "timestamp": time.time() + 1},
            },
            "agent_charlie_conv": {"name": "Charlie", "position": [8.0, 2.0, 0.0]},
        }
    )
    orchestrator.world_state.spatial_index.update(
        {
            "agent_alice_conv": (0.0, 0.0, 0.0),
            "agent_bob_conv": (5.0, 5.0, 0.0),
            "agent_charlie_conv": (8.0, 2.0, 0.0),
        }
    )

    print("Conversation:")
    print("  Alice: 'Hello everyone!'")
    print("  Bob: 'Hi Alice!'")
    print("  Charlie: (observing)")

    # Charlie detects conversation opportunity
    observation = await agents[2][1].get_current_observation()
    detector = CollaborationDetector()
    opportunities = detector.detect_opportunities(observation, "agent_charlie_conv")

    conv_opps = [o for o in opportunities if o.opportunity_type == "group_conversation"]

    if conv_opps:
        opp = conv_opps[0]
        print("\n✓ Charlie detected group conversation:")
        print(f"  Participants: {len(opp.involved_agents)} agents")
        print(f"  Description: {opp.description}")

        # Get action suggestion
        suggestion = suggest_collaborative_action(
            opp, "agent_charlie_conv", (8.0, 2.0, 0.0)
        )
        print(f"\n  Suggested action: {suggestion['action_type']}")
        print(f"  Message: {suggestion.get('text', 'N/A')}")

    print("\n✅ Group conversation demonstration complete")


async def demonstrate_task_coordination(orchestrator: Orchestrator):
    """Demonstrate task coordination without explicit synchronization."""
    print("\n" + "=" * 70)
    print("🎯 DEMONSTRATION 4: Task Coordination")
    print("=" * 70)
    print("Scenario: Agents coordinate to complete a collaborative task")
    print()

    policy = CollaborativeObservationPolicy(vision_range=30.0)

    # Register agents with unique IDs
    agents = []
    for name in ["Alice", "Bob"]:
        agent = await orchestrator.register_agent(f"agent_{name.lower()}_task", policy)
        agents.append((name, agent))

    # Set up world state with collaborative task
    orchestrator.world_state.entities.update(
        {
            "agent_alice_task": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
            "agent_bob_task": {"name": "Bob", "position": [5.0, 5.0, 0.0]},
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
            "agent_alice_task": (0.0, 0.0, 0.0),
            "agent_bob_task": (5.0, 5.0, 0.0),
        }
    )

    print("Task: 'Build a structure' (requires collaboration)")
    print("Difficulty: hard")
    print("Location: (2.0, 2.0, 0.0)")

    # Both agents independently detect the task
    detector = CollaborationDetector()
    tracker = CoordinationPatternTracker()

    for name, agent in agents:
        observation = await agent.get_current_observation()
        opportunities = detector.detect_opportunities(observation, agent.agent_id)

        task_opps = [
            o for o in opportunities if o.opportunity_type == "task_collaboration"
        ]

        if task_opps:
            opp = task_opps[0]
            print(f"\n✓ {name} detected task collaboration opportunity:")
            print(f"  Priority: {opp.priority}/10")

            # Get action suggestion
            agent_pos = orchestrator.world_state.spatial_index.get(agent.agent_id)
            suggestion = suggest_collaborative_action(opp, agent.agent_id, agent_pos)
            print(f"  Suggested action: {suggestion['action_type']}")
            print(f"  Message: {suggestion.get('text', 'N/A')}")

    # Track coordination pattern
    pattern_id = tracker.start_pattern(
        pattern_type="task_collaboration",
        initiator="agent_alice_task",
        participants=["agent_alice_task", "agent_bob_task"],
        metadata={"task_id": "task_1"},
    )

    print(f"\n✓ Coordination pattern established (ID: {pattern_id[:8]}...)")
    print("  Participants: Alice, Bob")
    print("  Type: task_collaboration")

    print("\n✅ Task coordination demonstration complete")


async def demonstrate_resource_sharing(orchestrator: Orchestrator):
    """Demonstrate resource sharing coordination."""
    print("\n" + "=" * 70)
    print("📦 DEMONSTRATION 5: Resource Sharing")
    print("=" * 70)
    print("Scenario: Agents coordinate to share a resource")
    print()

    policy = CollaborativeObservationPolicy(vision_range=30.0)

    # Register agents
    agents = []
    for name in ["Alice", "Bob"]:
        agent = await orchestrator.register_agent(f"agent_{name.lower()}_res", policy)
        agents.append((name, agent))

    # Set up world state with shared resource
    orchestrator.world_state.entities.update(
        {
            "agent_alice_res": {"name": "Alice", "position": [0.0, 0.0, 0.0]},
            "agent_bob_res": {"name": "Bob", "position": [5.0, 5.0, 0.0]},
            "resource_1": {
                "type": "resource",
                "name": "Information Hub",
                "position": [2.0, 2.0, 0.0],
            },
        }
    )
    orchestrator.world_state.spatial_index.update(
        {
            "agent_alice_res": (0.0, 0.0, 0.0),
            "agent_bob_res": (5.0, 5.0, 0.0),
        }
    )

    print("Resource: 'Information Hub'")
    print("Location: (2.0, 2.0, 0.0)")
    print("Nearby agents: Alice, Bob")

    # Detect resource sharing opportunity
    detector = CollaborationDetector(proximity_threshold=20.0)

    for name, agent in agents:
        observation = await agent.get_current_observation()
        opportunities = detector.detect_opportunities(observation, agent.agent_id)

        resource_opps = [
            o for o in opportunities if o.opportunity_type == "resource_sharing"
        ]

        if resource_opps:
            opp = resource_opps[0]
            print(f"\n✓ {name} detected resource sharing opportunity:")
            print(
                f"  Resource: {opp.metadata.get('resource_data', {}).get('name', 'Unknown')}"
            )
            print(f"  Involved agents: {len(opp.involved_agents)}")

            # Get action suggestion
            agent_pos = orchestrator.world_state.spatial_index.get(agent.agent_id)
            suggestion = suggest_collaborative_action(opp, agent.agent_id, agent_pos)
            print(f"  Suggested action: {suggestion['action_type']}")
            print(f"  Message: {suggestion.get('text', 'N/A')}")

    print("\n✅ Resource sharing demonstration complete")


async def main():
    """Run all collaborative behavior demonstrations."""
    print("🤝 COLLABORATIVE BEHAVIOR PATTERNS DEMONSTRATION")
    print("=" * 70)
    print("This demo showcases emergent collaborative behaviors:")
    print("  • Following patterns")
    print("  • Helping behaviors")
    print("  • Group conversations")
    print("  • Task coordination")
    print("  • Resource sharing")
    print()
    print("All coordination happens through observation without explicit")
    print("synchronization mechanisms.")
    print("=" * 70)

    # Create orchestrator
    config = OrchestratorConfig(use_in_memory_dedup=True)
    orchestrator = Orchestrator(config, world_id="collaborative_demo")
    await orchestrator.initialize()

    try:
        # Run demonstrations
        await demonstrate_following_pattern(orchestrator)
        await asyncio.sleep(0.5)

        await demonstrate_helping_behavior(orchestrator)
        await asyncio.sleep(0.5)

        await demonstrate_group_conversation(orchestrator)
        await asyncio.sleep(0.5)

        await demonstrate_task_coordination(orchestrator)
        await asyncio.sleep(0.5)

        await demonstrate_resource_sharing(orchestrator)

        print("\n" + "=" * 70)
        print("✅ ALL DEMONSTRATIONS COMPLETE")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  ✓ Agents detect collaboration opportunities through observation")
        print("  ✓ Coordination emerges without explicit synchronization")
        print("  ✓ Multiple collaboration patterns can coexist")
        print("  ✓ Agents make independent decisions based on shared context")
        print("  ✓ Collaborative behaviors scale naturally with agent count")
        print()
        print("Requirements demonstrated:")
        print("  ✓ 3.6: Multi-agent task coordination without explicit synchronization")
        print("  ✓ 4.6: Collaborative opportunities through observation")
        print("  ✓ 14.9: Coordination through observed actions and communication")
        print("=" * 70)

    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
