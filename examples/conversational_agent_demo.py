#!/usr/bin/env python3
"""Demonstration of conversational agent implementation.

This example shows how to use the ConversationalAgent class to create
agents that can engage in natural conversations, build context from
observations, and make LLM-driven decisions in multi-agent environments.
"""

import asyncio
import time

from gunn import (
    ConversationalAgent,
    MockLLMClient,
    Orchestrator,
    OrchestratorConfig,
)
from gunn.policies.observation import ObservationPolicy, PolicyConfig
from gunn.schemas.messages import View, WorldState
from gunn.schemas.types import Effect


class ConversationObservationPolicy(ObservationPolicy):
    """Observation policy optimized for conversational agents."""

    def __init__(self):
        config = PolicyConfig(distance_limit=20.0)
        super().__init__(config)

    def filter_world_state(self, world_state: WorldState, agent_id: str) -> View:
        """Return a view with nearby entities and recent messages."""
        # For this demo, show all entities to all agents
        # In a real implementation, you'd filter by distance and relationships
        return View(
            agent_id=agent_id,
            view_seq=0,
            visible_entities=world_state.entities,
            visible_relationships=world_state.relationships,
            context_digest=f"view_{agent_id}_{len(world_state.entities)}",
        )

    def should_observe_event(
        self, effect: Effect, agent_id: str, world_state: WorldState
    ) -> bool:
        """All agents observe all events for conversation purposes."""
        return True


class EnhancedMockLLMClient(MockLLMClient):
    """Enhanced mock LLM client with more realistic conversation patterns."""

    def __init__(self, agent_name: str, response_delay: float = 0.2):
        super().__init__(response_delay)
        self.agent_name = agent_name
        self.conversation_topics = [
            "the weather",
            "recent events",
            "plans for the day",
            "interesting observations",
            "collaborative projects",
        ]
        self.greetings = [
            f"Hello! I'm {agent_name}.",
            f"Hi there! {agent_name} here.",
            f"Greetings! Nice to meet you, I'm {agent_name}.",
        ]
        self.responses = [
            "That's interesting!",
            "I agree with that.",
            "Tell me more about that.",
            "I have a different perspective on that.",
            "That reminds me of something similar.",
        ]

    async def generate_response(
        self,
        context: str,
        personality: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
    ) -> "LLMResponse":
        """Generate more realistic conversational responses."""
        import random

        from gunn.core.conversational_agent import LLMResponse

        await asyncio.sleep(self.response_delay)
        context_lower = context.lower()

        # Respond to greetings
        if any(greeting in context_lower for greeting in ["hello", "hi", "greetings"]):
            if random.random() < 0.8:
                greeting = random.choice(self.greetings)
                return LLMResponse(
                    action_type="speak",
                    text=greeting,
                    reasoning="Responding to greeting",
                )

        # Respond to questions or mentions of agent name
        if self.agent_name.lower() in context_lower or "?" in context:
            if random.random() < 0.9:
                response = random.choice(self.responses)
                return LLMResponse(
                    action_type="speak",
                    text=f"{response} What do you think about {random.choice(self.conversation_topics)}?",
                    reasoning="Responding to question or mention",
                )

        # Start conversations when seeing other agents
        if "nearby agents" in context_lower and "0)" not in context_lower:
            if random.random() < 0.4:
                topic = random.choice(self.conversation_topics)
                return LLMResponse(
                    action_type="speak",
                    text=f"I've been thinking about {topic}. Anyone else have thoughts on this?",
                    reasoning="Starting conversation with nearby agents",
                )

        # Move occasionally to explore
        if random.random() < 0.2:
            return LLMResponse(
                action_type="move",
                target_position=[random.uniform(-15, 15), random.uniform(-15, 15), 0.0],
                reasoning="Exploring the area",
            )

        # Sometimes just observe
        return LLMResponse(action_type="wait", reasoning="Listening and observing")


async def simulate_message_events(orchestrator: Orchestrator, duration: float):
    """Simulate external message events to trigger conversations."""
    start_time = time.time()

    while time.time() - start_time < duration:
        await asyncio.sleep(5.0)  # Every 5 seconds

        # Simulate an external event that agents can observe
        from gunn.schemas.types import EffectDraft

        event_messages = [
            "A gentle breeze rustles through the area.",
            "The sun shifts position, casting new shadows.",
            "A distant sound echoes across the landscape.",
            "The temperature seems to have changed slightly.",
            "Something interesting appears on the horizon.",
        ]

        import random

        message = random.choice(event_messages)

        await orchestrator.broadcast_event(
            EffectDraft(
                kind="EnvironmentalEvent",
                payload={
                    "description": message,
                    "timestamp": time.time(),
                    "event_type": "observation",
                },
                source_id="environment",
                schema_version="1.0.0",
            )
        )


async def main():
    """Run the conversational agent demo."""
    print("🗣️  Starting Conversational Agent Demo")
    print("=" * 50)

    # Create orchestrator
    config = OrchestratorConfig(
        use_in_memory_dedup=True, max_agents=10, staleness_threshold=0
    )
    orchestrator = Orchestrator(config, world_id="conversation_demo")
    await orchestrator.initialize()

    try:
        # Create observation policy
        policy = ConversationObservationPolicy()

        # Create conversational agents with different personalities
        agents = []
        agent_configs = [
            {
                "name": "Alice",
                "personality": "curious and inquisitive, loves asking questions",
                "response_prob": 0.8,
                "move_prob": 0.1,
            },
            {
                "name": "Bob",
                "personality": "thoughtful and analytical, provides detailed responses",
                "response_prob": 0.6,
                "move_prob": 0.2,
            },
            {
                "name": "Charlie",
                "personality": "energetic and social, initiates conversations",
                "response_prob": 0.9,
                "move_prob": 0.3,
            },
        ]

        for config_data in agent_configs:
            agent_id = f"agent_{config_data['name'].lower()}"

            # Create enhanced LLM client for this agent
            llm_client = EnhancedMockLLMClient(
                agent_name=config_data["name"],
                response_delay=0.1 + len(agents) * 0.05,  # Stagger response times
            )

            # Create conversational agent
            agent_logic = ConversationalAgent(
                llm_client=llm_client,
                personality=config_data["personality"],
                name=config_data["name"],
                conversation_distance=25.0,
                response_probability=config_data["response_prob"],
                movement_probability=config_data["move_prob"],
            )

            # Register with orchestrator
            handle = await orchestrator.register_agent(agent_id, policy)
            agents.append((handle, agent_logic, config_data["name"]))

        # Initialize world state with agent positions
        initial_positions = [
            (0.0, 0.0, 0.0),  # Alice at center
            (5.0, 5.0, 0.0),  # Bob nearby
            (-3.0, 7.0, 0.0),  # Charlie nearby
        ]

        for i, (handle, logic, name) in enumerate(agents):
            agent_id = handle.agent_id
            pos = initial_positions[i]

            orchestrator.world_state.entities[agent_id] = {
                "name": name,
                "type": "conversational_agent",
                "position": list(pos),
                "personality": logic.personality,
            }
            orchestrator.world_state.spatial_index[agent_id] = pos

        # Add some environmental elements
        orchestrator.world_state.entities.update(
            {
                "central_plaza": {
                    "name": "Central Plaza",
                    "type": "landmark",
                    "position": [0.0, 0.0, 0.0],
                    "description": "A gathering place for conversations",
                },
                "garden": {
                    "name": "Peaceful Garden",
                    "type": "landmark",
                    "position": [10.0, 10.0, 0.0],
                    "description": "A quiet place for reflection",
                },
            }
        )

        orchestrator.world_state.spatial_index.update(
            {
                "central_plaza": (0.0, 0.0, 0.0),
                "garden": (10.0, 10.0, 0.0),
            }
        )

        print(
            f"🌍 World initialized with {len(orchestrator.world_state.entities)} entities"
        )
        print(f"🤖 Starting {len(agents)} conversational agents...")
        print()

        # Start all agent loops
        loop_tasks = []
        for handle, logic, name in agents:
            print(f"   🗣️  {name}: {logic.personality}")
            task = asyncio.create_task(handle.run_async_loop(logic))
            loop_tasks.append(task)

        # Start environmental event simulation
        env_task = asyncio.create_task(
            simulate_message_events(orchestrator, duration=30.0)
        )

        print("\n⏰ Letting agents converse for 30 seconds...")
        print(
            "   Watch for natural conversation patterns, context building, and collaboration!"
        )
        print()

        # Let the agents run and converse
        await asyncio.sleep(30.0)

        print("\n🛑 Stopping all agents...")

        # Stop environmental events
        env_task.cancel()
        try:
            await env_task
        except asyncio.CancelledError:
            pass

        # Stop all agent loops
        for handle, _, _ in agents:
            handle.stop_async_loop()

        # Wait for all loops to finish
        await asyncio.gather(*loop_tasks, return_exceptions=True)

        print("\n📊 Final Statistics:")
        print("-" * 30)

        for handle, logic, name in agents:
            stats = logic.get_stats()
            print(f"{name}:")
            print(f"  Messages sent: {stats['messages_sent']}")
            print(f"  Moves made: {stats['moves_made']}")
            print(f"  Observations processed: {stats['observations_processed']}")
            print(f"  Final position: {stats['current_position']}")
            print(
                f"  Memory: {stats['memory_messages']} messages, {stats['known_agents']} known agents"
            )
            print()

        print("📈 Conversation Demo completed!")
        print(
            f"📈 Final world state has {len(orchestrator.world_state.entities)} entities"
        )

    finally:
        await orchestrator.shutdown()
        print("🔚 Orchestrator shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
