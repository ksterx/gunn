#!/usr/bin/env python3
"""Demonstration of asynchronous agent loop infrastructure.

This example shows how to use the AsyncAgentLogic interface and AgentHandle
async loop methods to create agents that operate independently in an
observe-think-act pattern.
"""

import asyncio
import uuid

from gunn import AsyncAgentLogic, Orchestrator, OrchestratorConfig
from gunn.policies.observation import ObservationPolicy, PolicyConfig
from gunn.schemas.messages import View, WorldState
from gunn.schemas.types import Effect, Intent


class SimpleObservationPolicy(ObservationPolicy):
    """Simple observation policy that shows all entities to all agents."""

    def __init__(self):
        config = PolicyConfig(distance_limit=50.0)
        super().__init__(config)

    def filter_world_state(self, world_state: WorldState, agent_id: str) -> View:
        """Return a view with all entities visible."""
        return View(
            agent_id=agent_id,
            view_seq=0,
            visible_entities=world_state.entities,
            visible_relationships=world_state.relationships,
            context_digest=f"view_{agent_id}",
        )

    def should_observe_event(
        self, effect: Effect, agent_id: str, world_state: WorldState
    ) -> bool:
        """All agents observe all events."""
        return True


class ChatterAgent(AsyncAgentLogic):
    """Agent that occasionally sends chat messages."""

    def __init__(self, name: str, chat_probability: float = 0.3):
        self.name = name
        self.chat_probability = chat_probability
        self.message_count = 0
        self.observations_seen = 0

    async def process_observation(
        self, observation: View, agent_id: str
    ) -> Intent | None:
        """Process observation and occasionally generate a chat message."""
        import random

        self.observations_seen += 1

        # Look for other agents in the observation
        other_agents = [
            entity_id
            for entity_id in observation.visible_entities.keys()
            if entity_id != agent_id and entity_id.startswith("agent_")
        ]

        # Occasionally send a message
        if random.random() < self.chat_probability:
            self.message_count += 1

            if other_agents:
                message = f"Hello from {self.name}! I see {len(other_agents)} other agents. (Message #{self.message_count})"
            else:
                message = f"{self.name} here, looking around... (Message #{self.message_count})"

            return {
                "kind": "Speak",
                "payload": {"text": message, "agent_id": agent_id},
                "context_seq": observation.view_seq,
                "req_id": str(uuid.uuid4()),
                "agent_id": agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }

        return None  # No action this time

    async def on_loop_start(self, agent_id: str) -> None:
        """Called when agent loop starts."""
        print(f"🤖 {self.name} ({agent_id}) started its async loop")

    async def on_loop_stop(self, agent_id: str) -> None:
        """Called when agent loop stops."""
        print(f"🛑 {self.name} ({agent_id}) stopped its async loop")
        print(
            f"   Final stats: {self.message_count} messages sent, {self.observations_seen} observations processed"
        )

    async def on_error(self, agent_id: str, error: Exception) -> bool:
        """Handle errors during processing."""
        print(f"❌ {self.name} ({agent_id}) encountered error: {error}")
        return True  # Continue on error


class ExplorerAgent(AsyncAgentLogic):
    """Agent that moves around and explores."""

    def __init__(self, name: str, move_probability: float = 0.2):
        self.name = name
        self.move_probability = move_probability
        self.position = [0.0, 0.0, 0.0]
        self.move_count = 0
        self.observations_seen = 0

    async def process_observation(
        self, observation: View, agent_id: str
    ) -> Intent | None:
        """Process observation and occasionally move to a new location."""
        import random

        self.observations_seen += 1

        # Update our position from the observation if available
        if agent_id in observation.visible_entities:
            entity_data = observation.visible_entities[agent_id]
            if "position" in entity_data:
                self.position = entity_data["position"]

        # Occasionally move to a new location
        if random.random() < self.move_probability:
            self.move_count += 1

            # Generate a new random position within a reasonable range
            new_x = self.position[0] + random.uniform(-10, 10)
            new_y = self.position[1] + random.uniform(-10, 10)
            new_z = 0.0  # Stay on ground level

            return {
                "kind": "Move",
                "payload": {
                    "from": self.position,
                    "to": [new_x, new_y, new_z],
                    "agent_id": agent_id,
                },
                "context_seq": observation.view_seq,
                "req_id": str(uuid.uuid4()),
                "agent_id": agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }

        return None  # No action this time

    async def on_loop_start(self, agent_id: str) -> None:
        """Called when agent loop starts."""
        print(f"🚶 {self.name} ({agent_id}) started exploring")

    async def on_loop_stop(self, agent_id: str) -> None:
        """Called when agent loop stops."""
        print(f"🏁 {self.name} ({agent_id}) stopped exploring")
        print(
            f"   Final stats: {self.move_count} moves made, {self.observations_seen} observations processed"
        )
        print(f"   Final position: {self.position}")

    async def on_error(self, agent_id: str, error: Exception) -> bool:
        """Handle errors during processing."""
        print(f"❌ {self.name} ({agent_id}) encountered error: {error}")
        return True  # Continue on error


async def main():
    """Run the async agent demo."""
    print("🎬 Starting Async Agent Loop Demo")
    print("=" * 50)

    # Create orchestrator
    config = OrchestratorConfig(
        use_in_memory_dedup=True, max_agents=10, staleness_threshold=0
    )
    orchestrator = Orchestrator(config, world_id="async_demo")
    await orchestrator.initialize()

    try:
        # Create observation policy
        policy = SimpleObservationPolicy()

        # Register agents with different behaviors
        agents = []

        # Create chatter agents
        for i in range(2):
            agent_id = f"agent_chatter_{i}"
            handle = await orchestrator.register_agent(agent_id, policy)
            logic = ChatterAgent(f"Chatter-{i}", chat_probability=0.4)
            agents.append((handle, logic))

        # Create explorer agents
        for i in range(2):
            agent_id = f"agent_explorer_{i}"
            handle = await orchestrator.register_agent(agent_id, policy)
            logic = ExplorerAgent(f"Explorer-{i}", move_probability=0.3)
            agents.append((handle, logic))

        # Add some initial entities to the world
        orchestrator.world_state.entities.update(
            {
                "agent_chatter_0": {"name": "Chatter-0", "type": "agent"},
                "agent_chatter_1": {"name": "Chatter-1", "type": "agent"},
                "agent_explorer_0": {
                    "name": "Explorer-0",
                    "type": "agent",
                    "position": [5.0, 5.0, 0.0],
                },
                "agent_explorer_1": {
                    "name": "Explorer-1",
                    "type": "agent",
                    "position": [-5.0, -5.0, 0.0],
                },
                "landmark_1": {
                    "name": "Central Plaza",
                    "type": "landmark",
                    "position": [0.0, 0.0, 0.0],
                },
                "landmark_2": {
                    "name": "North Tower",
                    "type": "landmark",
                    "position": [0.0, 20.0, 0.0],
                },
            }
        )

        orchestrator.world_state.spatial_index.update(
            {
                "agent_explorer_0": (5.0, 5.0, 0.0),
                "agent_explorer_1": (-5.0, -5.0, 0.0),
                "landmark_1": (0.0, 0.0, 0.0),
                "landmark_2": (0.0, 20.0, 0.0),
            }
        )

        print(
            f"🌍 World initialized with {len(orchestrator.world_state.entities)} entities"
        )
        print(f"🤖 Starting {len(agents)} agents with async loops...")
        print()

        # Start all agent loops
        loop_tasks = []
        for handle, logic in agents:
            task = asyncio.create_task(handle.run_async_loop(logic))
            loop_tasks.append(task)

        # Let the agents run for a while
        print("⏰ Letting agents run for 10 seconds...")
        await asyncio.sleep(10.0)

        print("\n🛑 Stopping all agents...")

        # Stop all agent loops
        for handle, _ in agents:
            handle.stop_async_loop()

        # Wait for all loops to finish
        await asyncio.gather(*loop_tasks, return_exceptions=True)

        print("\n📊 Demo completed!")
        print(
            f"📈 Final world state has {len(orchestrator.world_state.entities)} entities"
        )

    finally:
        await orchestrator.shutdown()
        print("🔚 Orchestrator shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
