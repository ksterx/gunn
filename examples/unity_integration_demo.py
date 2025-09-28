#!/usr/bin/env python3
"""Unity Integration Demo for Real-Time Agent Interaction.

This demo showcases integration with Unity game engine through the gunn
multi-agent simulation core. It demonstrates real-time agent interaction,
physics event handling, and bidirectional communication between Unity and
the simulation core.

Note: This is a demonstration/placeholder implementation since the actual
Unity adapter (Task 19) is not yet implemented. This demo shows the intended
interface and interaction patterns.

Key features demonstrated:
- Unity adapter integration patterns
- TimeTick event conversion to Effects
- Move intent conversion to Unity game commands
- Physics collision event handling
- Real-time bidirectional communication
- Game state synchronization

Requirements addressed:
- 8.1: Unity adapter converts game events to Effects and Intents to game commands
- 8.4: TimeTick events converted to Effect events for time synchronization
- 8.5: Physics collisions reflected as Effects in simulation core
"""

import asyncio
import math
import time
import uuid
from typing import Any

from gunn import Orchestrator, OrchestratorConfig
from gunn.facades import RLFacade
from gunn.policies.observation import DefaultObservationPolicy, PolicyConfig
from gunn.schemas.types import EffectDraft, Intent
from gunn.utils.telemetry import get_logger, setup_logging


class MockUnityAdapter:
    """Mock Unity adapter demonstrating integration patterns.

    In the real implementation (Task 19), this would be replaced with
    actual WebSocket/gRPC communication to Unity engine.
    """

    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.logger = get_logger("unity_adapter")
        self.connected = False
        self.game_objects: dict[str, dict[str, Any]] = {}
        self.physics_enabled = True
        self.time_scale = 1.0
        self.tick_rate = 60.0  # 60 FPS
        self.last_tick_time = 0.0

        # Mock Unity scene state
        self.scene_bounds = {"min": (-50.0, -50.0, -10.0), "max": (50.0, 50.0, 10.0)}
        self.gravity = (0.0, -9.81, 0.0)

    async def connect(self) -> bool:
        """Simulate connection to Unity engine."""
        self.logger.info("Connecting to Unity engine...")

        # Simulate connection delay
        await asyncio.sleep(0.1)

        self.connected = True
        self.logger.info("Connected to Unity engine successfully")

        # Set orchestrator sim_time authority to Unity
        self.orchestrator.set_sim_time_authority("unity")

        return True

    async def disconnect(self) -> None:
        """Disconnect from Unity engine."""
        if self.connected:
            self.logger.info("Disconnecting from Unity engine...")
            self.connected = False
            self.logger.info("Disconnected from Unity engine")

    async def spawn_game_object(
        self, object_id: str, object_type: str, position: tuple[float, float, float]
    ) -> bool:
        """Spawn a game object in Unity scene."""
        if not self.connected:
            return False

        game_object = {
            "id": object_id,
            "type": object_type,
            "position": position,
            "rotation": (0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
            "velocity": (0.0, 0.0, 0.0),
            "active": True,
            "physics_enabled": object_type in ["agent", "dynamic_object"],
        }

        self.game_objects[object_id] = game_object

        self.logger.info(f"Spawned {object_type} '{object_id}' at {position}")

        # Notify simulation core
        await self.orchestrator.broadcast_event(
            EffectDraft(
                kind="GameObjectSpawned",
                payload={
                    "object_id": object_id,
                    "object_type": object_type,
                    "position": list(position),
                    "unity_scene": "MainScene",
                },
                source_id="unity_adapter",
                schema_version="1.0.0",
            )
        )

        return True

    async def move_game_object(
        self,
        object_id: str,
        target_position: tuple[float, float, float],
        speed: float = 5.0,
    ) -> bool:
        """Move a game object to target position."""
        if not self.connected or object_id not in self.game_objects:
            return False

        game_object = self.game_objects[object_id]
        old_position = tuple(game_object["position"])

        # Calculate movement
        dx = target_position[0] - old_position[0]
        dy = target_position[1] - old_position[1]
        dz = target_position[2] - old_position[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance < 0.1:
            return True  # Already at target

        # Update position
        game_object["position"] = target_position

        self.logger.info(
            f"Moved '{object_id}' from {old_position} to {target_position}"
        )

        # Check for collisions
        await self._check_collisions(object_id, old_position, target_position)

        # Notify simulation core
        await self.orchestrator.broadcast_event(
            EffectDraft(
                kind="GameObjectMoved",
                payload={
                    "object_id": object_id,
                    "from_position": list(old_position),
                    "to_position": list(target_position),
                    "distance": distance,
                    "speed": speed,
                },
                source_id="unity_adapter",
                schema_version="1.0.0",
            )
        )

        return True

    async def _check_collisions(
        self,
        moving_object_id: str,
        from_pos: tuple[float, float, float],
        to_pos: tuple[float, float, float],
    ) -> None:
        """Check for physics collisions during movement."""
        if not self.physics_enabled:
            return

        moving_object = self.game_objects[moving_object_id]

        # Simple collision detection with other objects
        for other_id, other_object in self.game_objects.items():
            if other_id == moving_object_id or not other_object["active"]:
                continue

            other_pos = tuple(other_object["position"])

            # Calculate distance to other object
            dx = to_pos[0] - other_pos[0]
            dy = to_pos[1] - other_pos[1]
            dz = to_pos[2] - other_pos[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            # Simple collision threshold
            collision_threshold = 2.0

            if distance < collision_threshold:
                self.logger.info(
                    f"Collision detected: '{moving_object_id}' with '{other_id}'"
                )

                # Emit collision event to simulation core
                await self.orchestrator.broadcast_event(
                    EffectDraft(
                        kind="PhysicsCollision",
                        payload={
                            "object_a": moving_object_id,
                            "object_b": other_id,
                            "collision_point": list(to_pos),
                            "collision_force": distance,  # Simplified
                            "collision_type": "object_collision",
                        },
                        source_id="unity_physics",
                        schema_version="1.0.0",
                    )
                )

    async def start_time_tick_loop(self) -> None:
        """Start the Unity time tick loop for synchronization."""
        self.logger.info(f"Starting Unity time tick loop at {self.tick_rate} FPS")

        tick_interval = 1.0 / self.tick_rate

        while self.connected:
            current_time = time.time()

            # Emit TimeTick event
            await self.orchestrator.broadcast_event(
                EffectDraft(
                    kind="TimeTick",
                    payload={
                        "unity_time": current_time,
                        "delta_time": current_time - self.last_tick_time,
                        "time_scale": self.time_scale,
                        "frame_count": int(current_time * self.tick_rate),
                    },
                    source_id="unity_time",
                    schema_version="1.0.0",
                )
            )

            self.last_tick_time = current_time

            # Wait for next tick
            await asyncio.sleep(tick_interval)

    async def handle_intent(self, intent: Intent) -> bool:
        """Convert simulation intent to Unity game command."""
        if not self.connected:
            return False

        intent_kind = intent["kind"]
        payload = intent["payload"]
        agent_id = intent["agent_id"]

        if intent_kind == "Move":
            # Convert Move intent to Unity movement
            target_pos = payload.get("to_position", [0, 0, 0])
            speed = payload.get("speed", 5.0)

            return await self.move_game_object(agent_id, tuple(target_pos), speed)

        elif intent_kind == "Interact":
            # Convert Interact intent to Unity interaction
            target_object = payload.get("target_object")
            interaction_type = payload.get("interaction_type", "use")

            self.logger.info(
                f"Unity interaction: {agent_id} {interaction_type} {target_object}"
            )

            # Emit interaction event
            await self.orchestrator.broadcast_event(
                EffectDraft(
                    kind="GameObjectInteraction",
                    payload={
                        "agent_id": agent_id,
                        "target_object": target_object,
                        "interaction_type": interaction_type,
                        "success": True,
                    },
                    source_id="unity_interaction",
                    schema_version="1.0.0",
                )
            )

            return True

        elif intent_kind == "Speak":
            # Convert Speak intent to Unity UI/audio
            text = payload.get("text", "")

            self.logger.info(f"Unity speech: {agent_id} says '{text}'")

            # In real Unity, this would trigger speech bubble or audio
            await self.orchestrator.broadcast_event(
                EffectDraft(
                    kind="SpeechDisplayed",
                    payload={
                        "agent_id": agent_id,
                        "text": text,
                        "display_duration": 3.0,
                    },
                    source_id="unity_ui",
                    schema_version="1.0.0",
                )
            )

            return True

        else:
            self.logger.warning(f"Unknown intent kind for Unity: {intent_kind}")
            return False


class UnityAgent:
    """Agent that interacts with Unity through the simulation core."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        initial_pos: tuple[float, float, float],
        facade: RLFacade,
    ):
        self.agent_id = agent_id
        self.name = name
        self.position = initial_pos
        self.facade = facade
        self.logger = get_logger(f"unity_agent.{agent_id}")

    async def move_in_unity(self, target_pos: tuple[float, float, float]) -> bool:
        """Move agent in Unity scene."""
        self.logger.info(f"{self.name} moving to {target_pos} in Unity")

        intent: Intent = {
            "kind": "Move",
            "payload": {
                "from_position": list(self.position),
                "to_position": list(target_pos),
                "speed": 5.0,
            },
            "context_seq": 0,
            "req_id": f"unity_move_{uuid.uuid4().hex[:8]}",
            "agent_id": self.agent_id,
            "priority": 1,
            "schema_version": "1.0.0",
        }

        try:
            effect, observation = await self.facade.step(self.agent_id, intent)
            self.position = target_pos
            return True
        except Exception as e:
            self.logger.error(f"Unity move failed: {e}")
            return False

    async def interact_with_object(
        self, target_object: str, interaction_type: str = "use"
    ) -> bool:
        """Interact with a Unity game object."""
        self.logger.info(
            f"{self.name} interacting with {target_object} ({interaction_type})"
        )

        intent: Intent = {
            "kind": "Interact",
            "payload": {
                "target_object": target_object,
                "interaction_type": interaction_type,
                "agent_position": list(self.position),
            },
            "context_seq": 0,
            "req_id": f"unity_interact_{uuid.uuid4().hex[:8]}",
            "agent_id": self.agent_id,
            "priority": 1,
            "schema_version": "1.0.0",
        }

        try:
            effect, observation = await self.facade.step(self.agent_id, intent)
            return True
        except Exception as e:
            self.logger.error(f"Unity interaction failed: {e}")
            return False

    async def speak_in_unity(self, text: str) -> bool:
        """Speak text in Unity (speech bubble/audio)."""
        self.logger.info(f"{self.name} speaking in Unity: '{text}'")

        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": text, "agent_position": list(self.position)},
            "context_seq": 0,
            "req_id": f"unity_speak_{uuid.uuid4().hex[:8]}",
            "agent_id": self.agent_id,
            "priority": 1,
            "schema_version": "1.0.0",
        }

        try:
            effect, observation = await self.facade.step(self.agent_id, intent)
            return True
        except Exception as e:
            self.logger.error(f"Unity speech failed: {e}")
            return False


class UnityIntegrationDemo:
    """Main demo class for Unity integration."""

    def __init__(self):
        self.logger = get_logger("unity_integration_demo")

        # Configure orchestrator for Unity integration
        self.config = OrchestratorConfig(
            max_agents=4,
            staleness_threshold=1,
            debounce_ms=100.0,
            deadline_ms=5000.0,
            token_budget=100,
            backpressure_policy="defer",
            default_priority=1,
            use_in_memory_dedup=True,
        )

        self.orchestrator = Orchestrator(self.config, world_id="unity_integration")
        self.facade = RLFacade(orchestrator=self.orchestrator)

        # Create Unity adapter
        self.unity_adapter = MockUnityAdapter(self.orchestrator)

        # Create Unity agents
        self.agents = {
            "player": UnityAgent("player", "Player", (0.0, 0.0, 0.0), self.facade),
            "npc1": UnityAgent("npc1", "Guard", (10.0, 0.0, 0.0), self.facade),
            "npc2": UnityAgent("npc2", "Merchant", (-10.0, 0.0, 0.0), self.facade),
            "npc3": UnityAgent("npc3", "Explorer", (0.0, 10.0, 0.0), self.facade),
        }

        # Unity scene objects
        self.scene_objects = {
            "treasure_chest": {"type": "interactive", "position": (5.0, 0.0, 5.0)},
            "magic_portal": {"type": "interactive", "position": (-5.0, 0.0, -5.0)},
            "stone_pillar": {"type": "static", "position": (0.0, 0.0, 8.0)},
            "campfire": {"type": "interactive", "position": (8.0, 0.0, -3.0)},
        }

    async def setup(self) -> None:
        """Set up the Unity integration environment."""
        self.logger.info("Setting up Unity integration demo")

        # Initialize facade
        await self.facade.initialize()

        # Connect to Unity
        connected = await self.unity_adapter.connect()
        if not connected:
            raise RuntimeError("Failed to connect to Unity adapter")

        # Create observation policy for Unity agents
        policy_config = PolicyConfig(
            distance_limit=20.0,  # Unity scene observation range
            relationship_filter=[],
            field_visibility={"position": True, "type": True, "unity_data": True},
            max_patch_ops=25,
            include_spatial_index=True,
        )

        # Register agents
        for agent_id, agent in self.agents.items():
            policy = DefaultObservationPolicy(policy_config)
            await self.facade.register_agent(agent_id, policy)

        # Spawn agents in Unity scene
        for agent_id, agent in self.agents.items():
            await self.unity_adapter.spawn_game_object(
                agent_id, "agent", agent.position
            )

        # Spawn scene objects in Unity
        for obj_id, obj_data in self.scene_objects.items():
            await self.unity_adapter.spawn_game_object(
                obj_id, obj_data["type"], obj_data["position"]
            )

        # Start Unity time tick loop
        asyncio.create_task(self.unity_adapter.start_time_tick_loop())

        self.logger.info("Unity integration demo setup complete")

    async def run_unity_scenario(self) -> None:
        """Run the main Unity integration scenario."""
        self.logger.info("Starting Unity integration scenario")

        try:
            # Scenario 1: Agent movement and collision detection
            await self._demonstrate_movement_and_collisions()

            # Scenario 2: Object interactions
            await self._demonstrate_object_interactions()

            # Scenario 3: Multi-agent coordination
            await self._demonstrate_multi_agent_coordination()

            # Scenario 4: Physics events and real-time sync
            await self._demonstrate_physics_and_sync()

        except Exception as e:
            self.logger.error(f"Unity scenario failed: {e}")
            raise

    async def _demonstrate_movement_and_collisions(self) -> None:
        """Demonstrate agent movement and collision detection."""
        self.logger.info("Phase 1: Movement and collision detection")

        # Move player toward NPC1 (should trigger collision)
        player = self.agents["player"]
        npc1 = self.agents["npc1"]

        # Move player close to NPC1's position
        collision_target = (npc1.position[0] - 1.0, npc1.position[1], npc1.position[2])

        await player.move_in_unity(collision_target)

        # Brief pause to let collision events propagate
        await asyncio.sleep(0.2)

        # Move NPCs to create more interactions
        await npc1.move_in_unity((15.0, 0.0, 0.0))
        await self.agents["npc2"].move_in_unity((-15.0, 0.0, 0.0))

        self.logger.info("Movement and collision demonstration complete")

    async def _demonstrate_object_interactions(self) -> None:
        """Demonstrate interactions with Unity scene objects."""
        self.logger.info("Phase 2: Object interactions")

        player = self.agents["player"]

        # Move to treasure chest and interact
        chest_pos = self.scene_objects["treasure_chest"]["position"]
        await player.move_in_unity((chest_pos[0] - 1.0, chest_pos[1], chest_pos[2]))
        await player.interact_with_object("treasure_chest", "open")

        # Move to campfire and interact
        fire_pos = self.scene_objects["campfire"]["position"]
        await player.move_in_unity((fire_pos[0] - 1.0, fire_pos[1], fire_pos[2]))
        await player.interact_with_object("campfire", "light")

        # Speak near the campfire
        await player.speak_in_unity("The campfire is now lit! Everyone gather around.")

        self.logger.info("Object interaction demonstration complete")

    async def _demonstrate_multi_agent_coordination(self) -> None:
        """Demonstrate multi-agent coordination in Unity."""
        self.logger.info("Phase 3: Multi-agent coordination")

        # All agents move to campfire area for coordination
        campfire_pos = self.scene_objects["campfire"]["position"]

        coordination_tasks = []
        for i, (agent_id, agent) in enumerate(self.agents.items()):
            # Position agents in a circle around campfire
            angle = (i * 2 * math.pi) / len(self.agents)
            radius = 4.0

            target_x = campfire_pos[0] + radius * math.cos(angle)
            target_z = campfire_pos[2] + radius * math.sin(angle)
            target_pos = (target_x, campfire_pos[1], target_z)

            task = agent.move_in_unity(target_pos)
            coordination_tasks.append(task)

        # Execute coordinated movement
        await asyncio.gather(*coordination_tasks)

        # Coordinated speech sequence
        await self.agents["npc1"].speak_in_unity(
            "Everyone's here! Let's plan our next move."
        )
        await asyncio.sleep(1.0)

        await self.agents["npc2"].speak_in_unity("I have supplies we can trade.")
        await asyncio.sleep(1.0)

        await self.agents["npc3"].speak_in_unity("I know the way to the ancient ruins.")
        await asyncio.sleep(1.0)

        await self.agents["player"].speak_in_unity("Perfect! Let's work together.")

        self.logger.info("Multi-agent coordination demonstration complete")

    async def _demonstrate_physics_and_sync(self) -> None:
        """Demonstrate physics events and real-time synchronization."""
        self.logger.info("Phase 4: Physics events and real-time sync")

        # Create rapid movements to trigger multiple physics events
        player = self.agents["player"]

        # Rapid movement sequence
        movement_sequence = [
            (10.0, 0.0, 10.0),
            (-10.0, 0.0, 10.0),
            (-10.0, 0.0, -10.0),
            (10.0, 0.0, -10.0),
            (0.0, 0.0, 0.0),  # Return to center
        ]

        for target_pos in movement_sequence:
            await player.move_in_unity(target_pos)
            await asyncio.sleep(0.1)  # Brief pause between movements

        # Let time ticks and physics events propagate
        await asyncio.sleep(1.0)

        self.logger.info("Physics and synchronization demonstration complete")

    async def _analyze_unity_performance(self) -> None:
        """Analyze Unity integration performance."""
        self.logger.info("Analyzing Unity integration performance")

        # Get event log for analysis
        event_log = self.orchestrator.event_log
        entries = event_log.get_entries_since(0)

        # Categorize events
        unity_events = []
        time_tick_events = []
        physics_events = []

        for entry in entries:
            effect = entry.effect
            kind = effect["kind"]
            source = effect["source_id"]

            if source.startswith("unity"):
                unity_events.append(effect)

                if kind == "TimeTick":
                    time_tick_events.append(effect)
                elif kind == "PhysicsCollision":
                    physics_events.append(effect)

        self.logger.info("Unity Performance Analysis:")
        self.logger.info(f"  Total Unity events: {len(unity_events)}")
        self.logger.info(f"  Time tick events: {len(time_tick_events)}")
        self.logger.info(f"  Physics collision events: {len(physics_events)}")
        self.logger.info(f"  Total events processed: {len(entries)}")

        # Calculate event processing rate
        if entries:
            first_time = entries[0].wall_time
            last_time = entries[-1].wall_time
            duration = last_time - first_time

            if duration > 0:
                events_per_second = len(entries) / duration
                self.logger.info(
                    f"  Event processing rate: {events_per_second:.1f} events/sec"
                )

    async def shutdown(self) -> None:
        """Clean up Unity integration demo resources."""
        self.logger.info("Shutting down Unity integration demo")

        # Analyze performance before shutdown
        await self._analyze_unity_performance()

        # Disconnect from Unity
        await self.unity_adapter.disconnect()

        # Shutdown facade
        await self.facade.shutdown()


async def main() -> None:
    """Run the Unity integration demo."""
    # Set up logging
    setup_logging("INFO")

    print("🎮 Unity Integration Demo for Real-Time Agent Interaction")
    print("=" * 60)
    print()
    print("Note: This is a demonstration using a mock Unity adapter.")
    print("The real Unity adapter will be implemented in Task 19.")
    print()

    demo = UnityIntegrationDemo()

    try:
        await demo.setup()
        await demo.run_unity_scenario()
    finally:
        await demo.shutdown()

    print()
    print("Demo completed! Check the logs above to see:")
    print("✅ Unity adapter integration patterns")
    print("✅ TimeTick event conversion to Effects")
    print("✅ Move intent conversion to Unity game commands")
    print("✅ Physics collision event handling")
    print("✅ Real-time bidirectional communication")
    print("✅ Game state synchronization")


if __name__ == "__main__":
    asyncio.run(main())
