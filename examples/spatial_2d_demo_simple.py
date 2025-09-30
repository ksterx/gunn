#!/usr/bin/env python3
"""Simplified 2D Spatial Simulation Demo.

This is a simplified version that focuses on basic movement and observation
without complex observation timeouts or heavy concurrent operations.
"""

import asyncio
import math
import random
import uuid

from gunn import Orchestrator, OrchestratorConfig
from gunn.facades import RLFacade
from gunn.policies.observation import (
    DefaultObservationPolicy,
    PolicyConfig,
)
from gunn.schemas.types import Intent
from gunn.utils.telemetry import get_logger, setup_logging


class SimpleSpatialAgent:
    """Simple spatial agent for basic movement demo."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        initial_pos: tuple[float, float],
        facade: RLFacade,
    ):
        self.agent_id = agent_id
        self.name = name
        self.position = initial_pos
        self.facade = facade
        self.logger = get_logger(f"agent.{agent_id}")
        self.move_speed = 10.0

    async def move_to(self, target_x: float, target_y: float) -> bool:
        """Move agent to target position."""
        self.logger.info(
            f"{self.name} moving from ({self.position[0]:.1f}, {self.position[1]:.1f}) "
            f"to ({target_x:.1f}, {target_y:.1f})"
        )

        # Create move intent (simple approach without complex observation)
        intent: Intent = {
            "kind": "Move",
            "payload": {
                "from": [self.position[0], self.position[1], 0.0],
                "to": [target_x, target_y, 0.0],
                "speed": self.move_speed,
                "agent_id": self.agent_id,
            },
            "context_seq": 0,  # Use simple context sequence
            "req_id": f"move_{uuid.uuid4().hex[:8]}",
            "agent_id": self.agent_id,
            "priority": 1,
            "schema_version": "1.0.0",
        }

        try:
            # Execute move
            _effect, _observation = await self.facade.step(self.agent_id, intent)

            # Update local position
            self.position = (target_x, target_y)
            self.logger.info(f"{self.name} moved successfully")
            return True

        except Exception as e:
            self.logger.error(f"{self.name} move failed: {e}")
            return False

    async def simple_patrol(self, steps: int = 3) -> None:
        """Simple patrol around starting position."""
        start_x, start_y = self.position

        for step in range(steps):
            # Generate random nearby target
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(5, 20)

            target_x = start_x + distance * math.cos(angle)
            target_y = start_y + distance * math.sin(angle)

            success = await self.move_to(target_x, target_y)
            if success:
                self.logger.info(f"{self.name} patrol step {step + 1}/{steps} complete")
                # Brief pause
                await asyncio.sleep(0.1)
            else:
                self.logger.warning(f"{self.name} patrol step {step + 1} failed")


class SimpleSpatial2DDemo:
    """Simplified 2D spatial simulation demo."""

    def __init__(self) -> None:
        self.logger = get_logger("spatial_demo")

        # Simple configuration
        self.config = OrchestratorConfig(
            max_agents=3,
            staleness_threshold=50,  # High threshold to avoid staleness issues
            debounce_ms=100.0,
            deadline_ms=10000.0,
            token_budget=100,
            backpressure_policy="defer",
            default_priority=1,
            use_in_memory_dedup=True,
        )

        self.orchestrator = Orchestrator(self.config, world_id="simple_spatial")
        self.facade = RLFacade(orchestrator=self.orchestrator)

        # Create 3 simple agents
        self.agents = {
            "explorer": SimpleSpatialAgent(
                "explorer", "Explorer", (50.0, 50.0), self.facade
            ),
            "guard": SimpleSpatialAgent("guard", "Guard", (150.0, 50.0), self.facade),
            "scout": SimpleSpatialAgent("scout", "Scout", (100.0, 150.0), self.facade),
        }

    async def setup(self) -> None:
        """Set up the spatial simulation environment."""
        self.logger.info("Setting up simple 2D spatial simulation")

        # Initialize facade
        await self.facade.initialize()

        # Simple observation policy
        policy_config = PolicyConfig(
            distance_limit=100.0,  # Large observation range
            relationship_filter=[],
            field_visibility={"position": True, "type": True},
            max_patch_ops=10,
        )

        # Register agents with simple setup
        for agent_id, agent in self.agents.items():
            policy = DefaultObservationPolicy(policy_config)
            await self.facade.register_agent(agent_id, policy)

            # Add agent directly to world state
            self.orchestrator.world_state.entities[agent_id] = {
                "id": agent_id,
                "name": agent.name,
                "type": "agent",
                "position": [*list(agent.position), 0.0],
                "observation_range": 100.0,
                "move_speed": agent.move_speed,
            }

            # Add agent position to spatial index
            self.orchestrator.world_state.spatial_index[agent_id] = (
                *tuple(agent.position),
                0.0,
            )

        self.logger.info("Simple 2D spatial simulation setup complete")

    async def run_simple_demo(self) -> None:
        """Run a simple movement demonstration."""
        self.logger.info("Starting simple movement demo")
        print("\n📍 Starting agent movement demonstrations...\n")

        try:
            # Sequential movement demo (avoid concurrency issues)
            for agent_name, agent in self.agents.items():
                print(f"=== {agent_name.upper()} PATROL ===")
                self.logger.info(f"=== {agent_name.upper()} DEMO ===")
                await agent.simple_patrol(steps=2)
                self.logger.info(f"{agent_name} demo complete")
                print(f"✓ {agent_name} patrol complete\n")
                await asyncio.sleep(0.2)  # Brief pause between agents

            # Convergence test - agents move near center with spacing
            print("=== CONVERGENCE TEST ===")
            self.logger.info("=== CONVERGENCE TEST ===")
            center = (100.0, 100.0)
            spacing = 5.0  # Maintain minimum spacing between agents

            for i, agent in enumerate(self.agents.values()):
                # Calculate position around center with spacing
                angle = (i * 2 * math.pi) / len(self.agents)
                target_x = center[0] + spacing * math.cos(angle)
                target_y = center[1] + spacing * math.sin(angle)

                success = await agent.move_to(target_x, target_y)
                if success:
                    print(f"✓ {agent.name} reached convergence position")
                    self.logger.info(
                        f"{agent.name} reached position near convergence point"
                    )
                else:
                    print(f"✗ {agent.name} failed to reach convergence position")
                    self.logger.warning(
                        f"{agent.name} failed to reach convergence point"
                    )

            print("\n📊 All movements completed!\n")

        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            raise

    async def shutdown(self) -> None:
        """Clean up demo resources."""
        self.logger.info("Shutting down simple spatial demo")
        await self.facade.shutdown()


async def main() -> None:
    """Run the simplified 2D spatial simulation demo."""
    # Set up logging
    setup_logging("INFO")

    print("🗺️  Simple 2D Spatial Simulation Demo")
    print("=" * 40)
    print()

    demo = SimpleSpatial2DDemo()

    try:
        await demo.setup()
        await demo.run_simple_demo()
        print()
        print("✅ Demo completed successfully!")
        print("- Agent movement with spatial coordinates")
        print("- Basic world state management")
        print("- Simple observation system")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise
    finally:
        await demo.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
