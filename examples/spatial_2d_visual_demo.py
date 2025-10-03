#!/usr/bin/env python3
"""Visual 2D Spatial Simulation Demo with real-time animation.

This demo provides visual feedback of agent movements using matplotlib,
allowing you to see the simulation in action.
"""

import asyncio
import math
import random
import uuid
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from gunn import Orchestrator, OrchestratorConfig
from gunn.facades import RLFacade
from gunn.policies.observation import (
    DefaultObservationPolicy,
    PolicyConfig,
)
from gunn.schemas.types import Intent
from gunn.utils.telemetry import get_logger, setup_logging


class VisualSpatialAgent:
    """Spatial agent with position tracking for visualization."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        initial_pos: tuple[float, float],
        color: str,
        facade: RLFacade,
    ):
        self.agent_id = agent_id
        self.name = name
        self.position = initial_pos
        self.color = color
        self.facade = facade
        self.logger = get_logger(f"agent.{agent_id}")
        self.move_speed = 15.0
        self.trajectory: deque[tuple[float, float]] = deque(
            maxlen=20
        )  # Track last 20 positions
        self.trajectory.append(initial_pos)

    async def move_to(self, target_x: float, target_y: float) -> bool:
        """Move agent to target position."""
        intent: Intent = {
            "kind": "Move",
            "payload": {
                "from": [self.position[0], self.position[1], 0.0],
                "to": [target_x, target_y, 0.0],
                "speed": self.move_speed,
                "agent_id": self.agent_id,
            },
            "context_seq": 0,
            "req_id": f"move_{uuid.uuid4().hex[:8]}",
            "agent_id": self.agent_id,
            "priority": 1,
            "schema_version": "1.0.0",
        }

        try:
            await self.facade.step(self.agent_id, intent)
            self.position = (target_x, target_y)
            self.trajectory.append(self.position)
            return True
        except Exception as e:
            self.logger.error(f"{self.name} move failed: {e}")
            return False


class VisualSpatial2DDemo:
    """Visual 2D spatial simulation with real-time display."""

    def __init__(self) -> None:
        self.logger = get_logger("visual_demo")

        # Simulation configuration
        self.config = OrchestratorConfig(
            max_agents=5,
            staleness_threshold=100,
            debounce_ms=100.0,
            deadline_ms=10000.0,
            token_budget=100,
            backpressure_policy="defer",
            default_priority=1,
            use_in_memory_dedup=True,
        )

        self.orchestrator = Orchestrator(self.config, world_id="visual_spatial")
        self.facade = RLFacade(orchestrator=self.orchestrator)

        # Create agents with distinct colors
        self.agents = {
            "explorer": VisualSpatialAgent(
                "explorer", "Explorer", (50.0, 50.0), "#FF6B6B", self.facade
            ),
            "guard": VisualSpatialAgent(
                "guard", "Guard", (150.0, 50.0), "#4ECDC4", self.facade
            ),
            "scout": VisualSpatialAgent(
                "scout", "Scout", (100.0, 150.0), "#FFE66D", self.facade
            ),
            "wanderer": VisualSpatialAgent(
                "wanderer", "Wanderer", (50.0, 150.0), "#95E1D3", self.facade
            ),
        }

        # Visualization state
        self.running = True
        self.current_phase = "Initializing"
        self.phase_counter = 0

        # World boundaries
        self.world_size = 200.0

    async def setup(self) -> None:
        """Set up the spatial simulation environment."""
        self.logger.info("Setting up visual 2D spatial simulation")

        await self.facade.initialize()

        policy_config = PolicyConfig(
            distance_limit=100.0,
            relationship_filter=[],
            field_visibility={"position": True, "type": True},
            max_patch_ops=10,
        )

        for agent_id, agent in self.agents.items():
            policy = DefaultObservationPolicy(policy_config)
            await self.facade.register_agent(agent_id, policy)

            # Register in world state
            self.orchestrator.world_state.entities[agent_id] = {
                "id": agent_id,
                "name": agent.name,
                "type": "agent",
                "position": [*list(agent.position), 0.0],
                "observation_range": 100.0,
                "move_speed": agent.move_speed,
            }

            self.orchestrator.world_state.spatial_index[agent_id] = (
                *tuple(agent.position),
                0.0,
            )

        self.logger.info("Visual 2D spatial simulation setup complete")

    def init_plot(self) -> tuple:
        """Initialize the matplotlib plot."""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0, self.world_size)
        self.ax.set_ylim(0, self.world_size)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Gunn Framework - 2D Spatial Simulation", fontsize=14, pad=20)
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")

        # Create artist objects for each agent
        self.agent_circles = {}
        self.agent_trails = {}
        self.agent_labels = {}

        for agent_id, agent in self.agents.items():
            # Main circle
            circle = Circle(
                agent.position, 3.0, color=agent.color, alpha=0.8, zorder=10
            )
            self.ax.add_patch(circle)
            self.agent_circles[agent_id] = circle

            # Trail line
            (trail,) = self.ax.plot([], [], color=agent.color, alpha=0.3, linewidth=1)
            self.agent_trails[agent_id] = trail

            # Label
            label = self.ax.text(
                agent.position[0],
                agent.position[1] + 5,
                agent.name,
                ha="center",
                fontsize=9,
                color=agent.color,
                weight="bold",
            )
            self.agent_labels[agent_id] = label

        # Phase text
        self.phase_text = self.ax.text(
            self.world_size / 2,
            self.world_size - 10,
            self.current_phase,
            ha="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8},
        )

        return (
            list(self.agent_circles.values())
            + list(self.agent_trails.values())
            + list(self.agent_labels.values())
            + [self.phase_text]
        )

    def update_plot(self, frame: int) -> tuple:
        """Update the plot for animation."""
        # Update agent positions and trails
        for agent_id, agent in self.agents.items():
            circle = self.agent_circles[agent_id]
            circle.center = agent.position

            # Update trail
            if len(agent.trajectory) > 1:
                xs, ys = zip(*agent.trajectory, strict=False)
                self.agent_trails[agent_id].set_data(xs, ys)

            # Update label position
            self.agent_labels[agent_id].set_position(
                (agent.position[0], agent.position[1] + 5)
            )

        # Update phase text
        self.phase_text.set_text(self.current_phase)

        return (
            list(self.agent_circles.values())
            + list(self.agent_trails.values())
            + list(self.agent_labels.values())
            + [self.phase_text]
        )

    async def run_movement_patterns(self) -> None:
        """Run various movement patterns for demonstration."""
        try:
            # Phase 1: Random patrol
            self.current_phase = "Phase 1: Random Patrol"
            print(f"\n{self.current_phase}")
            for _ in range(8):
                for agent in self.agents.values():
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(10, 25)
                    target_x = max(
                        10,
                        min(
                            self.world_size - 10,
                            agent.position[0] + distance * math.cos(angle),
                        ),
                    )
                    target_y = max(
                        10,
                        min(
                            self.world_size - 10,
                            agent.position[1] + distance * math.sin(angle),
                        ),
                    )
                    await agent.move_to(target_x, target_y)
                await asyncio.sleep(0.3)

            # Phase 2: Circular formation
            self.current_phase = "Phase 2: Circular Formation"
            print(f"\n{self.current_phase}")
            center = (self.world_size / 2, self.world_size / 2)
            radius = 40.0

            for i, agent in enumerate(self.agents.values()):
                angle = (i * 2 * math.pi) / len(self.agents)
                target_x = center[0] + radius * math.cos(angle)
                target_y = center[1] + radius * math.sin(angle)
                await agent.move_to(target_x, target_y)
                await asyncio.sleep(0.4)

            await asyncio.sleep(1.0)

            # Phase 3: Rotation
            self.current_phase = "Phase 3: Synchronized Rotation"
            print(f"\n{self.current_phase}")
            for rotation in range(8):
                for i, agent in enumerate(self.agents.values()):
                    base_angle = (i * 2 * math.pi) / len(self.agents)
                    angle = base_angle + (rotation * math.pi / 4)
                    target_x = center[0] + radius * math.cos(angle)
                    target_y = center[1] + radius * math.sin(angle)
                    await agent.move_to(target_x, target_y)
                await asyncio.sleep(0.4)

            # Phase 4: Expand and contract
            self.current_phase = "Phase 4: Expand & Contract"
            print(f"\n{self.current_phase}")
            for expansion in range(4):
                current_radius = radius + (expansion % 2) * 20
                for i, agent in enumerate(self.agents.values()):
                    angle = (i * 2 * math.pi) / len(self.agents)
                    target_x = center[0] + current_radius * math.cos(angle)
                    target_y = center[1] + current_radius * math.sin(angle)
                    await agent.move_to(target_x, target_y)
                await asyncio.sleep(0.6)

            # Phase 5: Spiral outward
            self.current_phase = "Phase 5: Spiral Outward"
            print(f"\n{self.current_phase}")
            for spiral_step in range(6):
                current_radius = radius + spiral_step * 8
                for i, agent in enumerate(self.agents.values()):
                    angle = (i * 2 * math.pi) / len(self.agents) + (
                        spiral_step * math.pi / 6
                    )
                    target_x = center[0] + current_radius * math.cos(angle)
                    target_y = center[1] + current_radius * math.sin(angle)
                    # Clamp to world boundaries
                    target_x = max(10, min(self.world_size - 10, target_x))
                    target_y = max(10, min(self.world_size - 10, target_y))
                    await agent.move_to(target_x, target_y)
                await asyncio.sleep(0.5)

            await asyncio.sleep(1.0)
            self.current_phase = "Demo Complete!"
            print(f"\n{self.current_phase}")

        except Exception as e:
            self.logger.error(f"Movement pattern failed: {e}")
            raise

    async def shutdown(self) -> None:
        """Clean up demo resources."""
        self.logger.info("Shutting down visual spatial demo")
        await self.facade.shutdown()


async def run_simulation(demo: VisualSpatial2DDemo) -> None:
    """Run the simulation async task."""
    await demo.setup()
    await asyncio.sleep(1.0)  # Initial pause
    await demo.run_movement_patterns()
    await asyncio.sleep(2.0)  # Final pause
    demo.running = False


def main() -> None:
    """Run the visual 2D spatial simulation demo."""
    setup_logging("WARNING")  # Reduce log noise for visual demo

    print("🎬 Gunn Framework - Visual 2D Spatial Simulation")
    print("=" * 50)
    print("\n📊 Starting real-time visualization...")
    print("    Close the window to exit\n")

    demo = VisualSpatial2DDemo()

    # Initialize plot
    artists = demo.init_plot()

    # Create animation
    anim = FuncAnimation(  # noqa: F841
        demo.fig, demo.update_plot, init_func=lambda: artists, blit=True, interval=50
    )

    # Run simulation in background
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run_with_cleanup():
        try:
            await run_simulation(demo)
        finally:
            await demo.shutdown()

    import threading

    def run_async():
        loop.run_until_complete(run_with_cleanup())
        plt.close(demo.fig)

    thread = threading.Thread(target=run_async, daemon=True)
    thread.start()

    # Show plot (blocking)
    plt.show()

    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
