"""
Demonstration of the game initialization and auto-start system.

This script shows how to use the GameInitializer and AutoStartManager
to set up battle simulations with different configurations.
"""

import asyncio

from demo.backend.game_initializer import (
    AutoStartManager,
    GameInitializer,
    InitializationConfig,
)
from demo.shared.models import BattleWorldState


class MockOrchestrator:
    """Mock orchestrator for demonstration purposes."""

    def __init__(self):
        self.world_state = BattleWorldState()
        self._initialized = False

    async def initialize(self, ai_decision_maker):
        """Mock initialization."""
        print("  🔧 Orchestrator initialized with AI decision maker")
        self._initialized = True


class MockAIDecisionMaker:
    """Mock AI decision maker for demonstration purposes."""

    def __init__(self):
        print("  🤖 AI Decision Maker created")


def print_world_state_summary(world_state: BattleWorldState, title: str):
    """Print a summary of the world state."""
    print(f"\n📊 {title}")
    print(f"   Agents: {len(world_state.agents)}")
    print(f"   Map Locations: {len(world_state.map_locations)}")
    print(f"   Game Status: {world_state.game_status}")
    print(f"   Game Time: {world_state.game_time}")

    # Show team distribution
    team_counts = {}
    for agent in world_state.agents.values():
        team_counts[agent.team] = team_counts.get(agent.team, 0) + 1

    print(f"   Team Distribution: {team_counts}")

    # Show forge locations
    forges = [
        (loc_id, loc.position)
        for loc_id, loc in world_state.map_locations.items()
        if loc.location_type.value == "forge"
    ]
    print(f"   Forges: {forges}")


async def demo_basic_initialization():
    """Demonstrate basic game initialization."""
    print("\n🎮 Basic Game Initialization Demo")
    print("=" * 50)

    # Create default configuration
    config = InitializationConfig()
    initializer = GameInitializer(config)
    world_state = BattleWorldState()

    print("🔄 Initializing game with default settings...")
    await initializer.initialize_game(world_state)

    print_world_state_summary(world_state, "Default Initialization")

    # Show initialization summary
    summary = initializer.get_initialization_summary(world_state)
    print("\n📋 Initialization Summary:")
    print(f"   Initialization Count: {summary['initialization_count']}")
    print(f"   Config: {summary['config']}")


async def demo_custom_configurations():
    """Demonstrate different initialization configurations."""
    print("\n🎯 Custom Configuration Demo")
    print("=" * 50)

    configurations = [
        {
            "name": "Corner Battle",
            "config": InitializationConfig(
                agents_per_team=3,
                positioning_strategy="corners",
                forge_placement="corners",
                use_random_seed=True,
                random_seed=42,
            ),
        },
        {
            "name": "Line Formation",
            "config": InitializationConfig(
                agents_per_team=4,
                positioning_strategy="lines",
                forge_placement="sides",
                use_random_seed=True,
                random_seed=123,
            ),
        },
        {
            "name": "Random Chaos",
            "config": InitializationConfig(
                agents_per_team=2,
                positioning_strategy="random",
                forge_placement="center",
                position_jitter=3.0,
                use_random_seed=True,
                random_seed=999,
            ),
        },
    ]

    for setup in configurations:
        print(f"\n🎲 {setup['name']}")
        print("-" * 30)

        initializer = GameInitializer(setup["config"])
        world_state = BattleWorldState()

        await initializer.initialize_game(world_state)
        print_world_state_summary(world_state, setup["name"])

        # Show agent positions
        print("   Agent Positions:")
        for agent_id, agent in world_state.agents.items():
            print(f"     {agent_id}: {agent.position} (Team: {agent.team})")


async def demo_auto_start_manager():
    """Demonstrate auto-start manager functionality."""
    print("\n🚀 Auto-Start Manager Demo")
    print("=" * 50)

    # Create components
    config = InitializationConfig(
        agents_per_team=2,
        positioning_strategy="corners",
        use_random_seed=True,
        random_seed=42,
    )

    initializer = GameInitializer(config)
    auto_start_manager = AutoStartManager(initializer)

    # Mock components
    orchestrator = MockOrchestrator()
    ai_decision_maker = MockAIDecisionMaker()

    print("🔄 Starting auto-start process...")
    success = await auto_start_manager.auto_start_game(
        orchestrator, ai_decision_maker, max_attempts=3
    )

    if success:
        print("✅ Auto-start completed successfully!")
        print_world_state_summary(orchestrator.world_state, "Auto-Started Game")
    else:
        print("❌ Auto-start failed!")

    # Show auto-start statistics
    stats = auto_start_manager.get_restart_statistics()
    print("\n📈 Auto-Start Statistics:")
    print(f"   Restart Count: {stats['restart_count']}")
    print(f"   Auto-Start Enabled: {stats['auto_start_enabled']}")


async def demo_restart_functionality():
    """Demonstrate game restart functionality."""
    print("\n🔄 Restart Functionality Demo")
    print("=" * 50)

    config = InitializationConfig(agents_per_team=2)
    initializer = GameInitializer(config)
    auto_start_manager = AutoStartManager(initializer)

    orchestrator = MockOrchestrator()
    ai_decision_maker = MockAIDecisionMaker()

    # Initial start
    print("🎬 Initial game start...")
    success = await auto_start_manager.auto_start_game(orchestrator, ai_decision_maker)
    print(f"   Initial start: {'✅ Success' if success else '❌ Failed'}")

    # Simulate game progress
    orchestrator.world_state.game_time = 150.0
    orchestrator.world_state.game_status = "team_a_wins"
    print(
        f"   Simulated game progress: Time={orchestrator.world_state.game_time}, Status={orchestrator.world_state.game_status}"
    )

    # Restart game
    print("\n🔄 Restarting game...")
    success = await auto_start_manager.restart_game(
        orchestrator, ai_decision_maker, reason="demo_restart"
    )
    print(f"   Restart: {'✅ Success' if success else '❌ Failed'}")

    # Show reset state
    print_world_state_summary(orchestrator.world_state, "Restarted Game")

    # Show restart statistics
    stats = auto_start_manager.get_restart_statistics()
    print(f"\n📊 Restart Statistics: {stats}")


async def demo_deterministic_behavior():
    """Demonstrate deterministic initialization behavior."""
    print("\n🎯 Deterministic Behavior Demo")
    print("=" * 50)

    seed = 12345

    print(f"🌱 Using seed: {seed}")
    print("🔄 Running two identical initializations...")

    # First initialization
    config1 = InitializationConfig(
        agents_per_team=3,
        positioning_strategy="corners",
        forge_placement="corners",
        use_random_seed=True,
        random_seed=seed,
        position_jitter=2.0,  # Add some randomness to test determinism
    )

    initializer1 = GameInitializer(config1)
    world_state1 = BattleWorldState()
    await initializer1.initialize_game(world_state1)

    # Second initialization with same seed
    config2 = InitializationConfig(
        agents_per_team=3,
        positioning_strategy="corners",
        forge_placement="corners",
        use_random_seed=True,
        random_seed=seed,
        position_jitter=2.0,
    )

    initializer2 = GameInitializer(config2)
    world_state2 = BattleWorldState()
    await initializer2.initialize_game(world_state2)

    # Compare results
    print("\n🔍 Comparing results...")

    positions_match = True
    for agent_id in world_state1.agents:
        pos1 = world_state1.agents[agent_id].position
        pos2 = world_state2.agents[agent_id].position

        if abs(pos1[0] - pos2[0]) > 0.001 or abs(pos1[1] - pos2[1]) > 0.001:
            positions_match = False
            break

    print(f"   Agent positions identical: {'✅ Yes' if positions_match else '❌ No'}")
    print(
        f"   Agent count match: {'✅ Yes' if len(world_state1.agents) == len(world_state2.agents) else '❌ No'}"
    )
    print(
        f"   Map locations match: {'✅ Yes' if len(world_state1.map_locations) == len(world_state2.map_locations) else '❌ No'}"
    )

    if positions_match:
        print("🎉 Deterministic behavior confirmed!")
    else:
        print("⚠️  Deterministic behavior failed!")


async def main():
    """Run all demonstrations."""
    print("🎮 Game Initialization System Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive game initialization")
    print("and auto-start system for the battle simulation.")

    try:
        await demo_basic_initialization()
        await demo_custom_configurations()
        await demo_auto_start_manager()
        await demo_restart_functionality()
        await demo_deterministic_behavior()

        print("\n🎉 All demonstrations completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("   ✅ Configurable team and agent creation")
        print("   ✅ Multiple positioning strategies")
        print("   ✅ Strategic forge placement")
        print("   ✅ Auto-start with retry logic")
        print("   ✅ Game restart functionality")
        print("   ✅ Deterministic initialization")
        print("   ✅ Comprehensive validation")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
