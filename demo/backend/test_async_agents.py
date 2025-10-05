"""
Test script for async agent execution in battle demo.

This script verifies that agents execute independently with their own
async loops, allowing for variable response times and asynchronous behavior.
"""

import asyncio
import logging
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


async def test_independent_agent_execution():
    """Test that agents execute independently with async loops."""
    from demo.backend.ai_decision import AIDecisionMaker
    from demo.backend.game_initializer import GameInitializer, InitializationConfig
    from demo.backend.gunn_integration import BattleOrchestrator

    logger.info("=== Testing Independent Async Agent Execution ===\n")

    # Initialize orchestrator
    orchestrator = BattleOrchestrator()

    # Initialize world state with agents using GameInitializer
    config = InitializationConfig(agents_per_team=2)
    initializer = GameInitializer(config=config)
    world_state = await initializer.initialize_game(orchestrator.world_state)

    # Create AI decision maker (will use OpenAI if API key available)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set - using fallback decision making")

    ai_decision_maker = AIDecisionMaker(api_key=api_key)

    # Initialize orchestrator with AI decision maker
    await orchestrator.initialize(ai_decision_maker)

    logger.info(f"Initialized {len(orchestrator._agent_tasks)} agent tasks")

    # Monitor agent execution for 10 seconds
    logger.info("\nMonitoring agent execution for 10 seconds...\n")

    start_time = asyncio.get_event_loop().time()
    last_stats_time = start_time

    try:
        while asyncio.get_event_loop().time() - start_time < 10.0:
            await asyncio.sleep(1.0)

            # Print stats every 2 seconds
            current_time = asyncio.get_event_loop().time()
            if current_time - last_stats_time >= 2.0:
                logger.info("\n--- Agent Statistics ---")

                # Get agent handles and check their stats
                for agent_id, agent in world_state.agents.items():
                    handle = orchestrator.orchestrator.agent_handles.get(agent_id)
                    if handle and hasattr(handle, "_logic"):
                        logic = handle._logic
                        if hasattr(logic, "get_stats"):
                            stats = logic.get_stats()
                            logger.info(
                                f"  {agent_id}: {stats['observations_processed']} obs, "
                                f"{stats['decisions_made']} decisions"
                            )

                # Check task states
                running_tasks = sum(
                    1 for task in orchestrator._agent_tasks if not task.done()
                )
                logger.info(f"\nRunning agent tasks: {running_tasks}/4\n")

                last_stats_time = current_time

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")

    finally:
        # Cleanup
        logger.info("\nCleaning up...")
        await orchestrator.reset()

        logger.info("\n=== Test Complete ===")


async def test_variable_response_times():
    """Test that agents can have variable response times."""
    logger.info("\n=== Testing Variable Response Times ===\n")

    from demo.backend.ai_decision import AIDecisionMaker
    from demo.backend.battle_agent import BattleAgent
    from demo.backend.game_initializer import GameInitializer, InitializationConfig
    from demo.shared.models import BattleWorldState

    # Create simple world state
    world_state = BattleWorldState()
    config = InitializationConfig(agents_per_team=1)
    initializer = GameInitializer(config=config)
    world_state = await initializer.initialize_game(world_state)

    # Create AI decision maker
    api_key = os.getenv("OPENAI_API_KEY")
    ai_decision_maker = AIDecisionMaker(api_key=api_key)

    # Create two battle agents
    agent_id = "team_a_agent_1"

    battle_agent = BattleAgent(
        agent_id=agent_id,
        ai_decision_maker=ai_decision_maker,
        world_state=world_state,
    )

    logger.info(f"Created BattleAgent: {agent_id}")

    # Test process_observation directly
    test_observation = {
        "agent_id": agent_id,
        "visible_entities": {
            agent_id: world_state.agents[agent_id].model_dump(),
        },
        "visible_relationships": {},
        "context_digest": "test_digest",
        "view_seq": 0,
    }

    logger.info("\nTesting direct observation processing...")

    # Time 3 consecutive observations
    times = []
    for i in range(3):
        start = asyncio.get_event_loop().time()
        intent = await battle_agent.process_observation(test_observation, agent_id)
        elapsed = asyncio.get_event_loop().time() - start
        times.append(elapsed)

        logger.info(f"  Observation {i + 1}: {elapsed:.3f}s - Intent: {intent['kind']}")

    avg_time = sum(times) / len(times)
    logger.info(f"\nAverage response time: {avg_time:.3f}s")

    # Get final stats
    stats = battle_agent.get_stats()
    logger.info(f"\nFinal stats: {stats}")

    logger.info("\n=== Variable Response Time Test Complete ===")


async def main():
    """Run all tests."""
    try:
        # Test 1: Independent agent execution
        await test_independent_agent_execution()

        # Test 2: Variable response times
        await test_variable_response_times()

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
